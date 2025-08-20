import typing
from typing import NamedTuple

import gymnasium as gym
import jax
import numpy as np
import optax
from flax import nnx
from jax import numpy as jnp
from pyparsing import Any

from bfms.data import MuJoCoDataset


def _project_latent(x: jax.Array) -> jax.Array:
    dim_latent = x.shape[-1]
    x = jnp.sqrt(dim_latent) * x / jnp.linalg.norm(x, ord=2, axis=-1, keepdims=True)
    return x


class ForwardModel(nnx.Module):
    def __init__(
        self,
        num_states: int,
        num_actions: int,
        dim_latent: int,
        dim_hidden: int = 1024,
        num_hidden: int = 1,
        *,
        rngs: nnx.Rngs,
    ):
        self._num_actions = num_actions

        self._embed_s = nnx.Sequential(
            nnx.Embed(num_states, dim_hidden, rngs=rngs),
            nnx.LayerNorm(dim_hidden, rngs=rngs),
            nnx.tanh,
            nnx.Linear(dim_hidden, dim_hidden // 2, rngs=rngs),
            nnx.relu,
        )
        self._embed_z = nnx.Sequential(
            nnx.Linear(dim_latent, dim_hidden, rngs=rngs),
            nnx.LayerNorm(dim_hidden, rngs=rngs),
            nnx.tanh,
            nnx.Linear(dim_hidden, dim_hidden // 2, rngs=rngs),
            nnx.relu,
        )

        seq_layers = []
        for _ in range(num_hidden):
            seq_layers.extend([nnx.Linear(dim_hidden, dim_hidden, rngs=rngs), nnx.relu])
        seq_layers.append(nnx.LinearGeneral(dim_hidden, (num_actions, dim_latent), rngs=rngs))
        self._embed_forward = nnx.Sequential(*seq_layers)

    def __call__(self, state: jax.Array, latent: jax.Array) -> jax.Array:
        s_emb = self._embed_s(state)
        z_emb = self._embed_z(latent)
        return self._embed_forward(jnp.concat([s_emb, z_emb], axis=-1))


class BackwardModel(nnx.Module):
    def __init__(
        self,
        num_states: int,
        dim_latent: int,
        dim_hidden: int = 256,
        num_hidden: int = 1,
        *,
        rngs: nnx.Rngs,
    ):
        self._embed_state = nnx.Sequential(
            nnx.Embed(num_states, dim_hidden, rngs=rngs),
            nnx.LayerNorm(dim_hidden, rngs=rngs),
            nnx.tanh,
        )

        seq_layers = []
        for _ in range(num_hidden):
            seq_layers.extend([nnx.Linear(dim_hidden, dim_hidden, rngs=rngs), nnx.relu])
        seq_layers.append(nnx.Linear(dim_hidden, dim_latent, rngs=rngs))
        self._embed_backward = nnx.Sequential(*seq_layers)

    def __call__(self, state: jax.Array) -> jax.Array:
        state_emb = self._embed_state(state)
        backward_emb = self._embed_backward(state_emb)
        return _project_latent(backward_emb)


class TrainConfig(NamedTuple):
    discount: float
    dim_latent: int


def make_train_step(config: TrainConfig, batch_size: int):
    def _train_step(train_state: tuple[jax.Array, Any, Any], batch: dict[str, jax.Array]):
        key, graph_def, state = train_state
        (
            forward_model,
            backward_model,
            target_forward_model,
            target_backward_model,
            forward_optimizer,
            backward_optimizer,
        ) = nnx.merge(graph_def, state)

        forward_model = typing.cast(ForwardModel, forward_model)
        backward_model = typing.cast(BackwardModel, backward_model)
        target_forward_model = typing.cast(ForwardModel, target_forward_model)
        target_backward_model = typing.cast(BackwardModel, target_backward_model)
        forward_optimizer = typing.cast(nnx.Optimizer, forward_optimizer)
        backward_optimizer = typing.cast(nnx.Optimizer, backward_optimizer)

        key, key_latent_ball = jax.random.split(key)
        # Sample z's from uniformly from the Euclidean ball of radius \sqrt{d}
        # (Touati et al., 2022, p.25), for each pair (s, a) to accelerate the
        # training (Cetin et al., 2025, p.9).
        latent_ball = jax.random.normal(key_latent_ball, (batch_size, config.dim_latent))
        latent_ball = _project_latent(latent_ball)

        # Sample z from B(s); this is equivalent to sampling random goals.
        key, key_latent_goal = jax.random.split(key)
        permuted_index = jax.random.permutation(key_latent_goal, batch_size)
        latent_goal = backward_model(batch["next_observation"][permuted_index])

        # Mix two latents.
        key, key_mask = jax.random.split(key)
        # TODO: Move to option.
        random_mask = jax.random.uniform(key_mask, (batch_size, 1)) < 0.5
        # (B, D)
        latent = jnp.where(random_mask, latent_goal, latent_ball)

        # Compute the target FB.
        # We can use multiple options for the future state, and the next
        # state is one version of them. TODO: Implement other options.
        key, key_next_action = jax.random.split(key)
        next_forward_emb = target_forward_model(batch["next_observation"], latent)
        next_q = jnp.einsum("bad,bd->ba", next_forward_emb, latent)
        next_action = jax.random.categorical(key_next_action, next_q)
        # next_action = jnp.argmax(next_q, axis=-1)

        next_forward_emb = next_forward_emb[jnp.arange(batch_size), next_action]
        target_backward_emb = target_backward_model(batch["next_observation"])
        target_measure = jnp.einsum("sd,td->st", next_forward_emb, target_backward_emb)

        def forward_loss_fn(forward_model: ForwardModel, backward_model: BackwardModel):
            forward_emb = forward_model(batch["observation"], latent)
            forward_emb = forward_emb[jnp.arange(batch_size), batch["action"]]
            backward_emb = backward_model(batch["next_observation"])
            measure = jnp.einsum("sd,td->st", forward_emb, backward_emb)

            off_diag = ~jnp.eye(batch_size, batch_size, dtype=jnp.bool_)
            diff_measure = measure - config.discount * (1.0 - batch["terminated"]) * target_measure
            loss_fb_diag = -diff_measure.diagonal().mean()
            loss_fb_off_diag = (
                0.5 * jnp.where(off_diag, diff_measure**2, 0.0).sum() / off_diag.sum()
            )
            loss_fb = loss_fb_off_diag + loss_fb_diag
            return loss_fb, {
                "loss_fb": loss_fb,
                "loss_fb_diag": loss_fb_diag,
                "loss_fb_off_diag": loss_fb_off_diag,
            }

        def backward_loss_fn(backward_model: BackwardModel, forward_model: ForwardModel):
            # loss_fb, _ = forward_loss_fn(forward_model, backward_model)

            forward_emb = forward_model(batch["observation"], latent)
            forward_emb = forward_emb[jnp.arange(batch_size), batch["action"]]
            backward_emb = backward_model(batch["next_observation"])
            measure = jnp.einsum("sd,td->st", forward_emb, backward_emb)

            off_diag = ~jnp.eye(batch_size, batch_size, dtype=jnp.bool_)
            diff_measure = measure - config.discount * (1.0 - batch["terminated"]) * target_measure
            loss_fb_diag = -diff_measure.diagonal().mean()
            loss_fb_off_diag = (
                0.5 * jnp.where(off_diag, diff_measure**2, 0.0).sum() / off_diag.sum()
            )
            loss_fb = loss_fb_off_diag + loss_fb_diag

            off_diag = ~jnp.eye(batch_size, batch_size, dtype=jnp.bool_)
            backward_emb = backward_model(batch["next_observation"])
            backward_gram_matrix = jnp.matmul(backward_emb, backward_emb.T)
            loss_orthonormal_diag = -backward_gram_matrix.diagonal().mean()
            loss_orthonormal_off_diag = (
                0.5 * jnp.where(off_diag, backward_gram_matrix**2, 0.0).sum() / off_diag.sum()
            )
            loss_orthonormal = loss_orthonormal_diag + loss_orthonormal_off_diag

            loss_backward = loss_fb + loss_orthonormal
            return loss_backward, {
                "loss_backward": loss_backward,
                "loss_orthonormal": loss_orthonormal,
                "loss_orthonormal_diag": loss_orthonormal_diag,
                "loss_orthonormal_off_diag": loss_orthonormal_off_diag,
            }

        (_, info_forward), grad_forward = nnx.value_and_grad(
            forward_loss_fn, argnums=0, has_aux=True
        )(forward_model, backward_model)
        forward_optimizer.update(grad_forward)

        (_, info_backward), grad_backward = nnx.value_and_grad(
            backward_loss_fn, argnums=0, has_aux=True
        )(backward_model, forward_model)
        backward_optimizer.update(grad_backward)

        graphdef, forward_params = nnx.split(forward_model)
        _, target_forward_params = nnx.split(target_forward_model)
        target_forward_params = optax.incremental_update(
            forward_params, target_forward_params, 0.01
        )
        target_forward_model = nnx.merge(graphdef, target_forward_params)

        graphdef, backward_params = nnx.split(backward_model)
        _, target_backward_params = nnx.split(target_backward_model)
        target_backward_params = optax.incremental_update(
            backward_params, target_backward_params, 0.01
        )
        target_backward_model = nnx.merge(graphdef, target_backward_params)

        graph_def, state = nnx.split(
            (
                forward_model,
                backward_model,
                target_forward_model,
                target_backward_model,
                forward_optimizer,
                backward_optimizer,
            )
        )
        return (key, graph_def, state), info_forward | info_backward

    return _train_step


@nnx.jit
def act(forward_model: ForwardModel, state: jax.Array, latent: jax.Array) -> jax.Array:
    forward_emb = forward_model(state, latent)  # (A, D)
    q = jnp.dot(forward_emb, latent)  # (A,)
    return jnp.argmax(q, axis=0)


@nnx.jit
def infer_latent(
    backward_model: BackwardModel, next_state: jax.Array, reward: jax.Array
) -> jax.Array:
    backward_emb = backward_model(next_state)  # (B, D)
    reward = jnp.expand_dims(reward, axis=1)  # (B, 1)
    z = (backward_emb * reward).sum(axis=0)  # (D,)
    return _project_latent(z)


def evaluate(
    forward_model: ForwardModel,
    backward_model: BackwardModel,
    dataset: MuJoCoDataset,
    task_id: str,
    num_inference_samples: int,
    render: bool = False,
) -> tuple[float, float] | tuple[float, float, list[np.ndarray]]:
    # env = dataset.recover_environment(task_id)
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
    batch = dataset.sample_with_rewards(task_id, num_inference_samples)
    batch = jax.tree.map(jnp.asarray, batch)
    latent = infer_latent(backward_model, batch["next_observation"], batch["reward"])

    score = 0.0
    done = False
    frames = []
    observation, _ = env.reset()

    # Estimate Q.
    action = act(forward_model, jnp.asarray(observation), latent)
    forward_emb = forward_model(jnp.asarray(observation), latent)  # (D, A)
    forward_emb = forward_emb[action]  # (D,)
    q = jnp.dot(forward_emb, latent)
    print(f"estimated {q=}")

    while not done:
        if render:
            frame = env.render()
            frames.append(frame)

        with jax.disable_jit():
            action = act(forward_model, jnp.asarray(observation), latent)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        score += float(reward)

    if render:
        frame = env.render()
        frames.append(frame)
        return score, score, frames

    return score, score
