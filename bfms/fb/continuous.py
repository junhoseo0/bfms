import functools
import typing
from typing import Any, Literal, NamedTuple, overload

import distrax
import jax
import numpy as np
import optax
from flax import nnx
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from bfms.data import Dataset
from bfms.typing import Image


def _project_latent(latent: Float[Array, "*batch dim"]) -> Float[Array, "*batch dim"]:
    l2_norm: jax.Array = jnp.linalg.norm(latent, ord=2, axis=-1, keepdims=True)
    return jnp.sqrt(latent.shape[-1]) * latent / l2_norm


class ForwardModel(nnx.Module):
    def __init__(
        self,
        dim_state: int,
        dim_action: int,
        dim_latent: int,
        dim_hidden: int = 1024,
        num_hidden: int = 1,
        *,
        rngs: nnx.Rngs,
    ):
        self._dim_state = dim_state
        self._dim_action = dim_action
        self._dim_latent = dim_latent

        self._embed_s_a = nnx.Sequential(
            nnx.Linear(dim_state + dim_action, dim_hidden, rngs=rngs),
            # TODO: Fill in the reasoning about LayerNorm; if I remember
            # correctly, it was about scale-invarance to the reward.
            nnx.LayerNorm(dim_hidden, rngs=rngs),
            # TODO: Add reasoning about using tanh.
            nnx.tanh,
            nnx.Linear(dim_hidden, dim_hidden // 2, rngs=rngs),
            nnx.relu,
        )
        self._embed_s_z = nnx.Sequential(
            nnx.Linear(dim_state + dim_latent, dim_hidden, rngs=rngs),
            nnx.LayerNorm(dim_hidden, rngs=rngs),
            nnx.tanh,
            nnx.Linear(dim_hidden, dim_hidden // 2, rngs=rngs),
            nnx.relu,
        )

        seq_layers = []
        for _ in range(num_hidden):
            seq_layers.extend([nnx.Linear(dim_hidden, dim_hidden, rngs=rngs), nnx.relu])
        seq_layers.append(nnx.Linear(dim_hidden, dim_latent, rngs=rngs))
        self._embed_forward = nnx.Sequential(*seq_layers)

    def __call__(
        self,
        state: Float[Array, "*batch {self._dim_state}"],
        action: Float[Array, "*batch {self._dim_action}"],
        latent: Float[Array, "*batch {self._dim_latent}"],
    ) -> Float[Array, "*batch {self._dim_latent}"]:
        state_action = jnp.concat((state, action), axis=-1)
        state_action_emb = self._embed_s_a(state_action)
        state_latent = jnp.concat((state, latent), axis=-1)
        state_latent_emb = self._embed_s_z(state_latent)
        x = jnp.concat((state_action_emb, state_latent_emb), axis=-1)
        return self._embed_forward(x)


class BackwardModel(nnx.Module):
    def __init__(
        self,
        dim_state: int,
        dim_latent: int,
        dim_hidden: int = 256,
        num_hidden: int = 1,
        normalize_latent: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        self._dim_state = dim_state
        self._dim_latent = dim_latent
        self._normalize_latent = normalize_latent

        seq_layers = [
            nnx.Linear(dim_state, dim_hidden, rngs=rngs),
            nnx.LayerNorm(dim_hidden, rngs=rngs),
            nnx.tanh,
        ]
        for _ in range(num_hidden):
            seq_layers.extend([nnx.Linear(dim_hidden, dim_hidden, rngs=rngs), nnx.relu])
        seq_layers.append(nnx.Linear(dim_hidden, dim_latent, rngs=rngs))
        self._embed_backward = nnx.Sequential(*seq_layers)

    # TODO: Implement Backward model for (s, a).
    def __call__(
        self, state: Float[Array, "*batch {self._dim_state}"]
    ) -> Float[Array, "*batch {self._dim_latent}"]:
        backward_emb = self._embed_backward(state)
        # TODO: Make the normalization optional.
        return _project_latent(backward_emb)


class TruncatedMultivariateNormalDiag(distrax.Distribution):
    """
    Truncated normal distribution (https://en.wikipedia.org/wiki/Truncated_normal_distribution)
    has PDF of
    ```
    pdf(x; loc, scale, low, high) =
        { (2 pi)**(-0.5) exp(-0.5 y**2) / (scale * z) for low <= x <= high
        { 0 otherwise
    where
    y = (x - loc) / scale
    z = normal_cdf((high - loc) / scale) - normal_cdf((low - loc) /scale)
    ```

    Therefore, log-prob is
    ```
    log_prob(x; loc, scale, low, high) =
        -0.5 log(2 pi) - 0.5 y**2 - log(scale) - log(z)
    ```
    """

    def __init__(
        self, loc: jax.Array, scale_diag: jax.Array, low: float, high: float, clip: float = 0.0
    ):
        self._loc = loc
        self._scale_diag = scale_diag
        self._low = jnp.full_like(loc, low)
        self._high = jnp.full_like(loc, high)
        self._clip = clip

    def mean(self) -> jax.Array:
        return self._loc

    def log_prob(self, value: jax.Array) -> jax.Array:
        log_prob = -(
            0.5 * ((value - self._loc) / self._scale_diag) ** 2
            + 0.5 * jnp.log(2.0 * jnp.pi)
            + jnp.log(self._scale_diag)
            + self._normal_cdf_difference()
        )
        return log_prob

    def event_shape(self) -> tuple[int, ...]:
        return self._loc.shape

    def _sample_n(self, key: jax.Array, n: int) -> jax.Array:
        eps = jax.random.normal(key, shape=(n, *self._loc.shape))
        eps *= self._scale_diag
        eps = jnp.where(self._clip > 0, jnp.clip(eps, -self._clip, self._clip), eps)
        x = self._loc + eps
        return self._clip_with_grad(x)

    def _clip_with_grad(self, x: jax.Array) -> jax.Array:
        clipped_x = jnp.clip(x, self._low + 1e-6, self._high - 1e-6)
        x = x - jax.lax.stop_gradient(x) + jax.lax.stop_gradient(clipped_x)
        return x

    def _normal_cdf_difference(self):
        # TODO: We might use techniques used in TFP:
        # https://github.com/tensorflow/probability/blob/65f265c62bb1e2d15ef3e25104afb245a6d52429/tensorflow_probability/python/distributions/truncated_normal.py#L63
        standardized_low = (self._low - self._loc) / self._scale_diag
        standardized_high = (self._high - self._loc) / self._scale_diag
        log_cdf_low = jax.scipy.special.log_ndtr(standardized_low)
        log_cdf_high = jax.scipy.special.log_ndtr(standardized_high)
        return log_cdf_high + jnp.log1p(-jnp.exp(log_cdf_low - log_cdf_high))


class GaussianActor(nnx.Module):
    def __init__(
        self,
        action_low: float,
        action_high: float,
        dim_state: int,
        dim_action: int,
        dim_latent: int,
        dim_hidden: int = 1024,
        num_hidden: int = 1,
        *,
        rngs: nnx.Rngs,
    ):
        self._dim_state = dim_state
        self._dim_action = dim_action
        self._dim_latent = dim_latent

        self._action_low = action_low
        self._action_high = action_high

        self._embed_s = nnx.Sequential(
            nnx.Linear(dim_state, dim_hidden, rngs=rngs),
            nnx.LayerNorm(dim_hidden, rngs=rngs),
            nnx.tanh,
            nnx.Linear(dim_hidden, dim_hidden // 2, rngs=rngs),
            nnx.relu,
        )
        self._embed_s_z = nnx.Sequential(
            nnx.Linear(dim_state + dim_latent, dim_hidden, rngs=rngs),
            nnx.LayerNorm(dim_hidden, rngs=rngs),
            nnx.tanh,
            nnx.Linear(dim_hidden, dim_hidden // 2, rngs=rngs),
            nnx.relu,
        )

        seq_layers = []
        for _ in range(num_hidden):
            seq_layers.extend([nnx.Linear(dim_hidden, dim_hidden, rngs=rngs), nnx.relu])
        seq_layers.append(nnx.Linear(dim_hidden, dim_action, rngs=rngs))
        self._policy = nnx.Sequential(*seq_layers)

    def __call__(
        self,
        state: Float[Array, "*batch {self._dim_state}"],
        latent: Float[Array, "*batch {self._dim_latent}"],
        stddev: float,
        clip: float = 0.0,
    ) -> TruncatedMultivariateNormalDiag:
        state_emb = self._embed_s(state)
        state_latent = jnp.concat((state, latent), axis=-1)
        state_latent_emb = self._embed_s_z(state_latent)
        embedding = jnp.concat((state_emb, state_latent_emb), axis=-1)
        loc = nnx.tanh(self._policy(embedding))
        scale_diag = jnp.full_like(loc, stddev)
        action_dist = TruncatedMultivariateNormalDiag(
            loc, scale_diag, self._action_low, self._action_high, clip
        )
        return action_dist


class ForwardBackwardModel(nnx.Module):
    def __init__(
        self,
        dim_state: int,
        dim_action: int,
        dim_latent: int,
        *,
        rngs: nnx.Rngs,
    ):
        self._dim_state = dim_state
        self._dim_action = dim_action
        self._dim_latent = dim_latent

        # TODO: Replace with customizable number of ensembles, possibly with nnx.vmap.
        self.embed_forward1 = ForwardModel(dim_state, dim_action, dim_latent, rngs=rngs)
        self.embed_forward2 = ForwardModel(dim_state, dim_action, dim_latent, rngs=rngs)
        self.embed_backward = BackwardModel(dim_state, dim_latent, rngs=rngs)

    def __call__(
        self,
        state: Float[Array, "*batch {self._dim_state}"],
        action: Float[Array, "*batch {self._dim_action}"],
        latent: Float[Array, "*batch {self._dim_latent}"],
        future_state: Float[Array, "*batch {self._dim_state}"],
    ) -> tuple[
        Float[Array, "*batch {self._dim_latent}"],
        Float[Array, "*batch {self._dim_latent}"],
        Float[Array, "*batch {self._dim_latent}"],
    ]:
        forward_emb1 = self.embed_forward1(state, action, latent)
        forward_emb2 = self.embed_forward2(state, action, latent)
        backward_emb = self.embed_backward(future_state)
        return forward_emb1, forward_emb2, backward_emb


class TrainConfig(NamedTuple):
    actor_stddev: float
    discount: float
    dim_latent: int


class TrainState(NamedTuple):
    fb_model: ForwardBackwardModel
    target_fb_model: ForwardBackwardModel
    actor: GaussianActor
    fb_optimizer: nnx.Optimizer
    actor_optimizer: nnx.Optimizer
    key: PRNGKeyArray


class BatchJAX(NamedTuple):
    observation: Float[Array, "batch obs"]
    action: Float[Array, "batch action"]
    next_observation: Float[Array, "batch obs"]
    terminated: Bool[Array, " batch"]


def make_train_step(config: TrainConfig, batch_size: int, log_interval: int):
    @functools.partial(nnx.scan, length=log_interval)
    @functools.partial(nnx.jit, donate_argnums=(0, 1))
    def _train_step(
        train_state: TrainState, batch: BatchJAX
    ) -> tuple[TrainState, dict[str, Float[Array, "1"]]]:
        fb_model, target_fb_model, actor, fb_optimizer, actor_optimizer, key = train_state

        key, key_latent_ball = jax.random.split(key)
        # # Sample z's from uniformly from the Euclidean ball of radius \sqrt{d}
        # # (Touati et al., 2022, p.25), for each pair (s, a) to accelerate the
        # # training (Cetin et al., 2025, p.9).
        latent_ball = jax.random.normal(key_latent_ball, (batch_size, config.dim_latent))
        latent_ball = _project_latent(latent_ball)

        # # Sample z from B(s); this is equivalent to sampling random goals.
        key, key_latent_goal = jax.random.split(key)
        permuted_index = jax.random.permutation(key_latent_goal, batch_size)
        latent_goal = fb_model.embed_backward(batch.next_observation[permuted_index])

        # # Mix two latents.
        # # TODO: Move to option.
        key, key_mask = jax.random.split(key)
        random_mask = jax.random.uniform(key_mask, (batch_size, 1)) < 0.5
        latent = jnp.where(random_mask, latent_goal, latent_ball)
        # latent = batch["latent"]

        # Compute the target FB.
        # We can use multiple options for the future state, and the next
        # state is one version of them. TODO: Implement other options.
        key, key_next_action = jax.random.split(key)
        next_action_dist = actor(batch.next_observation, latent, config.actor_stddev, 0.3)
        next_action = next_action_dist.sample(seed=key_next_action)
        next_f_emb1, next_f_emb2, next_b_emb = target_fb_model(
            batch.next_observation, next_action, latent, batch.next_observation
        )
        # s: source (= batch size), t: target (= batch size), d: latent dimension.
        next_measures1 = jnp.einsum("sd,td->st", next_f_emb1, next_b_emb)
        next_measures2 = jnp.einsum("sd,td->st", next_f_emb2, next_b_emb)
        next_measures = (next_measures1 + next_measures2) / 2.0

        # Prepare key for sampling on-policy actions.
        key, key_action = jax.random.split(key)

        # There's a two approach to handling the loss function that incorporates
        # outputs from multiple model:
        # 1) Use separate loss functions for each model, with the hopes that
        # jax.jit optimizes computational graph to avoid re-compute
        # (ref.: https://github.com/google/flax/discussions/3316#discussioncomment-6968965)
        # 2) Add another module that includes all the required model
        # (ref.: https://github.com/google/flax/blob/main/examples/nnx_toy_examples/05_vae.py)
        # TODO: Write about reasoning to choose the option 2).
        def fb_loss_fn(fb_model: ForwardBackwardModel) -> tuple[jax.Array, dict[str, jax.Array]]:
            f_emb1, f_emb2, b_emb = fb_model(
                batch.observation, batch.action, latent, batch.next_observation
            )
            measures1 = jnp.einsum("sd,td->st", f_emb1, b_emb)
            measures2 = jnp.einsum("sd,td->st", f_emb2, b_emb)

            # Compute FB loss.
            off_diag = ~jnp.eye(batch_size, batch_size, dtype=jnp.bool)
            diff_measures1 = measures1 - config.discount * next_measures
            diff_measures2 = measures2 - config.discount * next_measures
            loss_fb_off_diag = 0.5 * (
                jnp.where(off_diag, diff_measures1**2, 0.0).sum() / off_diag.sum()
                + jnp.where(off_diag, diff_measures2**2, 0.0).sum() / off_diag.sum()
            )
            loss_fb_diag = -(diff_measures1.diagonal().mean() + diff_measures2.diagonal().mean())
            loss_fb = loss_fb_off_diag + loss_fb_diag

            # Compute orthogonal regularization.
            # NOTE: There are two versions of this regularization, one involing
            # stop-gradients operator (Touati & Ollivier, 2021, p.14) and the
            # without it (Cetin et al., 2024, p.28). This version implements the
            # second one. TODO: Make it optional.
            b_gram = jnp.matmul(b_emb, b_emb.T)  # (B, B)
            loss_orthonormal_off_diag = (
                0.5 * jnp.where(off_diag, b_gram**2, 0.0).sum() / off_diag.sum()
            )
            loss_orthonormal_diag = -b_gram.diagonal().mean()
            loss_orthonormal = loss_orthonormal_off_diag + loss_orthonormal_diag

            loss = loss_fb + loss_orthonormal
            return loss, {
                "loss": loss,
                "loss_fb": loss_fb,
                "loss_fb_diag": loss_fb_diag,
                "loss_fb_off_diag": loss_fb_off_diag,
                "loss_orthonormal": loss_orthonormal,
                "loss_orthonormal_diag": loss_orthonormal_diag,
                "loss_orthonormal_off_diag": loss_orthonormal_off_diag,
                "measure1": measures1.mean(),
                "next_measure": next_measures.mean(),
                "f_emb1": f_emb1.mean(),
                "b_emb": b_emb.mean(),
                "b_norm": jnp.linalg.norm(b_emb, ord=2, axis=-1).mean(),
                "z_norm": jnp.linalg.norm(latent, ord=2, axis=-1).mean(),
            }

        def actor_loss_fn(
            actor: GaussianActor, fb_model: ForwardBackwardModel
        ) -> tuple[jax.Array, dict[str, jax.Array]]:
            # Compute actor loss.
            # TODO: Implement improved weighted importance sampling (IWIS)
            action_dist = actor(batch.observation, latent, config.actor_stddev, 0.3)
            action = action_dist.sample(seed=key_action)

            f_emb1 = fb_model.embed_forward1(batch.observation, action, latent)
            f_emb2 = fb_model.embed_forward2(batch.observation, action, latent)
            q1 = (f_emb1 * latent).sum(axis=-1)
            q2 = (f_emb2 * latent).sum(axis=-1)

            # Compute the uncertainty.
            qs = jnp.concat([jnp.expand_dims(q1, 0), jnp.expand_dims(q2, 0)], axis=0)  # (2, B)
            qs_left = jnp.expand_dims(qs, axis=0)
            qs_right = jnp.expand_dims(qs, axis=1)
            qs_diffs = jnp.abs(qs_left - qs_right)
            q_uncertainty = qs_diffs.sum(axis=(0, 1)) / 2

            # Compute uncertainty-penalized Q.
            q = qs.mean(axis=0) - 0.5 * q_uncertainty  # B
            loss = -q.mean()
            return loss, {"loss_actor": loss, "q": q.mean(), "q_uncertainty": q_uncertainty.mean()}

        (_, info_fb), grad = nnx.value_and_grad(fb_loss_fn, has_aux=True)(fb_model)
        fb_optimizer.update(fb_model, grad)

        f1_grad_norm = optax.global_norm(grad["embed_forward1"])
        f2_grad_norm = optax.global_norm(grad["embed_forward2"])
        b_grad_norm = optax.global_norm(grad["embed_backward"])

        (_, info_actor), grad = nnx.value_and_grad(actor_loss_fn, argnums=0, has_aux=True)(
            actor, fb_model
        )
        actor_optimizer.update(actor, grad)
        actor_grad_norm = optax.global_norm(grad)

        # Perform soft-update on the target networks.
        fb_params = nnx.state(fb_model, nnx.Param)
        target_fb_params = nnx.state(target_fb_model, nnx.Param)
        new_target_fb_params = jax.tree.map(
            lambda p, g: (0.99 * p) + (0.01 * g), target_fb_params, fb_params
        )
        nnx.update(target_fb_model, new_target_fb_params)

        return TrainState(
            fb_model, target_fb_model, actor, fb_optimizer, actor_optimizer, key
        ), info_fb | info_actor | {
            "f1_global_norm": f1_grad_norm,
            "f2_global_norm": f2_grad_norm,
            "b_global_norm": b_grad_norm,
            "actor_grad_norm": actor_grad_norm,
        }

    return _train_step


@nnx.jit
def act(
    actor: GaussianActor,
    state: Float[Array, "*batch {actor._dim_state}"],
    latent: Float[Array, "*batch {actor._dim_latent}"],
):
    action_dist = actor(state, latent, 0.0)
    return action_dist.mean()


@nnx.jit
def infer_latent(
    fb_model: ForwardBackwardModel,
    future_state: Float[Array, "*batch {fb_model.forward_emb1._dim_state}"],
    reward: Float[Array, "*batch"],
) -> Float[Array, "*batch {fb_model.forward_emb1._dim_latent}"]:
    b_emb = fb_model.embed_backward(future_state)
    reward = jnp.expand_dims(reward, axis=1)
    latent = jnp.matmul(reward.T, b_emb).squeeze(0)
    latent = _project_latent(latent)
    return latent


@overload
def evaluate(
    actor: GaussianActor,
    fb_model: ForwardBackwardModel,
    dataset: Dataset,
    num_inference_samples: int,
    tasks: list[str] | None = None,
    render: Literal[True] = True,
) -> dict[str, dict[str, float] | dict[str, list[Image]]]: ...


@overload
def evaluate(
    actor: GaussianActor,
    fb_model: ForwardBackwardModel,
    dataset: Dataset,
    num_inference_samples: int,
    tasks: list[str] | None = None,
    render: Literal[False] = False,
) -> dict[str, dict[str, float]]: ...


@overload
def evaluate(
    actor: GaussianActor,
    fb_model: ForwardBackwardModel,
    dataset: Dataset,
    num_inference_samples: int,
    tasks: list[str] | None = None,
) -> dict[str, dict[str, float]]: ...


def evaluate(
    actor: GaussianActor,
    fb_model: ForwardBackwardModel,
    dataset: Dataset,
    num_inference_samples: int,
    tasks: list[str] | None = None,
    render: bool = False,
) -> dict[str, dict[str, float]] | dict[str, dict[str, float] | dict[str, list[Image]]]:
    if tasks is None:
        tasks = dataset.tasks

    result = {"score": {}}
    if render:
        result["render"] = {}

    for task in tasks:
        env = dataset.recover_environment(task)

        # Infer latents.
        batch = dataset.sample(num_inference_samples, with_reward=True)
        latent = infer_latent(
            fb_model, jnp.asarray(batch.next_observation), jnp.asarray(batch.reward[task])
        )

        score = 0.0
        frames = []
        observation, _ = env.reset()
        terminated, truncated = False, False
        while not (terminated or truncated):
            if render:
                frame = env.render()
                frames.append(frame)

            action = act(actor, observation, latent)
            observation, reward, terminated, truncated, _ = env.step(np.asarray(action))
            score += float(reward)

        if render:
            frame = env.render()
            frames.append(frame)

        result["score"][task] = score
        if render:
            result["render"][task] = frames

    return result
