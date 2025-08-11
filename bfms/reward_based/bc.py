from typing import Any

import distrax
import jax
import numpy as np
from flax import nnx
from jax import numpy as jnp


class ContinuousBC(nnx.Module):
    def __init__(self, dim_state: int, dim_action: int, dim_hidden: int = 256, *, rngs: nnx.Rngs):
        self._dim_state = dim_state
        self._dim_action = dim_action

        self.lin1 = nnx.Linear(dim_state, dim_hidden, rngs=rngs)
        self.lin2 = nnx.Linear(dim_hidden, dim_hidden, rngs=rngs)
        self.means = nnx.Linear(dim_hidden, dim_action, rngs=rngs)
        self.log_stds = nnx.Param(jnp.zeros((dim_action,)))

    def __call__(self, state: jax.Array) -> distrax.MultivariateNormalDiag:
        x = nnx.relu(self.lin1(state))
        x = nnx.relu(self.lin2(x))
        means = nnx.tanh(self.means(x))

        log_stds = self.log_stds.value
        log_stds = jnp.clip(log_stds, -5.0, 2.0)

        action_dist = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds))
        return action_dist


def make_train_step(dataset: dict[str, np.ndarray | jax.Array], batch_size: int):
    if isinstance(dataset["observation"], np.ndarray):
        dataset = jax.tree.map(jnp.asarray, dataset)
    dataset_size = len(dataset["reward"])

    def _train_step(
        train_state: tuple[jax.Array, Any, Any], _
    ) -> tuple[tuple[jax.Array, Any, Any], jax.Array]:
        rng, graph_def, state = train_state
        model, optimizer = nnx.merge(graph_def, state)

        # Sample batch.
        rng, rng_batch = jax.random.split(rng)
        batch_indexes = jax.random.randint(rng_batch, (batch_size,), 0, dataset_size)
        batch = jax.tree.map(lambda x: x[batch_indexes], dataset)

        def loss_fn(model: ContinuousBC):
            action_dist = model(batch["observation"])
            log_prob = action_dist.log_prob(batch["action"])
            return -log_prob.mean()

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)

        graph_def, state = nnx.split((model, optimizer))
        return (rng, graph_def, state), loss

    return _train_step


@nnx.jit
def train_step(
    model: ContinuousBC, optimizer: nnx.Optimizer, batch: dict[str, jax.Array]
) -> jax.Array:
    def loss_fn(model: ContinuousBC):
        action_dist = model(batch["observation"])
        log_prob = action_dist.log_prob(batch["action"])
        return -log_prob.mean()

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)

    return loss


@nnx.jit
def act(model: ContinuousBC, state: jax.Array, *, key: jax.Array) -> jax.Array:
    action_dist = model(state)
    return action_dist.mean()
