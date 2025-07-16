import distrax
import jax
from flax import nnx
from jax import numpy as jnp


class ContinuousBC(nnx.Module):
    def __init__(self, dim_state: int, dim_action: int, dim_hidden: int = 256, *, rngs: nnx.Rngs):
        self._dim_state = dim_state
        self._dim_action = dim_action

        self.lin1 = nnx.Linear(dim_state, dim_hidden, rngs=rngs)
        self.lin2 = nnx.Linear(dim_hidden, dim_hidden, rngs=rngs)
        self.means = nnx.Linear(dim_hidden, dim_action, rngs=rngs)
        self.log_stds = nnx.Variable(jnp.zeros(shape=(dim_action,)))

    def __call__(self, state: jax.Array) -> distrax.MultivariateNormalDiag:
        x = nnx.relu(self.lin1(state))
        x = nnx.relu(self.lin2(x))
        means = self.means(x)
        log_stds = self.log_stds.raw_value
        action_dist = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds))
        return action_dist


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
    return action_dist.sample(seed=key)
