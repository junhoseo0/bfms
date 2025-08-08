import jax
import numpy as np
import optax
import tyro
from flax import nnx
from jax import numpy as jnp

from bfms.data import DatasetFactory
from bfms.reward_based import bc


def main(
    dataset_id: str,
    task_id: str,
    dataset_dir: str | None = None,
    seed: int = 294,
    train_step: int = int(1e6),
    batch_size: int = 256,
):
    dataset = DatasetFactory.create(dataset_id, dataset_dir)
    env = dataset.recover_environment(task_id)
    print(f"Dataset {dataset_id} loaded with {len(dataset):,} transitions.")

    key = jax.random.PRNGKey(seed=seed)
    key, nnx_rngs = jax.random.split(key)
    nnx_rngs = nnx.Rngs(nnx_rngs)
    dataset.seed(seed)

    dim_state = dataset["observation"].shape[-1]
    dim_action = dataset["action"].shape[-1]
    continuous_bc = bc.ContinuousBC(dim_state, dim_action, rngs=nnx_rngs)
    optimizer = nnx.Optimizer(continuous_bc, optax.adam(6e-5))

    for t in range(train_step):
        batch = dataset.sample(batch_size)
        batch = jax.tree.map(lambda x: jnp.asarray(x), batch)
        loss = bc.train_step(continuous_bc, optimizer, batch)
        if t % 100 == 0:
            print(f"{t}: loss={loss.item()}")

        if t % 1000 == 0:
            score = 0.0
            done = False
            observation, _ = env.reset()
            while not done:
                key, action_key = jax.random.split(key)
                action = bc.act(continuous_bc, observation, key=action_key)
                observation, reward, terminated, truncated, _ = env.step(np.asarray(action))
                done = terminated or truncated
                score += float(reward)

    print(f"Return on {dataset_id}: {score}")


if __name__ == "__main__":
    tyro.cli(main)
