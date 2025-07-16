import jax
import optax
import tyro
from dm_control import suite
from flax import nnx
from jax import numpy as jnp

from bfms.data import ExoRLDataset
from bfms.reward_based import bc


def main(
    dataset_dir: str,
    domain: str,
    collection_method: str,
    eval_task: str,
    seed: int = 294,
    train_step: int = int(1e6),
    batch_size: int = 256,
):
    exorl_dataset = ExoRLDataset(dataset_dir, domain, collection_method)
    print(f"Dataset {domain}/{collection_method} loaded with {len(exorl_dataset):,} transitions.")

    key = jax.random.PRNGKey(seed=seed)
    key, nnx_rngs = jax.random.split(key)
    nnx_rngs = nnx.Rngs(nnx_rngs)
    exorl_dataset.seed(seed)

    dim_state = exorl_dataset["observation"].shape[-1]
    dim_action = exorl_dataset["action"].shape[-1]
    continuous_bc = bc.ContinuousBC(dim_state, dim_action, rngs=nnx_rngs)
    optimizer = nnx.Optimizer(continuous_bc, optax.adam(1e-3))

    for t in range(train_step):
        batch = exorl_dataset.sample(batch_size)
        batch = jax.tree.map(lambda x: jnp.asarray(x), batch)
        loss = bc.train_step(continuous_bc, optimizer, batch)
        if t % 100 == 0:
            print(f"{t}: loss={loss.item()}")

    score = 0.0
    env = suite.load(domain_name=domain, task_name=eval_task)
    timestep = env.reset()
    while not timestep.last():
        key, action_key = jax.random.split(key)
        observation = jnp.concat(
            [
                timestep.observation["orientations"],
                jnp.array([timestep.observation["height"]]),
                timestep.observation["velocity"],
            ]
        )

        action = bc.act(continuous_bc, observation, key=action_key)
        timestep = env.step(action)
        score += timestep.reward
    print(f"Return on {domain}-{eval_task}: {score}")


if __name__ == "__main__":
    tyro.cli(main)
