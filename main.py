import jax
import numpy as np
import optax
import tyro
from dm_control import suite
from flax import nnx
from jax import numpy as jnp

from bfms.data import DATASET_REGISTRY, D4RLDataset, ExoRLDataset
from bfms.reward_based import bc


def main(
    dataset_dir: str,
    dataset_id: str,
    eval_task: str,
    seed: int = 294,
    train_step: int = int(1e6),
    batch_size: int = 256,
):
    if DATASET_REGISTRY[dataset_id] == "D4RL":
        _group, _env, _data_quality = dataset_id.split("/")
        dataset = D4RLDataset(_group, _env, _data_quality)
    elif DATASET_REGISTRY[dataset_id] == "ExoRL":
        _domain, _collection_method = dataset_id.split("/")
        dataset = ExoRLDataset(dataset_dir, _domain, _collection_method)
    print(f"Dataset {dataset_id} loaded with {len(dataset):,} transitions.")

    key = jax.random.PRNGKey(seed=seed)
    key, nnx_rngs = jax.random.split(key)
    nnx_rngs = nnx.Rngs(nnx_rngs)
    dataset.seed(seed)

    dim_state = dataset["observation"].shape[-1]
    dim_action = dataset["action"].shape[-1]
    continuous_bc = bc.ContinuousBC(dim_state, dim_action, rngs=nnx_rngs)
    optimizer = nnx.Optimizer(continuous_bc, optax.adam(1e-3))

    for t in range(train_step):
        batch = dataset.sample(batch_size)
        batch = jax.tree.map(lambda x: jnp.asarray(x), batch)
        loss = bc.train_step(continuous_bc, optimizer, batch)
        if t % 100 == 0:
            print(f"{t}: loss={loss.item()}")

    score = 0.0
    if DATASET_REGISTRY[dataset_id] == "D4RL":
        done = False
        env = dataset._raw_dataset.recover_environment()
        observation, _ = env.reset()
        while not done:
            key, action_key = jax.random.split(key)
            action = bc.act(continuous_bc, observation, key=action_key)
            observation, reward, terminated, truncated, _ = env.step(np.asarray(action))
            done = terminated or truncated
            score += float(reward)
    elif DATASET_REGISTRY[dataset_id] == "ExoRL":
        _domain, _collection_method = dataset_id.split("/")
        env = suite.load(domain_name=_domain, task_name=eval_task)
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
    print(f"Return on {dataset_id}: {score}")


if __name__ == "__main__":
    tyro.cli(main)
