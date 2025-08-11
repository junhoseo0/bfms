import datetime
import os
import time

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
    train_time_step: int = int(1e6),
    batch_size: int = 256,
    device_num: int = 0,
    log_interval: int = 10_000,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_num)

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
    optimizer = nnx.Optimizer(continuous_bc, optax.adam(3e-4))
    graph_def, state = nnx.split((continuous_bc, optimizer))
    train_step = bc.make_train_step(dataset.unwrapped(), batch_size)

    s_time = time.perf_counter()
    key, train_key = jax.random.split(key)
    for t in range(0, train_time_step, log_interval):
        (train_key, graph_def, state), losses = jax.lax.scan(
            train_step, (train_key, graph_def, state), None, log_interval
        )

        elapsed_time = time.perf_counter() - s_time
        expected_time = (train_time_step // log_interval) * elapsed_time

        print(
            f"{t}: {elapsed_time / log_interval:.3f}s/update",
            f"(interval: {elapsed_time:.2f}s / expected total: {datetime.timedelta(seconds=expected_time)})",
        )
        s_time = time.perf_counter()

        print(f"{t}: loss={losses[-1].item()}")

        continuous_bc, _ = nnx.merge(graph_def, state)
        score = 0.0
        done = False
        observation, _ = env.reset()
        while not done:
            key, action_key = jax.random.split(key)
            action = bc.act(continuous_bc, observation, key=action_key)
            observation, reward, terminated, truncated, _ = env.step(np.asarray(action))
            done = terminated or truncated
            score += float(reward)

        normalized_score = dataset.normalize_return(score)
        print(f"Return {normalized_score:.2f} ({score:.2f})")


if __name__ == "__main__":
    tyro.cli(main)
