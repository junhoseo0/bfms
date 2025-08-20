import os

# Headless rendering with EXT_platform_device.
os.environ["MUJOCO_GL"] = "egl"

import datetime
import time

import jax
import optax
import tyro
from flax import nnx
from imageio import v3 as iio
from jax import numpy as jnp

from bfms.data import DatasetFactory
from bfms.fb import continuous as fb_awr


def main(
    dataset_id: str,
    task_id: str,
    dataset_dir: str | None = None,
    seed: int = 294,
    train_time_step: int = int(1e6),
    batch_size: int = 1024,
    num_inference_samples: int = 50_000,
    device_num: int = 0,
    log_interval: int = 10_000,
    eval_interval: int = 100_000,
):
    # Set GPU for JAX to use.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_num)
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(device_num)

    dataset = DatasetFactory.create(dataset_id, dataset_dir)
    print(f"Dataset {dataset_id} loaded with {len(dataset):,} transitions.")

    key = jax.random.PRNGKey(seed=seed)
    key, nnx_rngs = jax.random.split(key)
    nnx_rngs = nnx.Rngs(nnx_rngs)
    dataset.seed(seed)

    dim_state = dataset["observation"].shape[-1]
    dim_action = dataset["action"].shape[-1]
    # TODO: Move this to config.
    dim_latent = 100

    fb_model = fb_awr.ForwardBackwardModel(dim_state, dim_action, dim_latent, rngs=nnx_rngs)
    fb_optimizer = nnx.Optimizer(fb_model, optax.adam(1e-4), wrt=nnx.Param)

    actor = fb_awr.GaussianActor(-1.0, 1.0, dim_state, dim_action, dim_latent, rngs=nnx_rngs)
    actor_optimizer = nnx.Optimizer(actor, optax.adam(1e-4), wrt=nnx.Param)

    fb_model_target = nnx.clone(fb_model)
    graph_def, state = nnx.split((fb_model, fb_model_target, actor, fb_optimizer, actor_optimizer))
    train_step = fb_awr.make_train_step(
        fb_awr.TrainConfig(actor_stddev=0.2, discount=0.98, dim_latent=dim_latent),
        batch_size,
    )

    key, key_eval = jax.random.split(key)
    normalized_score, score, frames = fb_awr.evaluate(
        actor, fb_model, dataset, task_id, num_inference_samples, render=True
    )
    iio.imwrite("0.gif", frames)
    print(f"Return {normalized_score:.2f} ({score:.2f})")

    s_time = time.perf_counter()
    key, key_train = jax.random.split(key)
    for t in range(0, train_time_step, log_interval):
        # Pre-sample batches to reduce VRAM load.
        batches = dataset.sample(log_interval * batch_size)
        batches = jax.tree.map(
            lambda x: jnp.asarray(x).reshape(log_interval, batch_size, -1), batches
        )
        (key_train, graph_def, state), infos = jax.block_until_ready(
            jax.lax.scan(train_step, (key_train, graph_def, state), batches, log_interval)
        )

        elapsed_time = time.perf_counter() - s_time
        expected_time = (train_time_step // log_interval) * elapsed_time

        print(
            f"{t + log_interval}: {elapsed_time / log_interval:.3f}s/update",
            f"(interval: {elapsed_time:.2f}s / expected total: {datetime.timedelta(seconds=expected_time)})",
        )
        s_time = time.perf_counter()

        last_info = jax.tree.map(lambda x: x[-1], infos)
        print(f"{t + log_interval}: {last_info}")

        # continuous_bc, _ = nnx.merge(graph_def, state)
        if (t + log_interval) % eval_interval == 0 or t == train_time_step - log_interval:
            fb_model, _, actor, _, _ = nnx.merge(graph_def, state)

            key, key_eval = jax.random.split(key)
            # with jax.disable_jit():
            normalized_score, score = fb_awr.evaluate(
                actor, fb_model, dataset, task_id, num_inference_samples, render=False
            )
            iio.imwrite(f"{t + log_interval}.gif", frames)

            print(f"Return {normalized_score:.2f} ({score:.2f})")


if __name__ == "__main__":
    tyro.cli(main)
