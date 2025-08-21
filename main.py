import os

# Headless rendering with EXT_platform_device.
os.environ["MUJOCO_GL"] = "egl"

# Performance Flags.
# fmt: off
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_latency_hiding_scheduler=true "
    "--xla_gpu_triton_gemm_any=true"
)
# fmt: on

import datetime
import time

import jax
import optax
import tyro
from flax import nnx
from imageio import v3 as iio
from jax import numpy as jnp

from bfms.data import DatasetFactory
from bfms.fb import continuous as fb
from bfms.logging import RecordWriter


def log_result(result, step: int) -> None:
    for task in result["render"]:
        iio.imwrite(f"{step}_{task}.gif", result["render"][task])
    print(f"{step}: ", result["score"])


def main(
    dataset_id: str,
    dataset_dir: str | None = None,
    seed: int = 294,
    train_time_step: int = int(1e6),
    batch_size: int = 1024,
    num_reward_samples: int = 50_000,
    device_num: int = 0,
    log_interval: int = 10_000,
    eval_interval: int = 100_000,
):
    # Set GPU for JAX to use.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_num)
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(device_num)

    dataset = DatasetFactory.create(dataset_id, dataset_dir)
    print(f"Dataset {dataset_id} loaded with {len(dataset):,} transitions.")

    key = jax.random.key(seed)
    key, nnx_rngs = jax.random.split(key)
    nnx_rngs = nnx.Rngs(nnx_rngs)
    dataset.seed(seed)

    dim_state = dataset["observation"].shape[-1]
    dim_action = dataset["action"].shape[-1]
    # TODO: Move this to config.
    dim_latent = 100

    fb_model = fb.ForwardBackwardModel(dim_state, dim_action, dim_latent, rngs=nnx_rngs)
    target_fb_model = nnx.clone(fb_model)
    fb_optimizer = nnx.Optimizer(fb_model, optax.adam(1e-4), wrt=nnx.Param)

    actor = fb.GaussianActor(-1.0, 1.0, dim_state, dim_action, dim_latent, rngs=nnx_rngs)
    actor_optimizer = nnx.Optimizer(actor, optax.adam(1e-4), wrt=nnx.Param)

    train_state = fb.TrainState(
        fb_model, target_fb_model, actor, fb_optimizer, actor_optimizer, key
    )
    train_step = fb.make_train_step(
        fb.TrainConfig(actor_stddev=0.2, discount=0.98, dim_latent=dim_latent),
        batch_size,
        log_interval,
    )
    train_step = nnx.cached_partial(train_step, train_state)

    logger = RecordWriter()
    for t in range(0, train_time_step, log_interval):
        if t % eval_interval == 0:
            result = fb.evaluate(
                train_state.actor, train_state.fb_model, dataset, num_reward_samples, render=True
            )
            log_result(result, t)

        batches = dataset.sample(log_interval * batch_size)
        batches = jax.tree.map(lambda x: x.reshape(log_interval, batch_size, x.shape[-1]), batches)
        batches = jax.tree.map(jnp.asarray, batches)

        s_time = time.perf_counter()
        train_state, metrics = train_step(batches)
        elapsed_time = time.perf_counter() - s_time
        expected_time = (train_time_step // log_interval) * elapsed_time
        print(
            f"{t + log_interval}: {elapsed_time / log_interval:.3f}s/update",
            f"(interval: {elapsed_time:.2f}s / expected total: {datetime.timedelta(seconds=expected_time)})",
        )

        logger(metrics, (t + log_interval))

    result = fb.evaluate(
        train_state.actor, train_state.fb_model, dataset, num_reward_samples, render=True
    )
    log_result(result, t + log_interval)


if __name__ == "__main__":
    tyro.cli(main)
