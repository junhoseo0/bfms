import typing

import jax
import numpy as np
from dm_control.mujoco import Physics
from dm_control.rl import control
from jax import numpy as jnp


class DMCToGym:
    def __init__(self, dmc_env: control.Environment):
        self._dmc_env = dmc_env

    def reset(self) -> tuple[np.ndarray, dict]:
        timestep = self._dmc_env.reset()
        return self._concat_observation(timestep.observation), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        timestep = self._dmc_env.step(action)
        return (
            timestep.observation["observations"],
            timestep.reward,
            timestep.last(),
            False,
            {},
        )

    def render(self) -> np.ndarray:
        self._dmc_env._physics = typing.cast(Physics, self._dmc_env._physics)
        return self._dmc_env._physics.render(camera_id=0)

    def _concat_observation(self, observation: dict[str, np.ndarray]):
        obs_concated = jax.tree.map(lambda x: jnp.atleast_1d(x), observation)
        obs_concated = np.concatenate(jax.tree.leaves(obs_concated))
        return obs_concated
