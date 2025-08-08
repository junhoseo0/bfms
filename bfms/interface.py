import jax
import numpy as np
from jax import numpy as jnp


class DMCToGym:
    def __init__(self, dmc_env):
        self._dmc_env = dmc_env

    def reset(self) -> tuple[np.ndarray, dict]:
        timestep = self._dmc_env.reset()
        return self._concat_observation(timestep.observation), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        timestep = self._dmc_env.step(action)
        return (
            self._concat_observation(timestep.observation),
            timestep.reward,
            timestep.last(),  # Termination
            False,  # Truncation
            {},
        )

    def _concat_observation(self, observation: dict[str, np.ndarray]):
        obs_concated = jax.tree.map(lambda x: jnp.atleast_1d(x), observation)
        obs_concated = np.concatenate(jax.tree.leaves(obs_concated))
        return obs_concated
