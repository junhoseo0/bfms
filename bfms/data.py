from pathlib import Path

import minari
import numpy as np
from numpy import typing as npt
from tqdm import tqdm

DATASET_REGISTRY = {
    "mujoco/halfcheetah/medium-v0": "D4RL",
    "walker/rnd": "ExoRL",
}


class D4RLDataset:
    def __init__(self, group: str, env: str, data_quality: str):
        self._dataset_id = f"{group}/{env}/{data_quality}"
        self._storage = self._load()
        self._np_rng = None

    def __len__(self):
        return len(self._storage["observation"])

    def __getitem__(self, key: str) -> npt.NDArray[np.float32]:
        return self._storage[key]

    def seed(self, seed: int) -> None:
        self._np_rng = np.random.default_rng(seed)

    def sample(self, num_transitions: int) -> dict[str, npt.NDArray[np.float32]]:
        if self._np_rng is None:
            # 294 is my favorite number.
            self.seed(seed=294)

        assert self._np_rng is not None

        rand_indexes = self._np_rng.integers(low=0, high=len(self), size=num_transitions)
        batch = {key: self._storage[key][rand_indexes] for key in self._storage}
        return batch

    def _load(self) -> dict[str, npt.NDArray[np.float32]]:
        self._raw_dataset = minari.load_dataset(self._dataset_id)
        episodes = {
            "observation": [],
            "action": [],
            "next_observation": [],
            "reward": [],
            "terminated": [],
        }
        for episode in tqdm(self._raw_dataset, desc="Reading Data"):
            episodes["observation"].append(episode.observations[:-1])
            episodes["action"].append(episode.actions)
            episodes["next_observation"].append(episode.observations[1:])
            episodes["reward"].append(episode.rewards)
            episodes["terminated"].append(episode.terminations)

        storage = {}
        for key in episodes:
            storage[key] = np.concat(episodes[key], dtype=np.float32)
        return storage


class ExoRLDataset:
    """
    ExoRL dataset stores transition as S_t, A_{t+1}, S_{t+1}, R_{t+1}, meaning
    that the first action and reward are always zero (vector). Also, observation
    includes the terminal observation, and the true state is given as "physics".
    """

    def __init__(self, root_dir: str, env: str, collection_method: str):
        self._env = env
        self._collection_method = collection_method
        self._dataset_dir = Path(root_dir) / env / collection_method / "buffer"
        self._storage = self._load()
        self._np_rng = None

    def __len__(self):
        return len(self._storage["observation"])

    def __getitem__(self, key: str) -> npt.NDArray[np.float32]:
        return self._storage[key]

    def seed(self, seed: int) -> None:
        self._np_rng = np.random.default_rng(seed=seed)

    def sample(self, num_transitions: int) -> dict[str, npt.NDArray[np.float32]]:
        if self._np_rng is None:
            # 294 is my favorite number.
            self.seed(seed=294)

        assert self._np_rng is not None

        rand_indexes = self._np_rng.integers(low=0, high=len(self), size=num_transitions)
        batch = {key: self._storage[key][rand_indexes] for key in self._storage}
        return batch

    def _load(self) -> dict[str, npt.NDArray[np.float32]]:
        episodes = {
            "state": [],
            "observation": [],
            "action": [],
            "next_state": [],
            "next_observation": [],
            "terminated": [],
        }
        for episode_npz in tqdm(self._dataset_dir.glob("episode_*_*.npz"), desc="Reading Data"):
            episode = np.load(episode_npz)
            episodes["state"].append(episode["physics"][:-1])
            episodes["observation"].append(episode["observation"][:-1])
            episodes["action"].append(episode["action"][1:])
            episodes["next_state"].append(episode["physics"][1:])
            episodes["next_observation"].append(episode["observation"][1:])
            episodes["terminated"].append(1.0 - episode["discount"][:-1])

        storage = {}
        for key in episodes:
            storage[key] = np.concat(episodes[key], dtype=np.float32)

        return storage
