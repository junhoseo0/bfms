from __future__ import annotations

import os
import re
import typing
from abc import abstractmethod
from pathlib import Path
from typing import Literal, overload

import gymnasium as gym
import minari
import mujoco
import numpy as np
from dm_control import suite
from dm_control.mujoco import Physics
from dm_control.rl import control
from dm_control.suite import base
from numpy import typing as npt
from tqdm import tqdm

from bfms.interface import DMCToGym


class DatasetFactory:
    _REGISTRY = {
        "mujoco/halfcheetah/medium-v0": "MuJoCo",
        "mujoco/halfcheetah/expert-v0": "MuJoCo",
        "frozenlake/random-v0": "MuJoCo",
        "walker/rnd": "ExoRL",
        "cheetah/rnd": "ExoRL",
    }

    @staticmethod
    def create(dataset_id: str, dataset_dir: str | None = None) -> Dataset:
        dataset_group = DatasetFactory._REGISTRY.get(dataset_id)
        if dataset_group == "MuJoCo":
            dataset = MuJoCoDataset(dataset_dir, dataset_id)
        elif dataset_group == "ExoRL":
            if dataset_dir is None:
                raise ValueError("dataset_dir must be given for ExoRL datasets.")

            m = re.match(r"^([a-z]+)/([a-z]+)$", dataset_id)
            assert m is not None
            env_id, collection_method = m.group(1, 2)
            dataset = ExoRLDataset(dataset_dir, env_id, collection_method)
        else:
            raise ValueError(f"{dataset_id} is an invalid dataset.")

        return dataset


class Dataset:
    _storage: dict[str, np.ndarray]
    _dataset_id: str
    _min_return: float
    _max_return: float

    _TIMEOUT_LEN = {
        "mujoco/halfcheetah/medium-v0": 1000,
        "mujoco/halfcheetah/expert-v0": 1000,
        "frozenlake/random-v0": 100,
        "walker": 1000,
        "cheetah": 1000,
    }

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, key: str) -> npt.NDArray[np.float32]:
        raise NotImplementedError()

    @abstractmethod
    def seed(self, seed: int) -> None:
        raise NotImplementedError()

    @abstractmethod
    def sample(self, num_transitions: int) -> dict[str, npt.NDArray[np.float32]]:
        raise NotImplementedError()

    @abstractmethod
    def sample_with_rewards(
        self, task_id: str, num_transitions: int
    ) -> dict[str, npt.NDArray[np.float32]]:
        raise NotImplementedError()

    @abstractmethod
    def recover_environment(self, task: str, gym_compatible: bool):
        raise NotImplementedError()

    @abstractmethod
    def normalize_return(self, episodic_return: float) -> float:
        return (episodic_return - self._min_return) / (self._max_return - self._min_return)

    @property
    @abstractmethod
    def unwrapped(self):
        raise NotImplementedError()

    def _compute_min_max_return(self) -> tuple[float, float]:
        starting_index = 0
        episodic_returns = []
        for i in range(len(self)):
            episode_len = i - starting_index
            if self._storage["terminated"][i] or (
                episode_len + 1 == self._TIMEOUT_LEN[self._dataset_id]
            ):
                episodic_returns.append(
                    self._storage["reward"][starting_index : starting_index + episode_len].sum()
                )
                starting_index = i

        return np.min(episodic_returns), np.max(episodic_returns)


class MuJoCoDataset(Dataset):
    def __init__(self, dataset_dir: str, dataset_id: str):
        os.environ["MINARI_DATASETS_PATH"] = dataset_dir
        self._dataset_id = dataset_id
        self._storage, self._env_id = self._load()
        self._min_return, self._max_return = self._compute_min_max_return()
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

    def recover_environment(self, task: str | None = None):
        del task

        if self._env_id is None:
            raise ValueError("env_id is not specified.")

        env = gym.make(self._env_id)
        return env

    def sample_with_rewards(
        self, task_id: str, num_transitions: int
    ) -> dict[str, npt.NDArray[np.float32]]:
        del task_id

        return self.sample(num_transitions)

    def unwrapped(self):
        return self._storage

    def _load(self) -> tuple[dict[str, npt.NDArray[np.float32]], str]:
        dataset = minari.load_dataset(self._dataset_id, download=True)
        if dataset._eval_env_spec is not None:
            env_id = dataset._eval_env_spec.id
        elif dataset._env_spec is not None:
            env_id = dataset._env_spec.id
        else:
            # TODO: Replace with Logging
            print(f"WARN: {self._dataset_id} doesn't provide env_id.")

        episodes = {
            "observation": [],
            "action": [],
            "next_observation": [],
            "reward": [],
            "terminated": [],
        }
        for episode in tqdm(dataset, desc="Reading Data"):
            episodes["observation"].append(episode.observations[:-1])
            episodes["action"].append(episode.actions)
            episodes["next_observation"].append(episode.observations[1:])
            episodes["reward"].append(episode.rewards)
            episodes["terminated"].append(episode.terminations)

        storage = {}
        for key in episodes:
            storage[key] = np.concat(episodes[key], dtype=np.float32)

        if storage["observation"].ndim == 1:  # Assume discrete TODO
            storage["observation"] = np.astype(storage["observation"], np.int32)
            storage["next_observation"] = np.astype(storage["next_observation"], np.int32)
        if storage["action"].ndim == 1:
            storage["action"] = np.astype(storage["action"], np.int32)

            # storage["observation"] = np.expand_dims(storage["observation"], axis=1)
            # storage["next_observation"] = np.expand_dims(storage["next_observation"], axis=1)

        return storage, env_id


class ExoRLDataset(Dataset):
    """
    ExoRL dataset stores transition as S_t, A_{t+1}, S_{t+1}, R_{t+1}, meaning
    that the first action and reward are always zero (vector). Also, observation
    includes the terminal observation, and the true state is given as "physics".
    """

    def __init__(self, root_dir: str, domain_id: str, collection_method: str):
        self._dataset_id = domain_id
        self._collection_method = collection_method
        self._dataset_dir = Path(root_dir) / domain_id / collection_method / "buffer"
        self._storage = self._load()
        # self._min_return, self._max_return = self._compute_min_max_return()
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

    def sample_with_rewards(self, task: str, num_samples: int):
        batch = self.sample(num_samples)
        env = self.recover_environment(task, gym_compatible=False)
        env._physics = typing.cast(Physics, env._physics)
        env._task = typing.cast(base.Task, env._task)

        rewards = []
        for i in range(num_samples):
            with env._physics.reset_context():
                env._physics.set_state(batch["next_state"][i])
                # Why action, not next_action?
                env._physics.set_control(batch["action"][i])
            mujoco.mj_forward(env._physics.model.ptr, env._physics.data.ptr)  # pyright: ignore[reportAttributeAccessIssue]
            mujoco.mj_fwdPosition(env._physics.model.ptr, env._physics.data.ptr)  # pyright: ignore[reportAttributeAccessIssue]
            mujoco.mj_sensorVel(env._physics.model.ptr, env._physics.data.ptr)  # pyright: ignore[reportAttributeAccessIssue]
            mujoco.mj_subtreeVel(env._physics.model.ptr, env._physics.data.ptr)  # pyright: ignore[reportAttributeAccessIssue]
            rewards.append(env._task.get_reward(env._physics))

        batch["reward"] = np.array(rewards)
        return batch

    @overload
    def recover_environment(self, task: str, gym_compatible: Literal[True]) -> DMCToGym: ...

    @overload
    def recover_environment(
        self, task: str, gym_compatible: Literal[False]
    ) -> control.Environment: ...

    def recover_environment(
        self, task: str, gym_compatible: bool = True
    ) -> DMCToGym | control.Environment:
        if self._dataset_id is None:
            raise ValueError("env_id is not specified.")

        env = suite.load(self._dataset_id, task, environment_kwargs={"flat_observation": True})
        if gym_compatible:
            env = DMCToGym(env)
        return env

    def unwrapped(self):
        return self._storage

    def _load(self) -> dict[str, npt.NDArray[np.float32]]:
        episodes = {
            "state": [],
            "observation": [],
            "action": [],
            "next_state": [],
            "next_observation": [],
            "terminated": [],
        }

        episode_files = list(self._dataset_dir.glob("episode_*_*.npz"))
        for i in tqdm(range(5000), desc="Reading Data"):
            episode_npz = episode_files[i]
            episode = np.load(episode_npz)
            episodes["state"].append(episode["physics"][:-1])
            episodes["observation"].append(episode["observation"][:-1])
            episodes["action"].append(episode["action"][1:])
            episodes["next_state"].append(episode["physics"][1:])
            episodes["next_observation"].append(episode["observation"][1:])
            episodes["terminated"].append(1.0 - episode["discount"][1:])

        storage = {}
        for key in episodes:
            storage[key] = np.concat(episodes[key], dtype=np.float32)

        return storage
