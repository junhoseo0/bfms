from __future__ import annotations

import os
import re
import typing
from abc import abstractmethod
from pathlib import Path
from typing import Literal, NamedTuple, overload

import gymnasium as gym
import minari
import mujoco
import numpy as np
from dm_control import suite
from dm_control.mujoco import Physics
from dm_control.rl import control
from dm_control.suite import base
from jaxtyping import Float
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


class Batch(NamedTuple):
    observation: Float[np.ndarray, "batch ..."]
    action: Float[np.ndarray, "batch ..."]
    next_observation: Float[np.ndarray, "batch ..."]
    terminated: Float[np.ndarray, "batch ..."]


class BatchWithReward(NamedTuple):
    observation: Float[np.ndarray, "batch ..."]
    action: Float[np.ndarray, "batch ..."]
    next_observation: Float[np.ndarray, "batch ..."]
    terminated: Float[np.ndarray, "batch ..."]
    reward: Float[np.ndarray, "batch ..."] | dict[str, Float[np.ndarray, "batch ..."]]


class Dataset:
    tasks: list[str]
    _storage: dict[str, Float[np.ndarray, "dataset ..."]]
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
    def __getitem__(self, key: str) -> Float[np.ndarray, "dataset ..."]:
        raise NotImplementedError()

    @abstractmethod
    def seed(self, seed: int) -> None:
        raise NotImplementedError()

    @overload
    def sample(self, size: int, with_reward: Literal[False]) -> Batch: ...

    @overload
    def sample(self, size: int, with_reward: Literal[True]) -> BatchWithReward: ...

    @overload
    def sample(self, size: int) -> Batch: ...

    @abstractmethod
    def sample(self, size: int, with_reward: bool = False) -> Batch | BatchWithReward:
        raise NotImplementedError()

    @abstractmethod
    def get_rewards(
        self,
        task_id: str,
        actions: Float[np.ndarray, "batch action"],
        next_observations: Float[np.ndarray, "batch obs"],
    ) -> Float[np.ndarray, " batch"]:
        raise NotImplementedError()

    @overload
    def recover_environment(self, task: str, gym_compatible: Literal[True]) -> DMCToGym: ...

    @overload
    def recover_environment(
        self, task: str, gym_compatible: Literal[False]
    ) -> control.Environment: ...

    @overload
    def recover_environment(self, task: str) -> DMCToGym: ...

    @abstractmethod
    def recover_environment(self, task: str, gym_compatible: bool = True):
        raise NotImplementedError()

    @property
    @abstractmethod
    def unwrapped(self):
        raise NotImplementedError()


class MuJoCoDataset(Dataset):
    def __init__(self, dataset_dir: str | None, dataset_id: str):
        if dataset_dir is not None:
            os.environ["MINARI_DATASETS_PATH"] = dataset_dir
        self._dataset_id = dataset_id
        self._storage, self._env_id = self._load()
        self._np_rng = None

        self.tasks = [dataset_id]

    def __len__(self) -> int:
        return len(self._storage["observation"])

    def __getitem__(self, key: str) -> Float[np.ndarray, "{dataset} ..."]:
        return self._storage[key]

    def seed(self, seed: int) -> None:
        self._np_rng = np.random.default_rng(seed)

    def sample(self, size: int, with_reward: bool = False) -> Batch | BatchWithReward:
        if self._np_rng is None:
            # 294 is my favorite number.
            self.seed(seed=294)
        assert self._np_rng is not None

        rand_indexes = self._np_rng.integers(low=0, high=len(self), size=size)
        batch = Batch(
            observation=self._storage["observation"][rand_indexes],
            action=self._storage["action"][rand_indexes],
            next_observation=self._storage["next_observation"][rand_indexes],
            terminated=self._storage["terminated"][rand_indexes],
        )
        if with_reward:
            batch = BatchWithReward(*batch, reward=self._storage["reward"][rand_indexes])
        return batch

    def get_rewards(
        self,
        task: str,
        actions: Float[np.ndarray, "batch action"],
        next_observations: Float[np.ndarray, "batch ..."],
    ) -> Float[np.ndarray, " batch"]:
        raise NotImplementedError("MuJoCo dataset does not support reward inference.")

    def recover_environment(self, task: str | None = None, gym_compatible: bool = True) -> gym.Env:
        del task

        if not gym_compatible:
            raise ValueError("MuJoCo dataset only supports Gym environment.")

        if self._env_id is None:
            raise ValueError("env_id is not specified.")

        env = gym.make(self._env_id)
        return env

    def unwrapped(self) -> dict[str, Float[np.ndarray, "dataset ..."]]:
        return self._storage

    def _load(
        self,
    ) -> tuple[dict[str, Float[np.ndarray, "dataset ..."]], str]:
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
        self._np_rng = None

        self.tasks: tuple[str, ...] = suite.TASKS_BY_DOMAIN[domain_id]

    def __len__(self):
        return len(self._storage["observation"])

    def __getitem__(self, key: str) -> Float[np.ndarray, "dataset ..."]:
        return self._storage[key]

    def seed(self, seed: int) -> None:
        self._np_rng = np.random.default_rng(seed=seed)

    def sample(self, size: int, with_reward: bool = False) -> Batch | BatchWithReward:
        if self._np_rng is None:
            # 294 is my favorite number.
            self.seed(seed=294)
        assert self._np_rng is not None

        rand_indexes = self._np_rng.integers(low=0, high=len(self), size=size)
        batch = Batch(
            observation=self._storage["observation"][rand_indexes],
            action=self._storage["action"][rand_indexes],
            next_observation=self._storage["next_observation"][rand_indexes],
            terminated=self._storage["terminated"][rand_indexes],
        )
        if with_reward:
            reward = {
                task: self.get_rewards(task, batch.action, batch.next_observation)
                for task in self.tasks
            }
            batch = BatchWithReward(*batch, reward=reward)

        return batch

    def get_rewards(
        self,
        task: str,
        actions: Float[np.ndarray, "batch action"],
        next_observations: Float[np.ndarray, "batch obs"],
    ) -> Float[np.ndarray, " batch"]:
        env = self.recover_environment(task, gym_compatible=False)
        env._physics = typing.cast(Physics, env._physics)
        env._task = typing.cast(base.Task, env._task)

        rewards = []
        size = next_observations.shape[0]
        for i in range(size):
            with env._physics.reset_context():
                env._physics.set_state(next_observations[i])
                env._physics.set_control(actions[i])
            mujoco.mj_forward(env._physics.model.ptr, env._physics.data.ptr)  # pyright: ignore[reportAttributeAccessIssue]
            mujoco.mj_fwdPosition(env._physics.model.ptr, env._physics.data.ptr)  # pyright: ignore[reportAttributeAccessIssue]
            mujoco.mj_sensorVel(env._physics.model.ptr, env._physics.data.ptr)  # pyright: ignore[reportAttributeAccessIssue]
            mujoco.mj_subtreeVel(env._physics.model.ptr, env._physics.data.ptr)  # pyright: ignore[reportAttributeAccessIssue]
            reward = typing.cast(float, env._task.get_reward(env._physics))
            rewards.append(reward)

        rewards = np.array(rewards, dtype=np.float32)
        return rewards

    @overload
    def recover_environment(self, task: str, gym_compatible: Literal[True]) -> DMCToGym: ...

    @overload
    def recover_environment(
        self, task: str, gym_compatible: Literal[False]
    ) -> control.Environment: ...

    @overload
    def recover_environment(self, task: str) -> DMCToGym: ...

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

    def _load(self) -> dict[str, Float[np.ndarray, "dataset ..."]]:
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
