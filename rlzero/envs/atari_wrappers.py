from collections import deque
from typing import Any, Dict, Tuple

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

cv2.ocl.setUseOpenCL(False)


class NoopResetEnv(gym.Wrapper):
    """Sample initial states by taking random number of no-ops on reset.

    This wrapper takes a random number of no-op actions (action 0) at the
    beginning of each episode to ensure different starting points.
    """

    def __init__(self, env: gym.Env, noop_max: int = 30) -> None:
        super().__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert (env.unwrapped.get_action_meanings()[0] == 'NOOP'
                ), 'The first action must be NOOP.'

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform no-op action for a random number of steps in [1,
        noop_max]."""
        self.env.reset(**kwargs)
        noops = self.override_num_noops or self.unwrapped.np_random.randint(
            1, self.noop_max + 1)
        assert noops > 0, 'Number of no-ops must be greater than 0.'

        obs = None
        for _ in range(noops):
            obs, rewards, terminal, truncated, info = self.env.step(
                self.noop_action)
            done = terminal or truncated
            if done:
                obs, info = self.env.reset(**kwargs)

        return obs, info

    def step(self,
             ac: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        return self.env.step(ac)


class FireResetEnv(gym.Wrapper):
    """Take a FIRE action after reset for environments that require it.

    This wrapper is used for environments like Space Invaders that require the
    player to fire after resetting to start the game.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.env.reset(**kwargs)
        obs, rewards, terminal, truncated, info = self.env.step(1)
        done = terminal or truncated
        if done:
            self.env.reset(**kwargs)
        obs, rewards, terminal, truncated, info = self.env.step(2)
        done = terminal or truncated
        if done:
            self.env.reset(**kwargs)
        return obs, info

    def step(self,
             ac: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        return self.env.step(ac)


class EpisodicLifeEnv(gym.Wrapper):
    """Make end-of-life == end-of-episode, but only reset on true game over.

    This helps with value estimation for games where lives are involved.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(
            self, action: Any
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, rewards, terminal, truncated, info = self.env.step(action)
        done = terminal or truncated
        self.was_real_done = done
        lives = self.env.unwrapped.ale.lives()

        # Make loss of life terminal
        if lives < self.lives and lives > 0:
            done = True
        self.lives = lives
        return obs, rewards, terminal, truncated, info

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Only reset when lives are exhausted."""
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, rewards, terminal, truncated, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    """Return every `skip`-th frame, skipping `skip-1` frames.

    This wrapper applies frame skipping and takes the maximum of the last two
    frames.
    """

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        self._obs_buffer = np.zeros((2, ) + env.observation_space.shape,
                                    dtype=np.uint8)
        self._skip = skip

    def step(
            self, action: Any
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, rewards, terminal, truncated, info = self.env.step(action)
            done = terminal or truncated
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += rewards
            if done:
                break

        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminal, truncated, info

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        return self.env.reset(**kwargs)


class ClipRewardEnv(gym.RewardWrapper):
    """Clip the rewards to {-1, 0, 1} based on their sign.

    This helps stabilize training by preventing large reward values from
    dominating learning.
    """

    def reward(self, reward: float) -> float:
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    """Warp frames to 84x84 as done in the DQN Nature paper.

    Optionally, convert frames to grayscale.
    """

    def __init__(
        self,
        env: gym.Env,
        width: int = 84,
        height: int = 84,
        grayscale: bool = True,
        dict_space_key: str = None,
    ) -> None:
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key

        num_colors = 1 if self._grayscale else 3
        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )

        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space

        assert original_space.dtype == np.uint8 and len(
            original_space.shape) == 3

    def observation(self, obs: np.ndarray) -> np.ndarray:
        frame = obs if self._key is None else obs[self._key]
        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self._width, self._height),
                           interpolation=cv2.INTER_AREA)
        if self._grayscale:
            frame = np.expand_dims(frame, -1)
        if self._key is None:
            return frame
        obs[self._key] = frame
        return obs


class FrameStack(gym.Wrapper):
    """Stack `k` last frames to give the model temporal context.

    This wrapper stacks the last `k` observations into a single observation.
    """

    def __init__(self, env: gym.Env, k: int) -> None:
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(shp[:-1] + (shp[-1] * k, )),
            dtype=env.observation_space.dtype,
        )

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset()
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_ob(), info

    def step(
            self, action: Any
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, rewards, terminal, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_ob(), rewards, terminal, truncated, info

    def _get_ob(self) -> np.ndarray:
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class ScaledFloatFrame(gym.ObservationWrapper):
    """Scale frames to the range [0, 1] to improve model convergence.

    This wrapper converts pixel values from 0-255 to 0-1.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return np.array(observation).astype(np.float32) / 255.0


class LazyFrames:
    """Efficiently stack frames to save memory by storing shared frames only
    once.

    This class is optimized for memory usage in the replay buffer.
    """

    def __init__(self, frames: list) -> None:
        self._frames = frames
        self._out = None

    def _force(self) -> np.ndarray:
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None) -> np.ndarray:
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self) -> int:
        return len(self._force())

    def __getitem__(self, i: int) -> np.ndarray:
        return self._force()[i]

    def count(self) -> int:
        return self._force().shape[self._force().ndim - 1]

    def frame(self, i: int) -> np.ndarray:
        return self._force()[..., i]


def make_atari(env_id: str, max_episode_steps=None) -> gym.Env:
    """Create an Atari environment with NoopReset and MaxAndSkip wrappers.

    Args:
        env_id: The Atari environment ID.
        max_episode_steps: Maximum number of steps per episode (not used here).

    Returns:
        Wrapped Atari environment.
    """
    env = gym.make(env_id)
    assert ('NoFrameskip' in env.spec.id
            ), "The environment must be configured with 'NoFrameskip'."
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)

    return env


def wrap_deepmind(
    env: gym.Env,
    episode_life: bool = True,
    clip_rewards: bool = True,
    frame_stack: bool = False,
    scale: bool = False,
) -> gym.Env:
    """Wrap the Atari environment with DeepMind-style preprocessing.

    Args:
        env: The Atari environment.
        episode_life: Whether to make episode life episodic.
        clip_rewards: Whether to clip rewards.
        frame_stack: Whether to stack frames.
        scale: Whether to scale frames to [0, 1].

    Returns:
        Wrapped environment.
    """
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env


class ImageToPyTorch(gym.ObservationWrapper):
    """Convert image observations to PyTorch format (channels x height x
    width)."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(old_shape[-1], old_shape[0], old_shape[1]),
            dtype=np.uint8,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return np.transpose(observation, axes=(2, 0, 1))


def wrap_pytorch(env: gym.Env) -> gym.Env:
    """Convert environment observations to PyTorch format.

    Args:
        env: The environment to wrap.

    Returns:
        PyTorch-compatible wrapped environment.
    """
    return ImageToPyTorch(env)
