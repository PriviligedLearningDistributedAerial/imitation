"""Custom policy classes and convenience methods."""

import abc
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch as th
from stable_baselines3.common import policies, torch_layers
from stable_baselines3.common.utils import obs_as_tensor, get_device
from stable_baselines3.sac import policies as sac_policies
from torch import nn

from imitation.data import types
from imitation.util import networks

SelfHomogeousActorCriticPolicy = TypeVar("SelfHomogeousActorCriticPolicy", bound="HomogenousActorCriticPolicy")

class HomogenousActorCriticPolicy(policies.ActorCriticPolicy):
    def __init__(
            self,
            observation_overide,
            action_overide,
            num_agents,
            **acp_kwargs
            ):
        super().__init__(**acp_kwargs)
        self.observation_overide = observation_overide
        self.action_overide = action_overide
        self.num_agents = num_agents

    def _predict(self, observation, **predict_kwargs):
        i_acts = []
        for i in range(self.num_agents):
            i_act = super()._predict(
                                    self.observation_overide(i, observation),
                                    predict_kwargs
                                    )
            i_acts.append(i_act)

        acts = th.hstack(i_acts)
        return acts
    
    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        # Check for common mistake that the user does not mix Gym/VecEnv API
        # Tuple obs are not supported by SB3, so we can safely do that check
        if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[1], dict):
            raise ValueError(
                "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
                "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
                "vs `obs = vec_env.reset()` (SB3 VecEnv). "
                "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
                "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
            )

        obs_tensor, vectorized_env = obs_as_tensor(observation, self.device), True

        with th.no_grad():
            actions = self._predict(obs_tensor, deterministic=deterministic)
        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy()

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)  # type: ignore[assignment, arg-type]
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(
                                    actions, 
                                    np.concatenate([self.action_space.low for i in range(self.num_agents)]), 
                                    np.concatenate([self.action_space.high for i in range(self.num_agents)])
                                    )  # type: ignore[assignment, arg-type]

        # Remove batch dimension if needed
        if not vectorized_env:
            assert isinstance(actions, np.ndarray)
            actions = actions.squeeze(axis=0)

        return actions, state  # type: ignore[return-value]
    
    @classmethod
    def load(cls: Type[SelfHomogeousActorCriticPolicy], 
            observation_overide,
            action_overide,
            num_agents,
            path: str, 
            device: Union[th.device, str] = "auto") -> SelfHomogeousActorCriticPolicy:
        """
        Load model from path.

        :param path:
        :param device: Device on which the policy should be loaded.
        :return:
        """
        device = get_device(device)
        # Note(antonin): we cannot use `weights_only=True` here because we need to allow
        # gymnasium imports for the policy to be loaded successfully
        saved_variables = th.load(path, map_location=device, weights_only=False)

        # Create policy object
        model = cls(observation_overide, action_overide, num_agents, **saved_variables["data"])
        # Load weights
        model.load_state_dict(saved_variables["state_dict"])
        model.to(device)
        return model


class NonTrainablePolicy(policies.BasePolicy, abc.ABC):
    """Abstract class for non-trainable (e.g. hard-coded or interactive) policies."""

    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        """Builds NonTrainablePolicy with specified observation and action space."""
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
        )

    def _predict(
        self,
        obs: Union[th.Tensor, Dict[str, th.Tensor]],
        deterministic: bool = False,
    ):
        np_actions = []
        if isinstance(obs, dict):
            np_obs = types.DictObs(
                {k: v.detach().cpu().numpy() for k, v in obs.items()},
            )
        else:
            np_obs = obs.detach().cpu().numpy()
        for np_ob in np_obs:
            np_ob_unwrapped = types.maybe_unwrap_dictobs(np_ob)
            assert self.observation_space.contains(np_ob_unwrapped)
            np_actions.append(self._choose_action(np_ob_unwrapped))
        np_actions = np.stack(np_actions, axis=0)
        th_actions = th.as_tensor(np_actions, device=self.device)
        return th_actions

    @abc.abstractmethod
    def _choose_action(
        self,
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
    ) -> np.ndarray:
        """Chooses an action, optionally based on observation obs."""

    def forward(self, *args):
        # technically BasePolicy is a Torch module, so this needs a forward()
        # method
        raise NotImplementedError  # pragma: no cover


class RandomPolicy(NonTrainablePolicy):
    """Returns random actions."""

    def _choose_action(
        self,
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
    ) -> np.ndarray:
        return self.action_space.sample()


class ZeroPolicy(NonTrainablePolicy):
    """Returns constant zero action."""

    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        """Builds ZeroPolicy with specified observation and action space."""
        super().__init__(observation_space, action_space)
        self._zero_action = np.zeros_like(
            action_space.sample(),
            dtype=action_space.dtype,
        )
        if self._zero_action not in action_space:
            raise ValueError(
                f"Zero action {self._zero_action} not in action space {action_space}",
            )

    def _choose_action(
        self,
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
    ) -> np.ndarray:
        return self._zero_action


class FeedForward32Policy(policies.ActorCriticPolicy):
    """A feed forward policy network with two hidden layers of 32 units.

    This matches the IRL policies in the original AIRL paper.

    Note: This differs from stable_baselines3 ActorCriticPolicy in two ways: by
    having 32 rather than 64 units, and by having policy and value networks
    share weights except at the final layer, where there are different linear heads.
    """

    def __init__(self, *args, **kwargs):
        """Builds FeedForward32Policy; arguments passed to `ActorCriticPolicy`."""
        super().__init__(*args, **kwargs, net_arch=[32, 32])

class HomogenousFeedForward32Policy(HomogenousActorCriticPolicy):
    """A feed forward policy network with two hidden layers of 32 units.

    This matches the IRL policies in the original AIRL paper.

    Note: This differs from stable_baselines3 ActorCriticPolicy in two ways: by
    having 32 rather than 64 units, and by having policy and value networks
    share weights except at the final layer, where there are different linear heads.
    """

    def __init__(self, *args, **kwargs):
        """Builds FeedForward32Policy; arguments passed to `ActorCriticPolicy`."""
        super().__init__(*args, **kwargs, net_arch=[256, 256, 128])


class SAC1024Policy(sac_policies.SACPolicy):
    """Actor and value networks with two hidden layers of 1024 units respectively.

    This matches the implementation of SAC policies in the PEBBLE paper. See:
    https://arxiv.org/pdf/2106.05091.pdf
    https://github.com/denisyarats/pytorch_sac/blob/master/config/agent/sac.yaml

    Note: This differs from stable_baselines3 SACPolicy by having 1024 hidden units
    in each layer instead of the default value of 256.
    """

    def __init__(self, *args, **kwargs):
        """Builds SAC1024Policy; arguments passed to `SACPolicy`."""
        super().__init__(*args, **kwargs, net_arch=[1024, 1024])


class NormalizeFeaturesExtractor(torch_layers.FlattenExtractor):
    """Feature extractor that flattens then normalizes input."""

    def __init__(
        self,
        observation_space: gym.Space,
        normalize_class: Type[nn.Module] = networks.RunningNorm,
    ):
        """Builds NormalizeFeaturesExtractor.

        Args:
            observation_space: The space observations lie in.
            normalize_class: The class to use to normalize observations (after being
                flattened). This can be any Module that preserves the shape;
                e.g. `nn.BatchNorm*` or `nn.LayerNorm`.
        """
        super().__init__(observation_space)
        # Below we have to ignore the type error when initializing the class because
        # there is no simple way of specifying a protocol that admits one positional
        # argument for the number of features while being compatible with nn.Module.
        # (it would require defining a base class and forcing all the subclasses
        # to inherit from it).
        self.normalize = normalize_class(self.features_dim)  # type: ignore[call-arg]

    def forward(self, observations: th.Tensor) -> th.Tensor:
        flattened = super().forward(observations)
        return self.normalize(flattened)
