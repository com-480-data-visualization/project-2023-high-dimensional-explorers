import random
import os
import logging
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

import hydra
from omegaconf import DictConfig, OmegaConf

import gym
import gym.utils

import caching


# logging_format = "%(lineno)4s: %(asctime)s: %(message)s"
logging_format = "%(asctime)s: %(message)s"
logging_level = 15
logging.basicConfig(level=logging_level,
                    format=logging_format)

logger = logging.getLogger(__name__)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def build_relu_layers(input_dim: int,
                      hidden_layer_widths: List[int],
                      output_dim: int,
                      include_bias: bool) -> List[torch.nn.Module]:
    num_layers = len(hidden_layer_widths)
    all_layer_widths = [input_dim] + hidden_layer_widths + [output_dim]
    final_linear_layer = torch.nn.Linear(hidden_layer_widths[-1],
                                         output_dim,
                                         bias=include_bias)
    layer_list = []
    for i in range(num_layers):
        w0 = all_layer_widths[i]
        w1 = all_layer_widths[i + 1]
        layer_list += [torch.nn.Linear(w0, w1, bias=include_bias), torch.nn.ReLU()]
    layer_list += [final_linear_layer]
    return layer_list


def predict(
    state: np.ndarray,
    policy: torch.nn.Module,
    is_deterministic_action: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    state_torch = torch.from_numpy(state).type(torch.FloatTensor)
    action_logits = policy(state_torch)

    action_probs = torch.nn.functional.softmax(action_logits, dim=-1)
    distribution = torch.distributions.Categorical(action_probs)

    if is_deterministic_action:
        _, action = action_logits.max(0)
    else:
        action = distribution.sample()
    action_logprob = distribution.log_prob(action).reshape(1)
    return action, action_logprob


def flip_cumsum_flip(x: np.ndarray) -> np.ndarray:
    return np.flip(np.cumsum(np.flip(x)))


def update_policy(optimizer: torch.optim.Optimizer,
                  episode_rewards: List[float],
                  episode_action_logprobs: List[torch.Tensor],
                  gamma: float) -> Tuple[float, float]:

    episode_reward_len = len(episode_rewards)
    discount_factor = gamma ** (np.arange(episode_reward_len, 0, -1) - 1)
    discounted_rewards = episode_rewards * discount_factor
    rewards_array = flip_cumsum_flip(discounted_rewards).tolist()

    rewards = torch.FloatTensor(rewards_array)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

    catted_actions = torch.cat(episode_action_logprobs)
    episode_losses = torch.mul(catted_actions, rewards).mul(-1)
    loss = torch.sum(episode_losses, -1)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    policy_loss = loss.item()
    policy_reward = np.sum(episode_rewards)
    return policy_loss, policy_reward


def _build_layers(num_hidden: int,
                  num_layers: int) -> List[torch.nn.Module]:
    state_space_dim = 4
    action_space_dim = 2

    input_dim = state_space_dim
    output_dim = action_space_dim
    hidden_layer_widths = [num_hidden] * num_layers

    include_bias = True
    layers = build_relu_layers(
        input_dim, hidden_layer_widths, output_dim, include_bias
    )
    return layers


def train_policy(env_name: str,
                 num_hidden: int,
                 num_layers: int,
                 episodes: int,
                 learning_rate: float) -> Dict[str, Any]:
    layer_list = _build_layers(num_hidden, num_layers)

    env = gym.make(env_name)
    # env.seed(0)
    env.reset(seed=42)

    assert env.observation_space.shape[0] == layer_list[0].in_features
    assert env.action_space.n == layer_list[-1].out_features

    policy = torch.nn.Sequential(*layer_list)

    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    episode_length = 1000
    gamma = 0.99

    log_every = 100
    reward_threshold = env.spec.reward_threshold

    frac_wrong_action_history = []
    loss_history = []
    reward_history = []
    scores = []
    octant_inclusion_history = []

    for episode in range(episodes):
        episode_states = []
        episode_rewards = []
        episode_action_logprobs = []
        episode_actions = []

        state, _ = env.reset()
        for idx in range(episode_length):
            action, action_logprob = predict(state, policy, False)
            state, reward, done, info, _ = env.step(action.item())
            episode_states.append(state)

            episode_rewards.append(reward)
            episode_action_logprobs.append(action_logprob)
            episode_actions.append(action)

            if done:
                break

        policy_loss, policy_reward = update_policy(optimizer,
                                                   episode_rewards,
                                                   episode_action_logprobs,
                                                   gamma)
        loss_history.append(policy_loss)
        reward_history.append(policy_reward)
        scores.append(idx)

        mean_score = np.mean(scores[-100:])

        if episode % log_every == 0:
            msg = "Episode {:>4}\tAverage length: {:>.2f}".format(episode, mean_score)
            logger.info(msg)

        if mean_score > reward_threshold:
            msg = "Solved after {} episodes! Running average is now {}. Last episode ran to {} time steps.".format(
                episode, mean_score, idx
            )
            logger.info(msg)
            break

    training_results = {
        "policy": policy,
        "octant_inclusion_history": octant_inclusion_history,
        "frac_wrong_action_history": frac_wrong_action_history,
        "loss_history": loss_history,
        "reward_history": reward_history,
    }
    return training_results


def _replace_numerical_inf_with_actual_inf(x: np.ndarray) -> np.ndarray:
    x_sign = np.sign(x)
    is_numerical_inf = np.abs(x) > 1e35
    x_replaced = np.where(is_numerical_inf, x_sign * np.inf, x)
    return x_replaced


@hydra.main(version_base=None,
            config_path="../config",
            config_name="config.yaml")
def main(cfg: DictConfig):
    seed = cfg.prng.seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if cfg.paths.tilde:
        tilde = cfg.paths.tilde
    else:
        tilde = os.path.expanduser("~")

    cfg_model = cfg.cartpole.model
    cfg_data = cfg.cartpole.data
    cfg_training = cfg.cartpole.training

    cache_dir = os.path.join(tilde, cfg.paths.cache)
    num_hidden = cfg_model.neurons_per_layer
    num_layers = cfg_model.num_relu_layers

    env_name = cfg_data.env_name
    episodes = cfg_data.episodes

    learning_rate = cfg_training.learning_rate

    calc_fun = train_policy
    calc_args = (env_name, num_hidden, num_layers, episodes, learning_rate)

    # force_regeneraton = False
    force_regeneraton = True

    calc_kwargs = {}
    training_results = caching.cached_calc(
        cache_dir, calc_fun, calc_args, calc_kwargs, force_regeneraton
    )
    model = training_results["policy"]

    if cfg.paths.tilde:
        tilde = cfg.paths.tilde
    else:
        tilde = os.path.expanduser("~")
    out_filedir = os.path.join(tilde, cfg.paths.nnet)
    os.makedirs(out_filedir, exist_ok=True)

    model_ident_pattern = cfg_model.ident_pattern

    filename = model_ident_pattern.format(cfg_model.num_relu_layers,
                                          cfg_model.neurons_per_layer)

    to_save = {
        "model": model,
    }
    state_dict_fullfilename = os.path.join(out_filedir, filename + ".pt")
    logger.info("Writing {}".format(state_dict_fullfilename))
    torch.save(to_save, state_dict_fullfilename)


if __name__ == "__main__":
    main()
