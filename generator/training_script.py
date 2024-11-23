import math
import shutil

import numpy as np
import torch
import torch.nn as nn
import xml.etree.ElementTree as ET

from tqdm import tqdm

from generator.enums import LayerName
from generator.map_utils import update_xml_map, convert_xml
from generator.memory_buffer import MemoryBuffer, Transition
from generator.unet_generator import Unet
from generator.value_function_extraction import squared_value_difference


def train(
    map_path: str,
    num_episodes: int,
    output_model_path: str,
    alpha: float = 0.8,
):
    policy_net = Unet()
    target_net = Unet()
    target_net.load_state_dict(policy_net.state_dict())
    replay_buffer = MemoryBuffer(10_000)
    optimizer = torch.optim.Adam(policy_net.parameters())
    loss_function = nn.MSELoss()

    # Hyperparameters
    epsilon = 1.0
    tau = 0.005

    fairness_reward_trace, asym_reward_trace = 0.0, 0.0
    estimated_fairness_average, estimated_asym_average = 0.0, 0.0
    fairness_counter, asym_counter = 1, 1

    for episode in tqdm(range(num_episodes)):
        # Test Using convert xml
        shutil.copy("input_maps/defaultMap.xml", "tempMap.xml")
        file_path = "tempMap.xml"
        xml_map = ET.parse(file_path)
        tensor_map, invalid_actions_mask = convert_xml(xml_map)

        epsilon = max(epsilon - 1 / num_episodes, 0.05)
        state = tensor_map

        id = 100
        for step in range(64):
            if np.random.random() > epsilon:
                q_values = policy_net(state)
                # mask q_values
                q_values = (
                    q_values * (1 - state) * (1 - invalid_actions_mask)[None, :, :]
                )
                action = torch.tensor(
                    torch.unravel_index(torch.argmax(q_values), q_values.shape)
                )
            else:
                while True:
                    x = np.random.randint(16)
                    y = np.random.randint(16)
                    # Mask Invalid Actions
                    if invalid_actions_mask[x, y] != 1.0:
                        break
                plane = np.random.randint(5)
                if state[plane, x, y] == 1:
                    plane = 5
                action = torch.tensor(
                    [torch.tensor(plane), torch.tensor(x), torch.tensor(y)]
                )
            old_state = state.clone()
            state[:, action[1], action[2]] = 0
            state[action[0], action[1], action[2]] = 1

            test_index = (
                old_state[:, action[1].item(), action[2].item()] == 1
            ).nonzero(as_tuple=True)[0]

            # Sometimes happens when a square is marked as empty and has a resource
            # TODO: This is likely because of a bug elsewhere. This should be fixed.
            if test_index.shape[0] == 2:
                old_index = test_index[1].item()
            else:
                old_index = test_index.item()

            update_xml_map(
                "tempMap.xml",
                LayerName(action[0].item()),
                LayerName(old_index),
                action[1].item(),
                action[2].item(),
                id,
            )
            id += 1
            reward = torch.tensor(0.0)

            if step == 63:
                terminal = torch.tensor(1)
                sym_score_output = sym_score(state).float()
                scaled_sym_score, asym_reward_trace, estimated_asym_average, asym_counter = scale_reward(sym_score_output.item(), asym_reward_trace, estimated_asym_average, asym_counter)
                reward +=  alpha * scaled_sym_score
                shutil.copy(
                    "tempMap.xml", "../gym_microrts/microrts/maps/16x16/tempMap.xml"
                )
                difference = squared_value_difference("maps/16x16/tempMap.xml")[0, 0]
                scaled_fairness_score, fairness_reward_trace, estimated_fairness_average, fairness_counter = scale_reward(difference, fairness_reward_trace, estimated_fairness_average, fairness_counter)
                reward -=  (1 - alpha) * scaled_fairness_score
            else:
                terminal = torch.tensor(0)
            replay_buffer.push(old_state, action, state, reward, terminal)

            if step % 4 == 0 and len(replay_buffer) > 1_000:
                transitions = replay_buffer.sample(16)
                batch = Transition(*zip(*transitions))

                state_batch = torch.stack(batch.state)
                action_batch = torch.stack(batch.action)
                reward_batch = torch.stack(batch.reward)
                next_state_batch = torch.stack(batch.next_state)
                terminal_batch = torch.stack(batch.terminal)

                # compute targets
                with torch.no_grad():
                    next_q = target_net(next_state_batch) * (1 - next_state_batch)
                    next_q_maxes = torch.max(next_q.view(16, -1), dim=1)[0]
                    target = reward_batch + (next_q_maxes * (1 - terminal_batch))

                # compute TD error
                predicted_q_values = policy_net(state_batch)[
                    torch.arange(16),
                    action_batch[:, 0],
                    action_batch[:, 1],
                    action_batch[:, 2],
                ]
                loss = loss_function(predicted_q_values, target)

                # update policy network
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
                optimizer.step()

                # update target network
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[
                        key
                    ] * tau + target_net_state_dict[key] * (1 - tau)
                target_net.load_state_dict(target_net_state_dict)
    torch.save(policy_net.state_dict(), output_model_path)


def sym_score(x):
    x = torch.argmax(x, dim=-3)
    reflected_x = torch.transpose(torch.flip(x, dims=[-1, -2]), -1, -2)
    return torch.sum(x != reflected_x)


def scale_reward(reward, reward_trace, estimated_mean, counter, gamma = 1):
    reward_trace = gamma * reward_trace + reward
    estimated_mean, sigma, counter = sample_mean_var(
        reward_trace, 0, estimated_mean, counter
    )
    return reward / math.sqrt(sigma + 1e-8), reward_trace, estimated_mean, counter


def sample_mean_var(reward_trace, mean, estimated_mean, counter):
    counter += 1
    update_mean = mean + (1 / counter) * (reward_trace - mean)
    estimated_mean += (reward_trace - mean) * (reward_trace - update_mean)
    sigma = estimated_mean / (counter - 1) if counter > 2 else 1
    return update_mean, sigma, counter


if __name__ == "__main__":
    train(
        "input_maps/defaultMap.xml",
        1_000,
        "models/policy_net_with_layer_norm_and_scaled_rewards_with_alpha_0.8.pt",
    )
