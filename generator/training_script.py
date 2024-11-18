import shutil

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from generator.enums import LayerName
from generator.map_utils import update_xml_map
from generator.memory_buffer import MemoryBuffer, Transition
from generator.unet_generator import Unet
from generator.value_function_extraction import squared_value_difference


def train(map_path: str):
    policy_net = Unet()
    target_net = Unet()
    target_net.load_state_dict(policy_net.state_dict())
    replay_buffer = MemoryBuffer(10_000)
    optimizer = torch.optim.Adam(policy_net.parameters())
    loss_function = nn.MSELoss()

    # Hyperparameters
    epsilon = 1.0
    tau = 0.005

    for episode in tqdm(range(100_000)):
        epsilon = max(epsilon - 1 / 100_000, 0.05)
        blank_planes = torch.zeros(5, 16, 16)
        empty_plane = torch.ones(1, 16, 16)
        state = torch.cat((empty_plane, blank_planes), dim=0)
        shutil.copy("defaultMap.xml", "tempMap.xml")
        id = 100
        for step in range(64):
            if np.random.random() > epsilon:
                q_values = policy_net(state)
                # TODO: account for other invalid actions when using starting map
                # mask q_values
                q_values = q_values * (1 - state)
                action = torch.tensor(torch.unravel_index(torch.argmax(q_values), q_values.shape))
            else:
                x = np.random.randint(16)
                y = np.random.randint(16)
                plane = np.random.randint(5)
                if state[plane, x, y] == 1:
                    plane = 5
                action = torch.tensor([torch.tensor(plane), torch.tensor(x), torch.tensor(y)])
            old_state = state.clone()
            state[:, action[1], action[2]] = 0
            state[action[0], action[1], action[2]] = 1

            old_index = (old_state[:, action[1].item(), action[2].item()] == 1).nonzero(as_tuple=True)[0].item()
            update_xml_map("tempMap.xml", LayerName(action[0].item()), LayerName(old_index), action[1].item(), action[2].item(), id)
            id += 1
            reward = torch.tensor(sym_score(state) - sym_score(old_state)) / 2

            if step == 63:
                terminal = torch.tensor(1)
                shutil.copy("tempMap.xml", "../gym_microrts/microrts/maps/16x16/tempMap.xml")
                reward -= squared_value_difference("maps/16x16/tempMap.xml")
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
                    torch.arange(16), action_batch[:, 0], action_batch[:, 1], action_batch[:, 2]]
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
                    target_net_state_dict[key] = policy_net_state_dict[key] * tau + target_net_state_dict[key] * (
                                1 - tau)
                target_net.load_state_dict(target_net_state_dict)

def sym_score(x):
    x = torch.argmax(x, dim=-3)
    reflected_x = torch.transpose(torch.flip(x, dims=[-1, -2]), -1, -2)
    return torch.sum(x != reflected_x)

if __name__ == '__main__':
    # ../maps/16x16/defaultMap.xml
    train("defaultMap.xml")
