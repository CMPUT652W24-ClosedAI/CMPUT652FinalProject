from dataclasses import dataclass
import math
import random
import shutil
import logging
import time

import numpy as np
import torch
import torch.nn as nn
import xml.etree.ElementTree as ET

from tqdm import tqdm
import tyro

from generator.enums import LayerName
from generator.map_utils import update_xml_map, convert_xml
from generator.maploader_controller import MapLoaderController
from generator.memory_buffer import MemoryBuffer, Transition
from generator.reward_scaling import ScaleReward
from generator.run_maploader import run_maploader
from generator.simple_value_function import baseline_fairness_score
from generator.unet_generator import Unet
from generator.value_function_extraction import squared_value_difference

# Configure the logger
logging.basicConfig(
    level=logging.DEBUG,  # Set the minimum level to capture
    format='%(asctime)s - %(levelname)s - %(message)s',
)
# Logging levels
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Set the desired logging level





@dataclass
class Args:
    num_episodes: int = 1000
    """Number of training episodes to run (times generator resets map to 0 and
    does episode_length steps)"""
    episode_length: int = 64
    """Number of changes to make to a map in a single episode (episode
    length)"""
    replay_buffer_size: int = 1000
    """Size of the replay buffer - number of steps that must be taken before
    changes are made"""
    replay_buffer_sample_size: int = 16
    """Count of elements out of the replay buffer to sample at one time"""
    step_jump: int = 4
    """Modulo value for steps to jump over when training - like in ppo atari
    (valid range: 1 to 1000)"""
    visualize_maps: bool = False
    """Run Java map visualizer"""
    epsilon: float = 1.0
    """Epsilon value used for epsilon-greedy decision-making"""
    tau: float = 0.005
    """Tau used for epsilon-greedy decision-making"""
    asym_to_fairness_ratio: float = 0.8
    """Higher ratio means asymmetrical reward is weighted more compared to
    fairness score - .8 is 80% asym score reward."""
    wall_reward: float = 0.1
    """If nonzero, adds artificial reward for placing walls"""
    use_baseline: bool = False
    """Whether to use the simple manhattan distance baseline fairness score, from simple_value_function.py"""




    # exp_name: str = os.path.basename(__file__)[: -len(".py")]
    # """the name of this experiment"""
    # torch_deterministic: bool = True
    # """if toggled, `torch.backends.cudnn.deterministic=False`"""
    # cuda: bool = True
    # """if toggled, cuda will be enabled by default"""
    # track: bool = False
    # """if toggled, this experiment will be tracked with Weights and Biases"""
    # wandb_project_name: str = "cleanRL"
    # """the wandb's project name"""
    # wandb_entity: str = None
    # """the entity (team) of wandb's project"""
    # capture_video: bool = False
    # """whether to capture videos of the agent performances (check out `videos` folder)"""
    # save_model: bool = False
    # """whether to save model into the `runs/{run_name}` folder"""
    # upload_model: bool = False
    # """whether to upload the saved model to huggingface"""
    # hf_entity: str = ""
    # """the user or org name of the model repository from the Hugging Face Hub"""

    def __post_init__(self):
        # Ensure that modulo_value is within the range [1, 1000]
        if not (1 <= self.step_jump <= 1000):
            raise ValueError("modulo_value must be between 1 and 1000, inclusive.")


def handle_error(message):
    raise RuntimeError(f"ALERT: An error occurred in MapLoader: {message}")

device = torch.device("cuda:0") if getattr(torch, "cuda", None) and torch.cuda.is_available() else torch.device("cpu")
controller = MapLoaderController()
controller.set_error_callback(handle_error)

def train(
    args: Args,
    map_paths=None,
    output_model_path: str = None,
):
    # Dynamically generate output_model_path with important argument values
    if output_model_path is None:
        timestamp = int(time.time())
        output_model_path = (
            f"models/output/training_net_"
            f"eps_{args.epsilon}_tau_{args.tau}_ratio_{args.asym_to_fairness_ratio}_"
            f"wr_{args.wall_reward}_ep_{args.num_episodes}_len_{args.episode_length}_baseline_{args.use_baseline}___{timestamp}.pt"
        )
    start_time = time.time()
    logger.info("Start of training")
    if map_paths is None:
        map_paths = ["input_maps/defaultMap.xml"]
    policy_net = Unet().to(device)
    target_net = Unet().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    replay_buffer = MemoryBuffer(args.replay_buffer_size)
    optimizer = torch.optim.Adam(policy_net.parameters())
    loss_function = nn.MSELoss()

    # Hyperparameters
    epsilon = args.epsilon
    tau = args.tau

    previous_sym_score = torch.tensor(0.0, device=device)
    previous_fairness_score = torch.tensor(0.0, device=device)

    fairness_scaling = ScaleReward(device=device)
    sym_scaling = ScaleReward(device=device)

    logger.info(f"Init complete after {time.time() - start_time} seconds")
    cumulative_rewards_list = torch.tensor([], device=device)

    for episode in tqdm(range(args.num_episodes), desc="Episodes"):
        cumulative_reward = torch.tensor(0.0, device=device)
        start_time = time.time()

        # Test Using convert xml
        map_path = random.choice(map_paths)
        shutil.copy(map_path, "tempMap.xml")
        file_path = "tempMap.xml"
        xml_map = ET.parse(file_path)
        tensor_map, invalid_actions_mask = convert_xml(xml_map)

        epsilon = max(epsilon - 1 / args.num_episodes, 0.05)
        state = tensor_map

        id = 100
        # logger.debug(f"Episode init finished after {time.time() - start_time} seconds")
        for step in tqdm(range(args.episode_length), desc=f"Steps in Episode {episode+1}", leave=False):
            if np.random.random() > epsilon:
                q_values = policy_net(state)
                # mask q_values
                q_values = (
                    q_values * (1 - state) * (1 - invalid_actions_mask)[None, :, :]
                )
                action = torch.tensor(
                    torch.unravel_index(torch.argmax(q_values), q_values.shape), device=device
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
                    [torch.tensor(plane, device=device), torch.tensor(x, device=device), torch.tensor(y, device=device)]
                , device=device)
            old_state = state.clone() # [ 5, 15,  8], [channel, x, y]
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
            # run_maploader("tempMap.xml")
            # Start MapLoader with the first map
            # if len(replay_buffer) > args.replay_buffer_size: # for debugging
            #     a = 1
            if args.visualize_maps:
                controller.start_maploader("tempMap.xml")

            id += 1



            sym_score_output = sym_score(state).float()

            shutil.copy(
                "tempMap.xml", "../gym_microrts/microrts/maps/16x16/tempMap.xml"
            )
            fairness_score_output = (
                squared_value_difference("maps/16x16/tempMap.xml")
                .reshape(-1)
                .squeeze(0)
            ) if not args.use_baseline else baseline_fairness_score(state)


            sym_score_difference = sym_score_output - previous_sym_score
            previous_sym_score = torch.clone(sym_score_output)

            fairness_score_difference = fairness_score_output - previous_fairness_score
            previous_fairness_score = torch.clone(fairness_score_output)

            if step == args.episode_length - 1:
                terminal = torch.tensor(1, device=device)
            else:
                terminal = torch.tensor(0, device=device)           

            scaled_sym_score = sym_scaling.scale_reward(sym_score_difference, terminal)
            scaled_fairness_score = fairness_scaling.scale_reward(fairness_score_difference, terminal)
            reward = args.asym_to_fairness_ratio * scaled_sym_score - (1 - args.asym_to_fairness_ratio) * scaled_fairness_score
            if action[0] == 1:
                reward += args.wall_reward

            cumulative_reward += reward


            replay_buffer.push(old_state, action, state, reward, terminal)

            if step % args.step_jump == 0 and len(replay_buffer) > args.replay_buffer_size:
                transitions = replay_buffer.sample(args.replay_buffer_sample_size)
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
        # logger.info(f"Episodic cumulative reward: {cumulative_reward}")
        cumulative_rewards_list = torch.cat(
            (cumulative_rewards_list, cumulative_reward.unsqueeze(0))
        )
    # Dedicated logger for cumulative rewards
    rewards_logger = logging.getLogger("rewards_logger")
    rewards_log_path = output_model_path.replace(".pt", "_rewards_log.txt")
    file_handler = logging.FileHandler(rewards_log_path, mode="w")
    file_handler.setFormatter(logging.Formatter('%(message)s'))  # Simple format, no timestamp
    rewards_logger.addHandler(file_handler)
    rewards_logger.setLevel(logging.INFO)
    for episode, reward in enumerate(cumulative_rewards_list, start=1):
        rewards_logger.info(f"Episode {episode}: Cumulative Reward = {reward.item():.4f}")
        # logger.info(f"Episode {episode}: Cumulative Reward = {reward.item():.4f}")
    torch.save(policy_net.state_dict(), output_model_path)
    logger.info(f"Saved network weights to {output_model_path}")


def sym_score(x):
    x = torch.argmax(x, dim=-3)
    reflected_x = torch.transpose(torch.flip(x, dims=[-1, -2]), -1, -2)
    return torch.sum(x != reflected_x)


if __name__ == "__main__":
    args = tyro.cli(Args)
    train(
        args,
        ["input_maps/defaultMap.xml", "input_maps/blank.xml", "input_maps/map-01.xml"],
    )
