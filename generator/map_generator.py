from dataclasses import dataclass
import math
import os
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@dataclass
class Args:

    input_map_path: str = "input_maps/defaultMap.xml"
    """Number of training episodes to run (times generator resets map to 0 and
    does episode_length steps)"""
    output_map_dir: str = f"./outputMaps/__{int(time.time())}"
    """output folder to place maps into"""
    model_path: str = f"models/policy_net_non_episodic_rewards.pt"
    """input path of the model used to generate these maps"""


    num_maps: int = 100
    """Number of maps to generate"""

    episode_length: int = 64
    """Number of changes to make to a map in a single episode (episode length)"""

    # replay_buffer_size: int = 1000
    # """Size of the replay buffer - number of steps that must be taken before
    # changes are made"""
    # replay_buffer_sample_size: int = 16
    # """Count of elements out of the replay buffer to sample at one time"""
    # step_jump: int = 4
    # """Modulo value for steps to jump over when training - like in ppo atari
    # (valid range: 1 to 1000)"""
    # visualize_maps: bool = False
    # """Run Java map visualizer"""
    # epsilon: float = 1.0
    # """Epsilon value used for epsilon-greedy decision-making"""
    # tau: float = 0.005
    # """Tau used for epsilon-greedy decision-making"""
    # asym_to_fairness_ratio: float = 0.8
    # """Higher ratio means asymmetrical reward is weighted more compared to
    # fairness score - .8 is 80% asym score reward."""
    # wall_reward: float = 0.1
    # """If nonzero, adds artificial reward for placing walls"""
    # use_baseline: bool = False
    # """Whether to use the simple manhattan distance baseline fairness score, from simple_value_function.py"""

    # def __post_init__(self):
    #     # Ensure that modulo_value is within the range [1, 1000]
    #     if not (1 <= self.step_jump <= 1000):
    #         raise ValueError("modulo_value must be between 1 and 1000, inclusive.")



def generate_maps(args: Args):
    policy_net = Unet().to(device=device)
    policy_net.load_state_dict(torch.load(args.model_path))
    target_net = Unet().to(device=device)
    target_net.load_state_dict(policy_net.state_dict())

    # Load the input map once
    original_xml_map = ET.parse(args.input_map_path)
    # Ensure the output directory exists
    os.makedirs(args.output_map_dir, exist_ok=True)

    for map_idx in range(1, args.num_maps + 1):
        # Create a new map file name for each map
        temp_map_path = f"{args.output_map_dir}/tempMap.xml"
        shutil.copy(args.input_map_path, temp_map_path)
        output_map_path = f"{args.output_map_dir}/map_{map_idx}.xml"
        
        # Copy the input map to the new file path
        shutil.copy(args.input_map_path, args.output_map_dir)
        
        # Reload the XML map for each iteration
        xml_map = ET.ElementTree(file=temp_map_path)
        tensor_map, invalid_actions_mask = convert_xml(xml_map)

        state = tensor_map
        id = 100

        for step in tqdm(range(args.episode_length), desc=f"Generating map {map_idx}"):
            with torch.no_grad():
                q_values = policy_net(state)
                # Mask q_values
                q_values = q_values * (1 - state) * (1 - invalid_actions_mask)[None, :, :]
                action = torch.tensor(
                    torch.unravel_index(torch.argmax(q_values), q_values.shape), device=device
                )
                old_state = state.clone()
                state[:, action[1], action[2]] = 0 # resets all state channels at this xy to 0
                state[action[0], action[1], action[2]] = 1 # sets xy to this channel to 1
                test_index = (
                    old_state[:, action[1].item(), action[2].item()] == 1
                ).nonzero(as_tuple=True)[0] # 

                # Handle potential indexing issues
                if test_index.shape[0] == 2:
                    old_index = test_index[1].item()
                else:
                    old_index = test_index.item()

                update_xml_map(
                    temp_map_path,
                    LayerName(action[0].item()),
                    LayerName(old_index),
                    action[1].item(),
                    action[2].item(),
                    id,
                )
                id += 1
        shutil.copy(temp_map_path, output_map_path)



if __name__ == "__main__":
    args = tyro.cli(Args)
    generate_maps(args)
