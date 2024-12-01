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

import atexit
import signal
import sys

from generator.enums import LayerName
from generator.map_utils import update_xml_map, convert_xml
from generator.maploader_controller import MapLoaderController
from generator.memory_buffer import MemoryBuffer, Transition
from generator.run_maploader import run_maploader
from generator.simple_value_function import baseline_fairness_score
from generator.training_script import asym_score
from generator.unet_generator import Unet
from generator.value_function_extraction import squared_value_difference


def delete_temp_file(file_path):
    """Ensure the temp file is deleted."""
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Temporary file {file_path} deleted.")

def handle_sigint(signal_num, frame):
    print("SIGINT received. Cleaning up...")
    sys.exit(0)  # Exit gracefully

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

    use_baseline: bool = False
    """Whether to use the simple manhattan distance baseline fairness score,
    from simple_value_function.py. Used here for the generated fairness score to
    be saved in the output data file"""
    use_random: bool = False
    """Whether to use the random map changer rather than the model output"""

    save_maps: bool = False
    """Whether to save the maps to separate files or just get the data"""

    top_k: int = 1
    """Top k values to use from the model"""



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
    # Extract model parameters from model_path for folder naming
    model_params = os.path.basename(args.model_path).replace(".pt", "")
    timestamp = int(time.time())

    # Modify the output directory to include model parameters
    args.output_map_dir = f"{args.output_map_dir}_model_{model_params}"

    policy_net = Unet().to(device=device)
    policy_net.load_state_dict(torch.load(args.model_path))
    target_net = Unet().to(device=device)
    target_net.load_state_dict(policy_net.state_dict())

    # Load the input map once
    original_xml_map = ET.parse(args.input_map_path)
    # Ensure the output directory exists
    os.makedirs(args.output_map_dir, exist_ok=True)
    # File to store asymmetry and fairness scores
    scores_file_path = os.path.join(args.output_map_dir, "scores.txt")
    with open(scores_file_path, "w") as scores_file:
        scores_file.write("MapIndex,AsymmetryScore,FairnessScore\n")  # Header
    temp_file_path_local = f"maps/16x16/tempMap-{timestamp}.xml"
    temp_file_path = f"../gym_microrts/microrts/{temp_file_path_local}"

    # Ensure the temp file is deleted on exit or interruption
    atexit.register(delete_temp_file, temp_file_path)
    
    # Also handle signals like SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, lambda *args: sys.exit(0))

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        for map_idx in range(1, args.num_maps + 1):
            # Create a new map file name for each map
            temp_map_path = f"{args.output_map_dir}/tempMap.xml"
            shutil.copy(args.input_map_path, temp_map_path)
            if args.save_maps:
                output_map_path = f"{args.output_map_dir}/map_{map_idx}.xml"
            else:
                output_map_path = f"{args.output_map_dir}/map_1.xml"

            # Copy the input map to the new file path
            shutil.copy(args.input_map_path, args.output_map_dir)
            
            # Reload the XML map for each iteration
            xml_map = ET.ElementTree(file=temp_map_path)
            tensor_map, invalid_actions_mask = convert_xml(xml_map)

            state = tensor_map
            id = 100

            for step in tqdm(range(args.episode_length), desc=f"Generating map {map_idx}"):
                with torch.no_grad():
                    if args.use_random:
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
                    else: 
                        q_values = policy_net(state)
                        # Mask q_values
                        q_values = q_values * (1 - state) * (1 - invalid_actions_mask)[None, :, :]
                        # Get the top N flat indices
                        
                        topk_values, topk_indices = torch.topk(q_values.flatten(), args.top_k)

                        # Randomly select one of the top N actions
                        selected_index = random.choice(topk_indices)
                        # Convert the flat index back to multidimensional indices
                        # action_indices = torch.unravel_index(selected_index.item(), q_values.shape)
                        # action = torch.tensor(action_indices, device=device)

                        action = torch.tensor(
                            torch.unravel_index(selected_index, q_values.shape), device=device
                        ) # 749 is torch.argmax

                    #     torch.unravel_index(torch.argmax(q_values), q_values.shape), device=device
                    # ) # 749 is torch.argmax
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
                    
                    # Compute scores
            with torch.no_grad():
                shutil.copy(
                    temp_map_path, temp_file_path
                )
                # temp_file_path_local = f"maps/16x16/tempMap-{timestamp}.xml"
                # temp_file_path = f"../gym_microrts/microrts/{temp_file_path_local}"
                asym_score_output = asym_score(state).float().item()  # Asymmetry score
                fairness_score_output = (
                    # squared_value_difference(temp_map_path)
                    squared_value_difference(temp_file_path_local)
                    .reshape(-1)
                    .squeeze(0)
                    .item()
                    if not args.use_baseline
                    else baseline_fairness_score(state).item()
                )

            # Save the scores to the file
            with open(scores_file_path, "a") as scores_file:
                scores_file.write(f"{map_idx},{asym_score_output},{fairness_score_output}\n")

            shutil.copy(temp_map_path, output_map_path)
            print(f"maps stored in {args.output_map_dir}")
    finally:
        # Cleanup temp file
        delete_temp_file(temp_file_path)
        print("deleted temp file!")




if __name__ == "__main__":
    args = tyro.cli(Args)
    generate_maps(args)
