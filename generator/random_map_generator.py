# Andrew obseleted these changes by adding similar functionality to map
# generator with the flag --use-random
import shutil

import numpy as np
import torch
import xml.etree.ElementTree as ET

from generator.enums import LayerName
from generator.map_utils import update_xml_map, convert_xml
from generator.simple_value_function import baseline_fairness_score
from generator.value_function_extraction import squared_value_difference

device = torch.device("cuda:0") if getattr(torch, "cuda", None) and torch.cuda.is_available() else torch.device("cpu")

def generate_random_map(input_map_path: str, output_map_path: str, evaluation = False, useBaseline = False):
    shutil.copy(input_map_path, output_map_path)
    xml_map = ET.parse(output_map_path)
    tensor_map, invalid_actions_mask = convert_xml(xml_map)
    state = tensor_map
    id = 100

    for step in range(64):
        with torch.no_grad():
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

            old_state = state.clone()
            state[:, action[1], action[2]] = 0
            state[action[0], action[1], action[2]] = 1
            test_index = (
                old_state[:, action[1].item(), action[2].item()] == 1
            ).nonzero(as_tuple=True)[0]

            if test_index.shape[0] == 2:
                old_index = test_index[1].item()
            else:
                old_index = test_index.item()

            update_xml_map(
                output_map_path,
                LayerName(action[0].item()),
                LayerName(old_index),
                action[1].item(),
                action[2].item(),
                id,
            )
            id += 1

    if evaluation:
        asym_score_output = asym_score(state).float()

        shutil.copy(
            "tempMap.xml", "../gym_microrts/microrts/maps/16x16/tempMap.xml"
        )
        fairness_score_output = (
            squared_value_difference("maps/16x16/tempMap.xml")
            .reshape(-1)
            .squeeze(0)
        ) if not useBaseline else baseline_fairness_score(state)
        return (asym_score_output, fairness_score_output)


def asym_score(x):
    x = torch.argmax(x, dim=-3)
    reflected_x = torch.transpose(torch.flip(x, dims=[-1, -2]), -1, -2)
    return torch.sum(x != reflected_x)


if __name__ == "__main__":
    x, y = generate_random_map(
        "input_maps/defaultMap.xml",
        "generated_maps/random.xml",
        evaluation = True,
    )
    print(f"Aymmetry Score: {x}, Fairness Score: {y}")
