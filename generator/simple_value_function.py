import xml.etree.ElementTree as ET
from generator.map_utils import update_xml_map, convert_xml

import torch

def simple_value_function(state, player=0):
    #empty_space_tensor = state[0]
    #wall_tensor = state[1]
    five_resource_tensor = state[2]
    ten_resource_tensor = state[3]
    fifteen_resource_tensor = state[4]
    twenty_resource_tensor = state[5]

    p0_coords = (1, 1)
    p1_coords = (14, 14)
    x, y = p0_coords
    if player == 1:
        x, y = p1_coords

    height, width = five_resource_tensor.shape

    # Create a grid of coordinates
    y_coords, x_coords = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij"
    )
    
    # Calculate distances
    #distances = torch.sqrt((x_coords - x)**2 + (y_coords - y)**2)
    distances = torch.abs(x_coords - x) + torch.abs(y_coords - y)
    distances[x, y] = 1

    score_grid = (five_resource_tensor / distances) + (ten_resource_tensor / distances) + (fifteen_resource_tensor / distances) + (twenty_resource_tensor / distances)
    score = torch.sum(score_grid)
    
    return score



if __name__ == "__main__":
    file_path = "input_maps/defaultMap.xml"
    xml_map = ET.parse(file_path)
    tensor_map, invalid_actions_mask = convert_xml(xml_map)
    simple_value_function(tensor_map, player=0)
