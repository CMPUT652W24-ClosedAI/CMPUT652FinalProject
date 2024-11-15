from xml.etree.ElementTree import ElementTree

import torch
from functorch.dim import Tensor
import xml.etree.ElementTree as ET

from generator.enums import LayerName

ALLOWED_RESOURCE_VALUES = {5, 10, 15, 20}

def convert_xml(xml_map: ElementTree) -> (Tensor, Tensor):
    root = xml_map.getroot()
    map_width = int(root.attrib['width'])
    map_height = int(root.attrib['height'])
    if map_width != map_height:
        raise Exception("Width and height dimensions do not match!")

    # Extract Terrain Features
    terrain_data = root.find('terrain').text.strip()
    terrain_values = [int(value) for value in terrain_data]  # Convert each character to an integer
    terrain_tensor = torch.tensor(terrain_values, dtype=torch.float32).reshape(map_width, map_height)
    empty_space_tensor = (terrain_tensor == 0).type(torch.float32)
    wall_tensor = (terrain_tensor == 1).type(torch.float32)

    # Process Units
    units = root.find('units')
    resources = []
    other_units = []
    for unit in units:
        if unit.attrib.get("type") == "Resource":
            if int(unit.attrib.get("resources")) not in ALLOWED_RESOURCE_VALUES:
                raise Exception("Invalid resource amount!")
            resources.append(unit)
        else:
            other_units.append(unit)

    # Create Resource Tensors
    five_resource_tensor = torch.zeros_like(wall_tensor, dtype=torch.float32)
    ten_resource_tensor = torch.zeros_like(wall_tensor, dtype=torch.float32)
    fifteen_resource_tensor = torch.zeros_like(wall_tensor, dtype=torch.float32)
    twenty_resource_tensor = torch.zeros_like(wall_tensor, dtype=torch.float32)
    map_tensor = torch.stack([empty_space_tensor, wall_tensor, five_resource_tensor, ten_resource_tensor, fifteen_resource_tensor, twenty_resource_tensor], dim=0)

    for resource in resources:
        x_cord = int(resource.attrib["x"])
        y_cord = int(resource.attrib["x"])
        resource_amount = int(resource.attrib["resources"])
        layer = LayerName.FIVE_RESOURCES if resource_amount == 5 else LayerName.TEN_RESOURCES if resource_amount == 10 else LayerName.FIFTEEN_RESOURCES if resource_amount == 15 else LayerName.TWENTY_RESOURCES
        map_tensor[layer.value, x_cord, y_cord] = 1.0

    """
    Extract Invalid Actions
    This is changing the tiles that workers and bases are initially on.
    """
    invalid_actions_map = torch.zeros_like(wall_tensor, dtype=torch.float32)
    for other in other_units:
        x_cord = int(other.attrib["x"])
        y_cord = int(other.attrib["x"])
        invalid_actions_map[ x_cord, y_cord] = 1.0

    return (map_tensor, invalid_actions_map)

def update_xml_map(file_path, xml_map, layer_update: LayerName, layer_old: LayerName, x_cord, y_cord):
    root = xml_map.getroot()
    map_height = int(root.attrib['height'])

    if layer_update.value < 2:
        terrain = root.find('terrain').text.strip()
        update_index = y_cord * map_height + x_cord
        updated_terrain = list(terrain)
        updated_terrain[update_index] = layer_update.value



    xml_map.write(file_path)


def update_tensor_map():
    print("Not implemented")


if __name__ == "__main__":
    file_path = "basesWorkers16x16.xml"
    xml_map = ET.parse(file_path)
    convert_xml(xml_map)
    update_xml_map(file_path, xml_map, LayerName.EMPTY, 1, 10, LayerName.TWENTY_RESOURCES)
