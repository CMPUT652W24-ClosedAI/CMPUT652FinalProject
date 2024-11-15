import torch
from functorch.dim import Tensor
import xml.etree.ElementTree as ET

from generator.enums import LayerName

ALLOWED_RESOURCE_VALUES = {5, 10, 15, 20}

def convert_xml(file_path: str) -> Tensor:
    xml_map = ET.parse(file_path)
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


    """
    Extract Invalid Actions
    This is changing the tiles that workers and bases are initially on.
    """




if __name__ == "__main__":
    file_path = "basesWorkers16x16.xml"
    convert_xml(file_path)
