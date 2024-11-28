from xml.etree.ElementTree import ElementTree

import torch
from functorch.dim import Tensor
import xml.etree.ElementTree as ET

from generator.enums import LayerName

ALLOWED_RESOURCE_VALUES = {5, 10, 15, 20}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def convert_xml(xml_map: ElementTree) -> (Tensor, Tensor):
    root = xml_map.getroot()
    map_width = int(root.attrib["width"])
    map_height = int(root.attrib["height"])
    if map_width != map_height:
        raise Exception("Width and height dimensions do not match!")

    # Extract Terrain Features
    terrain_data = root.find("terrain").text.strip()
    terrain_values = [
        int(value) for value in terrain_data
    ]  # Convert each character to an integer
    terrain_tensor = torch.tensor(terrain_values, dtype=torch.float32, device=device).reshape(
        map_width, map_height
    )
    empty_space_tensor = (terrain_tensor == 0).type(torch.float32)
    wall_tensor = (terrain_tensor == 1).type(torch.float32)

    # Process Units
    units = root.find("units")
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
    five_resource_tensor = torch.zeros_like(wall_tensor, dtype=torch.float32, device=device)
    ten_resource_tensor = torch.zeros_like(wall_tensor, dtype=torch.float32, device=device)
    fifteen_resource_tensor = torch.zeros_like(wall_tensor, dtype=torch.float32, device=device)
    twenty_resource_tensor = torch.zeros_like(wall_tensor, dtype=torch.float32, device=device)
    map_tensor = torch.stack(
        [
            empty_space_tensor,
            wall_tensor,
            five_resource_tensor,
            ten_resource_tensor,
            fifteen_resource_tensor,
            twenty_resource_tensor,
        ],
        dim=0,
    )

    for resource in resources:
        x_cord = int(resource.attrib["x"])
        y_cord = int(resource.attrib["y"])
        resource_amount = int(resource.attrib["resources"])
        layer = LayerName(int(resource_amount / 5) + 1)
        map_tensor[layer.value, x_cord, y_cord] = 1.0

    """
    Extract Invalid Actions
    This is changing the tiles that workers and bases are initially on.
    """
    invalid_actions_map = torch.zeros_like(wall_tensor, dtype=torch.float32, device=device)
    for other in other_units:
        x_cord = int(other.attrib["x"])
        y_cord = int(other.attrib["x"])
        invalid_actions_map[x_cord, y_cord] = 1.0

    return (map_tensor, invalid_actions_map)


def update_xml_map(
    file_path,
    layer_update: LayerName,
    layer_old: LayerName,
    x_cord: int,
    y_cord: int,
    id: int,
):
    xml_map = ET.parse(file_path)
    root = xml_map.getroot()
    map_height = int(root.attrib["height"])

    # Update terrain text
    terrain_node = root.find("terrain")
    terrain_text = terrain_node.text
    update_index = y_cord * map_height + x_cord
    updated_terrain = (
        terrain_text[:update_index]
        + ("1" if layer_update.value == 1 else "0")
        + terrain_text[update_index + 1 :]
    )
    terrain_node.text = updated_terrain
    # Unit updates
    units = root.find("units")
    resource_amount = (
        5
        if layer_update == LayerName.FIVE_RESOURCES
        else (
            10
            if layer_update == LayerName.TEN_RESOURCES
            else 15 if layer_update == LayerName.FIFTEEN_RESOURCES else 20
        )
    )
    resource_amount = str(resource_amount)

    # Change Existing Resource Amount
    if layer_old.value > 1 and layer_update.value > 1:
        for unit in units:
            if (
                int(unit.attrib.get("x")) == x_cord
                and int(unit.attrib.get("y")) == y_cord
            ):
                unit.set("resources", resource_amount)
                break
    elif layer_old.value < 2 and layer_update.value > 1:
        # We need to add a new sub-element to the XML
        ET.SubElement(
            units,
            "rts.units.Unit",
            type="Resource",
            ID=str(id),
            player="-1",
            x=str(x_cord),
            y=str(y_cord),
            resources=resource_amount,
            hitpoints="1",
        )
    elif layer_old.value > 1 and layer_update.value < 2:
        for unit in units:
            if (
                int(unit.attrib.get("x")) == x_cord
                and int(unit.attrib.get("y")) == y_cord
            ):
                units.remove(unit)
                break

    # Hopefully fix formatting
    ET.indent(root)
    xml_map.write(file_path)


def tensor_to_xml_map(map_tensor: Tensor):
    root = ET.Element(
        "rts.PhysicalGameState",
        width=str(map_tensor.shape[1]),
        height=str(map_tensor.shape[2]),
    )
    terrain = ET.SubElement(root, "terrain")


if __name__ == "__main__":
    file_path = "input_maps/defaultMap.xml"
    xml_map = ET.parse(file_path)
    map_tensor, invalid_actions = convert_xml(xml_map)
    #
    # # Change existing resource amount
    # update_xml_map(
    #     file_path, LayerName.TWENTY_RESOURCES, LayerName.FIVE_RESOURCES, 15, 15
    # )
    # # Remove old tag
    # update_xml_map(file_path, LayerName.WALL, LayerName.TEN_RESOURCES, 0, 0)
    # # Add new tag
    # update_xml_map(
    #     file_path, LayerName.FIFTEEN_RESOURCES, LayerName.EMPTY, 8, 4
    # )

    tensor_to_xml_map(map_tensor)
