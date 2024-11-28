import torch

from generator.map_utils import convert_xml

# from generator.value_function_extraction import squared_value_difference
import xml.etree.ElementTree as ET


def sym_score(x):
    x = torch.argmax(x, dim=-3)
    reflected_x = torch.transpose(torch.flip(x, dims=[-1, -2]), -1, -2)
    return torch.sum(x != reflected_x)


if __name__ == "__main__":
    file_path = "input_maps/defaultMap.xml"
    xml_map = ET.parse(file_path)
    tensor_map, invalid_actions_mask = convert_xml(xml_map)
    print(sym_score(tensor_map))
