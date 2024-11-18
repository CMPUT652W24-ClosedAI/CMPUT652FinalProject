import shutil

import torch
import xml.etree.ElementTree as ET

from generator.enums import LayerName
from generator.map_utils import update_xml_map, convert_xml
from generator.unet_generator import Unet


def generate_map(map_path: str):
    policy_net = Unet()
    policy_net.load_state_dict(torch.load("policy_net.pt"))
    target_net = Unet()
    target_net.load_state_dict(policy_net.state_dict())

    # Test Using convert xml
    shutil.copy("defaultMap.xml", "outputMap.xml")
    file_path = "outputMap.xml"
    xml_map = ET.parse(file_path)
    tensor_map, invalid_actions_mask = convert_xml(xml_map)

    state = tensor_map
    id = 100

    for step in range(100):
        with torch.no_grad():
            q_values = policy_net(state)
            # mask q_values
            q_values = q_values * (1 - state) * (1 - invalid_actions_mask)[None, :, :]
            action = torch.tensor(torch.unravel_index(torch.argmax(q_values), q_values.shape))
            old_state = state.clone()
            state[:, action[1], action[2]] = 0
            state[action[0], action[1], action[2]] = 1
            test_index = (old_state[:, action[1].item(), action[2].item()] == 1).nonzero(as_tuple=True)[0]

            # Sometimes happens when a square is marked as empty and has a resource
            # TODO: This is likely because of a bug elsewhere. This should be fixed.
            if test_index.shape[0] == 2:
                old_index = test_index[1].item()
            else:
                old_index = test_index.item()

            update_xml_map("outputMap.xml", LayerName(action[0].item()), LayerName(old_index), action[1].item(), action[2].item(), id)
            id += 1


if __name__ == '__main__':
    generate_map("defaultMap.xml")