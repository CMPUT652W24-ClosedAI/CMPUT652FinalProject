# http://proceedings.mlr.press/v97/han19a/han19a.pdf

import random

import numpy as np
import torch
from gym.spaces import MultiDiscrete
from stable_baselines3.common.vec_env import VecMonitor

from experiments.ppo_gridnet import MicroRTSStatsRecorder, Agent
from gym_microrts import microrts_ai  # noqa
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv


def squared_value_difference(
    map_path: str = "maps/16x16/basesWorkers16x16A.xml",
    seed: int = 0,
    model_path: str = "gym-microrts-static-files/agent_sota.pt",
) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    ais = []
    envs = MicroRTSGridModeVecEnv(
        num_bot_envs=len(ais),
        num_selfplay_envs=2,
        partial_obs=False,
        max_steps=1,
        render_theme=2,
        ai2s=ais,
        map_paths=[map_path],
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
        autobuild=False,
    )
    assert isinstance(
        envs.action_space, MultiDiscrete
    ), "only MultiDiscrete action space is supported"

    agent = Agent(envs).to(device)
    agent2 = Agent(envs).to(device)

    # Note how `next_obs` and `next_done` are used; their usage is equivalent to
    # https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/84a7582477fb0d5c82ad6d850fe476829dddd2e1/a2c_ppo_acktr/storage.py#L60
    next_obs = torch.Tensor(envs.reset()).to(device)

    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()
    agent2.load_state_dict(torch.load(model_path, map_location=device))
    agent2.eval()

    # ALGO LOGIC: put action logic here
    with torch.no_grad():
        invalid_action_masks = torch.tensor(np.array(envs.get_action_mask())).to(device)

        p1_obs = next_obs[::2]
        p2_obs = next_obs[1::2]
        p1_mask = invalid_action_masks[::2]
        p2_mask = invalid_action_masks[1::2]

        p2_obs_reversed = torch.flip(p2_obs, dims=[1, 2])
        p2_mask_reversed = torch.flip(p2_mask, dims=[1])

        # 1, 256, 7
        p1_action, _, _, _, p1_value = agent.get_action_and_value(
            p1_obs, envs=envs, invalid_action_masks=p1_mask, device=device
        )

        p2_action, _, _, _, p2_value = agent2.get_action_and_value(
            p2_obs_reversed,
            envs=envs,
            invalid_action_masks=p2_mask_reversed,
            device=device,
        )

    return (p1_value - p2_value) ** 2


if __name__ == "__main__":
    print(squared_value_difference())
    print(squared_value_difference())
