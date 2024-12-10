# http://proceedings.mlr.press/v97/han19a/han19a.pdf

import argparse
import csv
import itertools
from multiprocessing import Pool, cpu_count
import os
import random
import time
from distutils.util import strtobool

import numpy as np
import torch
from gym.spaces import MultiDiscrete
from stable_baselines3.common.vec_env import VecMonitor, VecVideoRecorder
from gym_microrts import microrts_ai  # noqa`


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
        help='the name of this experiment')
    parser.add_argument('--map_name', type=str, default="defaultMap",
        help='the name of this experiment')
    parser.add_argument('--map_path', type=str, default="maps/testingMaps",
        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="MicroRTSGridModeVecEnv",
        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=11111,
        help='seed of the experiment')
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod-mode', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='whether to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
        help="the entity (team) of wandb's project")

    # Algorithm specific arguments
    parser.add_argument('--partial-obs', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, the game will have partial observability')
    parser.add_argument('--num-steps', type=int, default=5000,
        help='the number of steps per game environment')
    parser.add_argument("--agent-model-path", type=str, default="gym-microrts-static-files/agent_sota.pt",
        help="the path to the agent's model")
    parser.add_argument("--agent2-model-path", type=str, default="gym-microrts-static-files/agent_sota.pt",
        help="the path to the agent's model")
    parser.add_argument('--ai', type=str, default="",
        help='the opponent AI to evaluate against')
    parser.add_argument('--model-type', type=str, default=f"ppo_gridnet", choices=["ppo_gridnet_large", "ppo_gridnet"],
        help='the output path of the leaderboard csv')
    parser.add_argument('--data_dir', type=str, default=f"results/self_play",
        help='the output path of the self play csv')
    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())
    args.num_bot_envs, args.num_selfplay_envs = 0, 2
    args.num_envs = args.num_selfplay_envs + args.num_bot_envs
    # fmt: on
    return args


def self_play(args, map_name: str, seed: int):
    if args.model_type == "ppo_gridnet_large":
        from ppo_gridnet_large import Agent, MicroRTSStatsRecorder

        from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
    else:
        from ppo_gridnet import Agent, MicroRTSStatsRecorder

        from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv

    # TRY NOT TO MODIFY: setup the environment
    experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # TRY NOT TO MODIFY: seeding
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    ais = []
    if args.ai:
        ais = [eval(f"microrts_ai.{args.ai}")]
    envs = MicroRTSGridModeVecEnv(
        num_bot_envs=len(ais),
        num_selfplay_envs=args.num_selfplay_envs,
        partial_obs=args.partial_obs,
        max_steps=10_000,
        render_theme=2,
        ai2s=ais,
        map_paths=[f"{args.map_path}/{map_name}.xml"],
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
        autobuild=False
    )
    envs = MicroRTSStatsRecorder(envs)
    envs = VecMonitor(envs)
    if args.capture_video:
        envs = VecVideoRecorder(
            envs,
            f"videos/{experiment_name}",
            record_video_trigger=lambda x: x % 100000 == 0,
            video_length=4800,
        )
    assert isinstance(
        envs.action_space, MultiDiscrete
    ), "only MultiDiscrete action space is supported"

    agent = Agent(envs).to(device)
    agent2 = Agent(envs).to(device)

    agent.load_state_dict(torch.load(args.agent_model_path, map_location=device))
    agent.eval()
    agent2.load_state_dict(torch.load(args.agent2_model_path, map_location=device))
    agent2.eval()

    num_episodes = 5
    data = [-2] * num_episodes
    for episode in range(num_episodes):
        next_obs = torch.Tensor(envs.reset()).to(device)
        next_done = torch.zeros(args.num_envs).to(device)

        for step in range(0, args.num_steps):
            with torch.no_grad():
                invalid_action_masks = torch.tensor(
                    np.array(envs.get_action_mask())
                ).to(device)

                p1_obs = next_obs[::2]
                p2_obs = next_obs[1::2]
                p1_mask = invalid_action_masks[::2]
                p2_mask = invalid_action_masks[1::2]

                p2_obs_reversed = torch.flip(p2_obs, dims=[1, 2])
                p2_mask_reversed = p2_mask.reshape(1, 16, 16, 78).flip(dims=[1, 2]).reshape(1, 256, 78)

                p2_mask_reversed[..., [6, 8]] = p2_mask_reversed[..., [8, 6]]
                p2_mask_reversed[..., [7, 9]] = p2_mask_reversed[..., [9, 7]]
                p2_mask_reversed[..., [10, 12]] = p2_mask_reversed[..., [12, 10]]
                p2_mask_reversed[..., [11, 13]] = p2_mask_reversed[..., [13, 11]]
                p2_mask_reversed[..., [14, 16]] = p2_mask_reversed[..., [16, 14]]
                p2_mask_reversed[..., [15, 17]] = p2_mask_reversed[..., [17, 15]]
                p2_mask_reversed[..., [18, 20]] = p2_mask_reversed[..., [20, 18]]
                p2_mask_reversed[..., [19, 21]] = p2_mask_reversed[..., [21, 19]]

                p1_action, _, _, _, _ = agent.get_action_and_value(
                    p1_obs, envs=envs, invalid_action_masks=p1_mask, device=device
                )

                p2_action, _, _, _, _ = agent2.get_action_and_value(
                    p2_obs_reversed, envs=envs, invalid_action_masks=p2_mask_reversed, device=device
                )

                p2_fixed_action = p2_action.reshape(1, 16, 16, 7).flip(dims=[1, 2]).reshape(1, 256, 7)
                p2_fixed_action[:, :, 1: 5] = (p2_fixed_action[:, :, 1: 5] + 2 )% 4

                p2_relative_attack = torch.clone(p2_fixed_action[:, :, -1])

                p2_fixed_action[:, :, -1] = torch.where(p2_relative_attack == 31, torch.tensor(-1), p2_relative_attack)
                p2_fixed_action[:, :, -1] = torch.where(p2_relative_attack == 17, torch.tensor(31),p2_relative_attack)
                p2_fixed_action[:, :, -1] = torch.where(p2_relative_attack == -1, torch.tensor(17),p2_relative_attack)

                p2_fixed_action[:, :, -1] = torch.where(p2_relative_attack == 25, torch.tensor(-1), p2_relative_attack)
                p2_fixed_action[:, :, -1] = torch.where(p2_relative_attack == 23, torch.tensor(25), p2_relative_attack)
                p2_fixed_action[:, :, -1] = torch.where(p2_relative_attack == -1, torch.tensor(23), p2_relative_attack)

                action = torch.zeros(
                    (args.num_envs, p2_action.shape[1], p2_action.shape[2])
                )

                action[::2] = p1_action
                action[1::2] = p2_fixed_action

            try:
                next_obs, rs, ds, infos = envs.step(
                    action.cpu().numpy().reshape(envs.num_envs, -1)
                )
                next_obs = torch.Tensor(next_obs).to(device)
            except Exception as e:
                e.printStackTrace()
                raise

            for idx, info in enumerate(infos):
                if "episode" in info.keys():
                    if args.ai:
                        print(
                            "against",
                            args.ai,
                            info["microrts_stats"]["WinLossRewardFunction"],
                        )
                    else:
                        if idx % 2 == 0:
                            print(
                                f"player{idx % 2}",
                                info["microrts_stats"]["WinLossRewardFunction"],
                            )
                            data[episode] = info["microrts_stats"]["WinLossRewardFunction"]
            if ds[0] or ds[1]:
                break

    # Save data
    directory = f"{args.data_dir}/{map_name}/{args.seed}"
    if len(data) == 0:
        data.append(-2)

    os.makedirs(directory, exist_ok=True)

    with open(os.path.join(directory, "results.csv"), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)
    envs.close()

def run_parallel(args):
    num_maps = range(1, 101)
    map_nams = [f"map_{i}" for i in num_maps]
    seeds = [np.random.randint(1, 10000001) for i in range(10)]
    args_list = [args]

    combinations = list(itertools.product(args_list, map_nams, seeds))

    with Pool(6) as pool:
        pool.starmap(self_play, combinations)


if __name__ == "__main__":
    args = parse_args()
    run_parallel(args)
    # self_play(args, args.map_name, args.seed)
