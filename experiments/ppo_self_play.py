# http://proceedings.mlr.press/v97/han19a/han19a.pdf

import argparse
import csv
import os
import random
import signal
import time
from distutils.util import strtobool

import numpy as np
import torch
from gym.spaces import MultiDiscrete
from stable_baselines3.common.vec_env import VecMonitor, VecVideoRecorder
from torch.utils.tensorboard import SummaryWriter

from gym_microrts import microrts_ai  # noqa`


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
        help='the name of this experiment')
    parser.add_argument('--mapName', type=str, default="map_1.xml",
        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="MicroRTSGridModeVecEnv",
        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=11111,
        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=5120,
        help='total timesteps of the experiments')
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
    parser.add_argument('--num-steps', type=int, default=256,
        help='the number of steps per game environment')
    parser.add_argument("--agent-model-path", type=str, default="gym-microrts-static-files/agent_sota.pt",
        help="the path to the agent's model")
    parser.add_argument("--agent2-model-path", type=str, default="gym-microrts-static-files/agent_sota.pt",
        help="the path to the agent's model")
    parser.add_argument('--ai', type=str, default="",
        help='the opponent AI to evaluate against')
    parser.add_argument('--model-type', type=str, default=f"ppo_gridnet", choices=["ppo_gridnet_large", "ppo_gridnet"],
        help='the output path of the leaderboard csv')
    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())
    if args.ai:
        args.num_bot_envs, args.num_selfplay_envs = 1, 0
    else:
        args.num_bot_envs, args.num_selfplay_envs = 0, 2
    args.num_envs = args.num_selfplay_envs + args.num_bot_envs
    args.batch_size = int(args.num_envs * args.num_steps)
    args.num_updates = args.total_timesteps // args.batch_size
    # fmt: on
    return args


def main(args):
    data = []
    if args.model_type == "ppo_gridnet_large":
        from ppo_gridnet_large import Agent, MicroRTSStatsRecorder

        from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
    else:
        from ppo_gridnet import Agent, MicroRTSStatsRecorder

        from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv

    # TRY NOT TO MODIFY: setup the environment
    experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.prod_mode:
        import wandb

        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=experiment_name,
            monitor_gym=True,
            save_code=True,
        )
        CHECKPOINT_FREQUENCY = 10
    writer = SummaryWriter(f"runs/{experiment_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

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
        max_steps=5000,
        render_theme=2,
        ai2s=ais,
        map_paths=[f"maps/testingMaps/{args.mapName}.xml"],
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
    )
    envs = MicroRTSStatsRecorder(envs)
    envs = VecMonitor(envs)
    if args.capture_video:
        envs = VecVideoRecorder(
            envs,
            f"videos/{experiment_name}",
            record_video_trigger=lambda x: x % 100000 == 0,
            video_length=2000,
        )
    assert isinstance(
        envs.action_space, MultiDiscrete
    ), "only MultiDiscrete action space is supported"

    agent = Agent(envs).to(device)
    agent2 = Agent(envs).to(device)

    # ALGO Logic: Storage for epoch data
    mapsize = 16 * 16
    invalid_action_shape = (mapsize, envs.action_plane_space.nvec.sum())

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    # Note how `next_obs` and `next_done` are used; their usage is equivalent to
    # https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/84a7582477fb0d5c82ad6d850fe476829dddd2e1/a2c_ppo_acktr/storage.py#L60
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    ## CRASH AND RESUME LOGIC:
    starting_update = 1
    agent.load_state_dict(torch.load(args.agent_model_path, map_location=device))
    agent.eval()
    if not args.ai:
        agent2.load_state_dict(torch.load(args.agent2_model_path, map_location=device))
        agent2.eval()

    print("Model's state_dict:")
    for param_tensor in agent.state_dict():
        print(param_tensor, "\t", agent.state_dict()[param_tensor].size())
    total_params = sum([param.nelement() for param in agent.parameters()])
    print("Model's total parameters:", total_params)

    for update in range(starting_update, args.num_updates + 1):
        # TRY NOT TO MODIFY: prepare the execution of the game.
        for step in range(0, args.num_steps):
            envs.render()
            global_step += 1 * args.num_envs
            # ALGO LOGIC: put action logic here
            with torch.no_grad():
                invalid_action_masks = torch.tensor(
                    np.array(envs.get_action_mask())
                ).to(device)

                if args.ai:
                    action, logproba, _, _, vs = agent.get_action_and_value(
                        next_obs,
                        envs=envs,
                        invalid_action_masks=invalid_action_masks,
                        device=device,
                    )
                else:
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

                    # other_seed = 3721

                    # random.seed(args.seed)
                    # np.random.seed(args.seed)
                    # torch.manual_seed(args.seed)
                    # torch.backends.cudnn.deterministic = args.torch_deterministic

                    p1_action, _, _, _, _ = agent.get_action_and_value(
                        p1_obs, envs=envs, invalid_action_masks=p1_mask, device=device
                    )

                    # random.seed(args.seed)
                    # np.random.seed(args.seed)
                    # torch.manual_seed(args.seed)
                    # torch.backends.cudnn.deterministic = args.torch_deterministic

                    p2_action, _, _, _, _ = agent2.get_action_and_value(
                        p2_obs_reversed, envs=envs, invalid_action_masks=p2_mask_reversed, device=device
                    )

                    p2_relative_attack = torch.clone(p2_action[:, :, -1])
                    p2_fixed_action = p2_action.reshape(1, 16, 16, 7).flip(dims=[1, 2]).reshape(1, 256, 7)
                    p2_fixed_action[:, :, 1: 5] = (p2_fixed_action[:, :, 1: 5] + 2 )% 4
                    #
                    p2_relative_attack = torch.clone(p2_fixed_action[:, :, -1])
                    # x = (p2_relative_attack % 7)
                    # y = (p2_relative_attack // 7)
                    # x_prime =( 6 - x)
                    # y_prime = (6 - y)
                    # result = (7 * y_prime) + x_prime
                    # p2_fixed_action[:, :, -1] = result
                    #

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
                            data.append(info["microrts_stats"]["WinLossRewardFunction"])
    filename = f"self_play_results/{args.mapName}/{args.seed}/results.csv"
    if len(data) == 0:
        data.append(-2)

    os.makedirs(f"self_play_results/{args.mapName}/{args.seed}", exist_ok=True)

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

    os.kill(os.getpid(), signal.SIGKILL)

if __name__ == "__main__":
    args = parse_args()
    main(args)
