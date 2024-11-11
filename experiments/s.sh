# NOTES: The --prod-mode is used to run the wanddb evaluator. The new version of
# wanddb which works with our current numpy version will ask us to make an
# account every time we want to use it. It may prove useful but for now we can
# just disable it by not adding the --prod-mode flag.


# Andrew added

python ppo_gridnet_andrew.py \
    --agent-model-path gym-microrts-static-files/agent_sota.pt \
    --agent2-model-path gym-microrts-static-files/agent_sota.pt \


python ppo_gridnet_large.py \
    --total-timesteps 300000000 \
    --agent-model-path agent_sota.pt \
    --num-bot-envs 1 \
    --num-selfplay-envs 0 \
    --partial-obs False \
    --capture-video


# num-bot-envs is number f games
# 

# full obs
## training against diverse bots
xvfb-run -a python ppo_gridnet.py \
    --total-timesteps 300000000 \
    --num-bot-envs 24 \
    --num-selfplay-envs 0 \
    --partial-obs False \
    --prod-mode --capture-video

## training using selfplay
xvfb-run -a python ppo_gridnet.py \
    --total-timesteps 300000000 \
    --num-bot-envs 0 \
    --num-selfplay-envs 24 \
    --partial-obs False \
    --prod-mode --capture-video

## evaluating against a particular AI (in this cas)
python ppo_gridnet_eval.py \
    --agent-model-path agent_sota.pt \
    --num-selfplay-envs 0 \
    --ai randomBiasedAI 

## evaluating against selfplay
python ppo_gridnet_eval.py \
    --agent-model-path agent_sota.pt \
    --num-selfplay-envs 2

# partial obs
## training against diverse bots
xvfb-run -a python ppo_gridnet.py \
    --total-timesteps 300000000 \
    --num-bot-envs 24 \
    --num-selfplay-envs 0 \
    --partial-obs True \
    --prod-mode --capture-video

## training using selfplay
xvfb-run -a python ppo_gridnet.py \
    --total-timesteps 300000000 \
    --num-bot-envs 0 \
    --num-selfplay-envs 24 \
    --partial-obs True \
    --prod-mode --capture-video

## evaluating against a particular AI (in this cas)
python ppo_gridnet_eval.py \
    --agent-model-path agent_po.pt \
    --num-selfplay-envs 0 \
    --partial-obs True \
    --ai randomBiasedAI 

## evaluating against selfplay
python ppo_gridnet_eval.py \
    --agent-model-path agent_po.pt \
    --partial-obs True \
    --num-selfplay-envs 2

WANDB_ENTITY=vwxyzjn WANDB_PROJECT=gym-microrts-league python league.py --prod-mode\
    --built-in-ais randomBiasedAI workerRushAI lightRushAI coacAI randomAI passiveAI naiveMCTSAI mixedBot rojo izanagi tiamat droplet guidedRojoA3N \
    --rl-ais agent_sota.pt
