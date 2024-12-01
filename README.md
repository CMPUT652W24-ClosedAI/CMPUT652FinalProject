# CMPUT652FinalProject

- Seth Akins
- Andrew Freeman
- Nicolas Ong
- Justin Rozeboom
- Sam Scholnick-Hughes


## Info

Forked from https://github.com/Farama-Foundation/MicroRTS-Py at b6bf191 on Oct. 25, 2024


## Setup Instructions

## Get Started

> [!note]
> Andrew's Installation instructions notes can be found in docs/andrews-install-notes.md

> [!note]
> Pre-existing experiments can be found in ./experiments - and one had been preloaded for use in vscode by .vscode/launch.json

Prerequisites:
- Python 3.9.7 (use pyenv or dead snakes to manage your version)
- [Poetry](https://python-poetry.org) (Use the official installer, not pipx)
- Java 8.0+
- FFmpeg (for video recording utilities)

```bash
$ git clone --recursive https://github.com/CMPUT652W24-ClosedAI/CMPUT652FinalProject.git && \
cd CMPUT652FinalProject
poetry install
# The `poetry install` command above creates a virtual environment for us, in which all the dependencies are installed.
# We can use `poetry shell` to create a new shell in which this environment is activated. Once we are done working with
# MicroRTS, we can leave it again using `exit`.
poetry shell
# By default, the torch wheel is built with CUDA 10.2. If you are using newer NVIDIA GPUs (e.g., 3060 TI), you may need to specifically install CUDA 11.3 wheels by overriding the torch dependency with pip:
# poetry run pip install "torch==1.12.1" --upgrade --extra-index-url https://download.pytorch.org/whl/cu113
python hello_world.py
```

If the `poetry install` command gets stuck on a Linux machine, [it may help to first run](https://github.com/python-poetry/poetry/issues/8623): `export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring`.

## Linting Code

- There is an action on this repository which automatically makes sure code is linted properly with black before it can be merged.
```bash
# To run the linter, use the following command:
poetry run black .
```


## Running with args
cd generator
python3 training_script.py --num-episodes 1000 --episode-length 64 --replay-buffer-size 1000 --step-jump 4 --asym_to_fairness_ratio 0.8 --wall-reward 0.1

python3 map_generator.py --input-map-path input_maps/defaultMap.xml --model-path "models/output/training_net_output___1732838472 episodes 1000 --episode-length 32 --replay-buffer-size 100 --step-jump 1 --asym_to_fairness_ratio 0.8 --wall-reward 0.1 --visualize_maps --use_baseline.pt" --num-maps 100 --episode_length 64


cd maploader
java -cp "lib/*:src:." MapLoader

../generator/outputMaps/__1733021997 training_net_output___1733013680 numep100 eplength 64 replay-buffer-size 1000 step-jump 4 ratio 0.8 wallreward 0.1/map_96.xml


../generator/outputMaps/__1733023420_model_training_net_eps_1.0_tau_0.005_ratio_0.8_wr_0.1_ep_1000_len_64_baseline_True___1733022366/map_100.xml
