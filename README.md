# CMPUT652FinalProject

- Seth Akins
- Andrew Freeman
- Nicolas Ong
- Justin Rozeboom
- Sam Scholnick-Hughes

## Setup Instructions

## Get Started

Prerequisites:
- Python 3.9.7 (use pyenv or dead snakes to manage your version)
- [Poetry](https://python-poetry.org) (Use the official installer, not pipx)
- Java 8.0+
- FFmpeg (for video recording utilities)

```bash
$ git clone --recursive https://github.com/Farama-Foundation/MicroRTS-Py.git && \
cd MicroRTS-Py
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
