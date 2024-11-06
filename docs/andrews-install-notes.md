# Andrew's Install notes

## Ubuntu Installation

### Pyenv

#### Installation
>[!note]
>Requires python build dependencies because it compiles the python version locally in order to install. Can be found here: <https://github.com/pyenv/pyenv/wiki#suggested-build-environment>. 
>TLDR IS:

```sh
sudo apt update; sudo apt install build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev curl git \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev llvm
```
This is like 700mb of space though...


Requires Pyenv for getting version above 3.9.7 (can't do 3.10 or higher though I think...):

<https://github.com/pyenv/pyenv>

Which is:
```
curl https://pyenv.run | bash
```
Giving result of requiring the following to add to .bashrc or bash_profile:
```
WARNING: seems you still have not added 'pyenv' to the load path.

# Load pyenv automatically by appending
# the following to 
# ~/.bash_profile if it exists, otherwise ~/.profile (for login shells)
# and ~/.bashrc (for interactive shells) :

export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# Restart your shell for the changes to take effect.

# Load pyenv-virtualenv automatically by adding
# the following to ~/.bashrc:

eval "$(pyenv virtualenv-init -)"
```
#### Usage

```
pyenv install 3.9.7
pyenv global 3.9.7
```


### Poetry
Basically npm for python. Why we use it idk...

#### Installing Poetry
Seth says to install with official installer for some reason:
<https://python-poetry.org/docs/#installing-with-the-official-installer>


```
curl -sSL https://install.python-poetry.org | python3 -
```

Installs poetry to:
```
$HOME/.local/bin
```
So we need to add it to path:
```
export PATH="$HOME/.local/bin:$PATH"
```


#### Using Poetry
<https://python-poetry.org/docs/basic-usage/>

Its kinda just like a venv wrapper:
```
poetry install

poetry shell # activate function

python hello_world.py

```
Need to check how to use debugger
#### Using Debugger / Interpreter Path with Poetry:
<https://github.com/python-poetry/poetry/issues/106>
```
poetry show -v
```

### Torch / Cuda

Ok, so this thing uses torch / cuda.
Installing this is gonna be a pain...


### Installing torch/cuda on Wsl:
<https://docs.nvidia.com/cuda/wsl-user-guide/index.html>
1. Install update to geforce game ready driver
2. make sure wsl is installed

#### Testing Cuda on Wsl
<https://stackoverflow.com/questions/9727688/how-to-get-the-cuda-version>

For testing nvidia driver cuda version supported:
```
nvidia-smi
```
Also 
```
nvcc --version
```

> [!bug] 
> If stuff bugs out on wsl, just use `wsl.exe --shutdown` until it works. :)




## Installing the Repo
<https://github.com/CMPUT652W24-ClosedAI/CMPUT652FinalProject>
After cloning, cd in and do:
```sh
git clone --recursive https://github.com/CMPUT652W24-ClosedAI/CMPUT652FinalProject.git && \
cd CMPUT652FinalProject
poetry install
# The `poetry install` command above creates a virtual environment for us, in which all the dependencies are installed.
# We can use `poetry shell` to create a new shell in which this environment is activated. Once we are done working with
# MicroRTS, we can leave it again using `exit`.
poetry shell
# By default, the torch wheel is built with CUDA 10.2. If you are using newer NVIDIA GPUs (e.g., 3060 TI), you may need to specifically install CUDA 11.3 wheels by overriding the torch dependency with pip:
# poetry run pip install "torch==1.12.1" --upgrade --extra-index-url https://download.pytorch.org/whl/cu113
python hello_world.py



poetry show -v

```

### Java
What we really need is the java jdk:
<https://www.digitalocean.com/community/tutorials/how-to-install-java-with-apt-on-ubuntu-22-04>

```
sudo apt install default-jdk
javac -version
```

DONT NEED JRE (already comes installed with jdk, run command to test it though). 
RUNTIME ENVIRONMENT: (I think we don't need this one...)
Needs java 8.0+
```
sudo apt install default-jre
java -version
```

200mb...




### Install Memory Size Log
Started with 146g free on c drive
After install:
136g...

~1gig installed




### Debugging
<https://stackoverflow.com/questions/69106483/python-project-with-poetry-how-to-debug-it-in-visual-studio-code>

Ok looks like vscode is nice enough to find the poetry path for us.


### Visualizing:
Methinks we can run a virtual xvfb server - that's what the xvfb-run commands do:


### Running Experiments:

```sh
cd experiments
xvfb-run -a python ppo_gridnet.py \
    --total-timesteps 300000000 \
    --num-bot-envs 24 \
    --num-selfplay-envs 0 \
    --partial-obs False \
    --prod-mode 
```



### Issues
#### Wandb
Wandb np.float_
<https://github.com/wandb/wandb>


Ok, so looks like we are using wandb 0.12.16,
Most recent version is 0.18.5

Right now we are running numpy 2.0.2
but the microrts-py lock says that numpy should be 
1.21.4



## General Notes

Uses java... darn.
jpype is what it uses



Ok, so hello_world runs, looks like it compiles the java jar each time...




