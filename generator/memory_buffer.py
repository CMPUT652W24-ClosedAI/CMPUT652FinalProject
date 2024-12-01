from collections import namedtuple, deque
import random

import torch

Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "terminal")
)

device = torch.device("cuda:0") if getattr(torch, "cuda", None) and torch.cuda.is_available() else torch.device(
    "cpu")

class MemoryBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition, ensuring tensors are on the correct device"""
        device_args = [arg.to(device) if isinstance(arg, torch.Tensor) else arg for arg in args]
        self.memory.append(Transition(*device_args))


    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)