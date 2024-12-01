import torch
from torch.cuda import device


class SampleMeanStd:
    def __init__(self, device):
        self.mean = torch.tensor(0.0, device=device)
        self.var = torch.tensor(1.0, device=device)
        self.p = torch.tensor(1.0, device=device)
        self.count = torch.tensor(0, device=device)
        self.device = device

    def update(self, x):
        if self.count == 0:
            self.mean = torch.clone(x)
            self.p = torch.tensor(0.0, device=self.device)
        self.mean, self.var, self.p, self.count = self.update_mean_var_count_from_moments(self.mean, self.p, self.count, x)

    def update_mean_var_count_from_moments(self, mean, p, count, sample):
        new_count = count + 1
        new_mean = mean + (sample - mean) / new_count
        p = p + (sample - mean) * (sample - new_mean)
        new_var = 1 if new_count < 2 else p / (new_count - 1)
        return new_mean, new_var, p, new_count

class ScaleReward:
    def __init__(self, device, gamma: float = 0.99, epsilon: float = 1e-8):
        self.reward_stats = SampleMeanStd(device)
        self.reward_trace = torch.tensor(0.0, device=device)
        self.gamma = torch.tensor(gamma, device=device)
        self.epsilon = torch.tensor(epsilon, device=device)

    def scale_reward(self, reward, term):
        self.reward_trace = self.reward_trace * self.gamma * (1 - term) + reward
        return self.normalize(reward)

    def normalize(self, reward):
        self.reward_stats.update(self.reward_trace)
        return reward / torch.sqrt(self.reward_stats.var + self.epsilon)
