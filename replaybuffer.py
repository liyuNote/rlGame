import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, args):
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.batch_size = args.batch_size
        self.s = np.zeros((args.batch_size, args.state_dim), dtype=np.float32)
        self.a = np.zeros((args.batch_size, args.action_dim), dtype=np.float32)
        self.a_logprob = np.zeros((args.batch_size, args.action_dim), dtype=np.float32)
        self.r = np.zeros((args.batch_size, 1), dtype=np.float32)
        self.s_ = np.zeros((args.batch_size, args.state_dim), dtype=np.float32)
        self.terminated = np.zeros((args.batch_size, 1), dtype=np.float32)
        self.episode_done = np.zeros((args.batch_size, 1), dtype=np.float32)
        self.count = 0

    def store(self, s, a, a_logprob, r, s_, terminated, episode_done):
        index = self.count % self.batch_size
        self.s[index] = s
        self.a[index] = a
        self.a_logprob[index] = a_logprob
        self.r[index] = r
        self.s_[index] = s_
        self.terminated[index] = terminated
        self.episode_done[index] = episode_done
        self.count += 1

    def numpy_to_tensor(self):
        return (
            torch.tensor(self.s, dtype=torch.float32),
            torch.tensor(self.a, dtype=torch.float32),
            torch.tensor(self.a_logprob, dtype=torch.float32),
            torch.tensor(self.r, dtype=torch.float32),
            torch.tensor(self.s_, dtype=torch.float32),
            torch.tensor(self.terminated, dtype=torch.float32),
            torch.tensor(self.episode_done, dtype=torch.float32),
        )
