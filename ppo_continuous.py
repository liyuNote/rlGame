"""PPO 连续动作算法实现。

本文件包含三类核心对象：
- Actor_Beta / Actor_Gaussian：连续动作策略网络；
- Critic：状态价值网络 V(s)；
- PPO_continuous：封装动作采样、推理、GAE 计算、PPO clipped 更新和模型保存。
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def load_state_dict_safely(path):
    """兼容不同 PyTorch 版本的 checkpoint 加载。

    新版本 PyTorch 支持 weights_only=True，可以减少反序列化非权重对象的风险；
    老版本没有该参数，所以这里做一次兼容回退。
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def orthogonal_init(layer, gain=1.0):
    """正交初始化线性层。

    PPO 里常用正交初始化提升训练初期稳定性。策略输出层通常使用较小 gain，
    避免初始动作均值或分布参数过大。
    """
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Actor_Beta(nn.Module):
    """Beta 分布策略网络。

    Beta 分布天然输出 [0, 1] 区间的动作。主流程中会把它线性映射到
    [-max_action, max_action]，适合严格有界的连续动作控制。
    """

    def __init__(self, args, agent_idx):
        super().__init__()
        self.agent_name = "agent_%s" % agent_idx
        self.chkpt_file = os.path.join(args.chkpt_dir, self.agent_name + "_actor_Beta")

        # 两层 MLP 提取状态特征。
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)

        # 每个动作维度输出一组 alpha/beta 参数。
        self.alpha_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.beta_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.alpha_layer, gain=0.01)
            orthogonal_init(self.beta_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))

        # softplus 保证参数为正；+1 让初始分布更平滑，不容易贴边。
        alpha = F.softplus(self.alpha_layer(s)) + 1.0
        beta = F.softplus(self.beta_layer(s)) + 1.0
        return alpha, beta

    def get_dist(self, s):
        alpha, beta = self.forward(s)
        return Beta(alpha, beta)

    def mean(self, s):
        alpha, beta = self.forward(s)
        return alpha / (alpha + beta)

    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.chkpt_file), exist_ok=True)
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        if not os.path.exists(self.chkpt_file):
            raise FileNotFoundError(f"Checkpoint not found: {self.chkpt_file}")
        try:
            self.load_state_dict(load_state_dict_safely(self.chkpt_file))
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to load {self.chkpt_file}. The checkpoint network shape does not match "
                "the current arguments. Use the same --hidden-width, --policy-dist, state_dim and "
                "action_dim used during training, or retrain to overwrite this checkpoint."
            ) from exc


class Actor_Gaussian(nn.Module):
    """Gaussian 分布策略网络。

    网络输出动作均值 mean；log_std 是可训练参数，控制探索噪声大小。
    采样动作后会裁剪到环境允许的动作范围。
    """

    def __init__(self, args, agent_idx):
        super().__init__()
        self.agent_name = "agent_%s" % agent_idx
        self.chkpt_file = os.path.join(args.chkpt_dir, self.agent_name + "_actor_Gaussian")
        self.max_action = args.max_action

        # 两层 MLP 提取状态特征。
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)

        # mean_layer 输出每个动作维度的均值，log_std 表示标准差的对数。
        self.mean_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.mean_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))

        # tanh 将均值限制到 [-1, 1]，再乘 max_action 映射到动作范围。
        return self.max_action * torch.tanh(self.mean_layer(s))

    def get_dist(self, s):
        mean = self.forward(s)

        # expand_as 让 [1, action_dim] 的 log_std 匹配当前 batch 的 mean 形状。
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        return Normal(mean, std)

    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.chkpt_file), exist_ok=True)
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        if not os.path.exists(self.chkpt_file):
            raise FileNotFoundError(f"Checkpoint not found: {self.chkpt_file}")
        try:
            self.load_state_dict(load_state_dict_safely(self.chkpt_file))
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to load {self.chkpt_file}. The checkpoint network shape does not match "
                "the current arguments. Use the same --hidden-width, --policy-dist, state_dim and "
                "action_dim used during training, or retrain to overwrite this checkpoint."
            ) from exc


class Critic(nn.Module):
    """状态价值网络 V(s)。

    Critic 估计从当前状态开始的未来折扣回报，用于构造 advantage 和 value target。
    """

    def __init__(self, args, agent_idx):
        super().__init__()
        self.agent_name = "agent_%s" % agent_idx
        self.chkpt_file = os.path.join(args.chkpt_dir, self.agent_name + "_critic")

        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        return self.fc3(s)

    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.chkpt_file), exist_ok=True)
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        if not os.path.exists(self.chkpt_file):
            raise FileNotFoundError(f"Checkpoint not found: {self.chkpt_file}")
        try:
            self.load_state_dict(load_state_dict_safely(self.chkpt_file))
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to load {self.chkpt_file}. The checkpoint network shape does not match "
                "the current arguments. Use the same --hidden-width, --policy-dist, state_dim and "
                "action_dim used during training, or retrain to overwrite this checkpoint."
            ) from exc


class PPO_continuous:
    """连续动作 PPO 智能体。

    训练时：
    1. choose_action() 从旧策略采样动作并记录 log_prob；
    2. 环境交互得到 transition，写入 ReplayBuffer；
    3. buffer 满后 update() 用 clipped objective 更新 actor，用 MSE 更新 critic。

    推理时：
    - evaluate() 使用策略均值动作，不再随机采样。
    """

    def __init__(self, args, agent_idx):
        self.policy_dist = args.policy_dist
        self.max_action = args.max_action
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr_a = args.lr_a
        self.lr_c = args.lr_c
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.epsilon = args.epsilon
        self.K_epochs = args.K_epochs
        self.entropy_coef = args.entropy_coef
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm

        # 根据配置选择 Beta 或 Gaussian 策略。
        if self.policy_dist == "Beta":
            self.actor = Actor_Beta(args, agent_idx)
        else:
            self.actor = Actor_Gaussian(args, agent_idx)
        self.critic = Critic(args, agent_idx)

        # Adam eps=1e-5 是 PPO 常用稳定性 trick。
        if self.set_adam_eps:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

    def evaluate(self, s):
        """推理阶段使用策略均值动作。"""
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float32), 0)
        if self.policy_dist == "Beta":
            a = self.actor.mean(s).detach().numpy().flatten()
        else:
            a = self.actor(s).detach().numpy().flatten()
        return a

    def choose_action(self, s):
        """训练阶段从策略分布中采样动作。

        返回：
        - a：采样动作；
        - a_logprob：动作在当前策略下的 log_prob，后续作为 PPO old_logprob 使用。
        """
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float32), 0)
        with torch.no_grad():
            dist = self.actor.get_dist(s)
            a = dist.sample()
            if self.policy_dist != "Beta":
                a = torch.clamp(a, -self.max_action, self.max_action)
            a_logprob = dist.log_prob(a)
        return a.numpy().flatten(), a_logprob.numpy().flatten()

    def update(self, replay_buffer, total_steps):
        """用一个 batch 的 on-policy 数据执行 PPO 更新。

        terminated=True 表示没有可用于 bootstrap 的后续状态；
        episode_done=True 表示当前采样轨迹结束，用于截断 GAE 递推。
        """
        s, a, a_logprob, r, s_, terminated, episode_done = replay_buffer.numpy_to_tensor()

        # 1. 计算 GAE advantage 和 critic 的监督目标 v_target。
        adv = []
        gae = 0
        with torch.no_grad():
            vs = self.critic(s)
            vs_ = self.critic(s_)

            # TD residual: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)。
            # 仅真正终止时不使用下一状态价值；时间截断仍然 bootstrap。
            deltas = r + self.gamma * (1.0 - terminated) * vs_ - vs

            # 从后往前递推 GAE：
            # A_t = delta_t + gamma * lambda * A_{t+1}。
            # 终止和截断都会结束当前轨迹，不能让 GAE 跨越 reset 继续传播。
            for delta, trajectory_ended in zip(
                reversed(deltas.flatten().numpy()),
                reversed(episode_done.flatten().numpy()),
            ):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - trajectory_ended)
                adv.insert(0, gae)

            adv = torch.tensor(adv, dtype=torch.float32).view(-1, 1)
            v_target = adv + vs

            # Advantage 标准化可以降低方差，使 actor 更新更稳。
            if self.use_adv_norm:
                adv = (adv - adv.mean()) / (adv.std() + 1e-5)

        # 2. 对同一批数据重复优化 K_epochs 次。
        for _ in range(self.K_epochs):
            sampler = BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False)
            for index in sampler:
                dist_now = self.actor.get_dist(s[index])
                dist_entropy = dist_now.entropy().sum(1, keepdim=True)
                a_logprob_now = dist_now.log_prob(a[index])

                # 多维连续动作的 log_prob 需要对动作维度求和。
                # ratio = pi_new(a|s) / pi_old(a|s) = exp(log_new - log_old)。
                ratios = torch.exp(a_logprob_now.sum(1, keepdim=True) - a_logprob[index].sum(1, keepdim=True))

                # PPO clipped surrogate objective：限制新旧策略差异，避免更新过猛。
                surr1 = ratios * adv[index]
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy

                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                # Critic 用均方误差拟合 v_target。
                v_s = self.critic(s[index])
                critic_loss = F.mse_loss(v_target[index], v_s)
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

        if self.use_lr_decay:
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        """按训练进度线性衰减 actor 和 critic 学习率。"""
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p["lr"] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p["lr"] = lr_c_now

    def save_checkpoint(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_checkpoint(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
