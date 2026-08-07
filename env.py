import numpy as np
from gymnasium import spaces

import satellite_function as sf


class satellites:
    """卫星追逃强化学习环境。

    环境中有两个航天器：
    - Pursuer：追踪方，也是当前训练的 PPO 智能体；
    - Escaper：逃逸方，当前由另一个未训练策略产生动作。

    状态为 18 维连续向量，动作为 3 维连续控制量。每个 step 会先根据动作更新速度，
    再通过 Clohessy-Wiltshire 相对运动模型推进到下一时刻。
    """

    def __init__(
        self,
        Pursuer_position=np.array([200000.0, 0.0, 0.0]),
        Pursuer_vector=np.array([0.0, 0.0, 0.0]),
        Escaper_position=np.array([18000.0, 0.0, 0.0]),
        Escaper_vector=np.array([0.0, 0.0, 0.0]),
        M=0.4,
        dis_safe=1000.0,
        d_capture=100000.0,
        Flag=0,
        fuel_c=320.0,
        fuel_t=320.0,
        args=None,
    ):
        # 保存初始状态，reset() 时会恢复到这些值。统一转为 float64，减少动力学计算误差。
        self.initial_pursuer_position = np.asarray(Pursuer_position, dtype=np.float64)
        self.initial_pursuer_vector = np.asarray(Pursuer_vector, dtype=np.float64)
        self.initial_escaper_position = np.asarray(Escaper_position, dtype=np.float64)
        self.initial_escaper_vector = np.asarray(Escaper_vector, dtype=np.float64)

        # 环境基础参数。
        # M 是质量参数占位；当前简化模型未直接用质量计算推力。
        self.M = M
        self.dis_safe = dis_safe

        # 捕获距离：追踪方和逃逸方距离小于该阈值时视为捕获成功。
        self.d_capture = d_capture

        # Flag 用于区分模式：0 训练追踪方，1 可扩展为训练逃逸方，2 测试。
        self.Flag = Flag

        # 简化燃料模型：动作绝对值越大，每步消耗燃料越多。
        self.initial_fuel_c = fuel_c
        self.initial_fuel_t = fuel_t
        self.fuel_c = fuel_c
        self.fuel_t = fuel_t

        # 终局奖励。捕获成功给 win_reward；超时/燃料耗尽给 burn_reward。
        self.burn_reward = 0.0
        self.win_reward = 100.0
        self.evader_capture_penalty = -100.0

        # dangerous_zone 表示是否处在“正在接近且进入预警范围”的区域。
        self.dangerous_zone = 0

        # 单回合最大步数，以及每步动力学推进时间。
        self.max_episode_steps = getattr(args, "max_episode_steps", 1000)
        self.dt = getattr(args, "env_dt", 100.0)

        # observation_space 的上下界按状态向量排列：
        # [相对位置3, 相对速度3, 追踪方位置3, 追踪方速度3, 逃逸方位置3, 逃逸方速度3]。
        position_low = np.array(
            [-500000.0, -500000.0, -500000.0, -1e7, -1e7, -1e7, -1e7, -1e7, -1e7]
        )
        position_high = np.array(
            [500000.0, 500000.0, 500000.0, 1e7, 1e7, 1e7, 1e7, 1e7, 1e7]
        )
        velocity_low = np.array(
            [-10000.0, -10000.0, -10000.0, -50000.0, -50000.0, -50000.0, -50000.0, -50000.0, -50000.0]
        )
        velocity_high = np.array(
            [10000.0, 10000.0, 10000.0, 50000.0, 50000.0, 50000.0, 50000.0, 50000.0, 50000.0]
        )
        self.observation_space = spaces.Box(
            low=np.concatenate((position_low, velocity_low)).astype(np.float32),
            high=np.concatenate((position_high, velocity_high)).astype(np.float32),
            shape=(18,),
            dtype=np.float32,
        )

        # 动作空间为 3 个轴向控制量，每个分量限制在 [-1.6, 1.6]。
        self.action_space = np.array([[-1.6, 1.6], [-1.6, 1.6], [-1.6, 1.6]], dtype=np.float32)

        # 预留离散动作空间，当前主流程不用它。
        self.action_space_beta = spaces.Discrete(5)

        self.reset(Flag)

    def reset(self, Flag=0):
        """重置一个 episode，并返回初始状态。"""
        self.Pursuer_position = self.initial_pursuer_position.copy()
        self.Pursuer_vector = self.initial_pursuer_vector.copy()
        self.Escaper_position = self.initial_escaper_position.copy()
        self.Escaper_vector = self.initial_escaper_vector.copy()
        self.fuel_c = self.initial_fuel_c
        self.fuel_t = self.initial_fuel_t
        self.pursuer_reward = 0.0
        self.escaper_reward = 0.0
        self.last_evader_reward = 0.0
        self.dangerous_zone = 0
        self.Flag = Flag
        self.dis = np.linalg.norm(self.Pursuer_position - self.Escaper_position)
        return self._get_state()

    def step(self, pursuer_action, escaper_action, epsiode_count):
        """环境推进一步。

        参数：
        - pursuer_action：追踪方 3 维动作；
        - escaper_action：逃逸方 3 维动作；
        - epsiode_count：当前 episode 内第几步。

        返回：
        - next_state：18 维下一状态；
        - reward：追踪方奖励；
        - terminated：是否因捕获或燃料耗尽而真正终止；
        - truncated：是否因达到最大步数而截断。
        """
        # 动作裁剪保证策略输出不会超过环境允许的控制能力。
        pursuer_action = np.clip(np.asarray(pursuer_action, dtype=np.float64), -1.6, 1.6)
        escaper_action = np.clip(np.asarray(escaper_action, dtype=np.float64), -1.6, 1.6)
        pursuer_action = self._fuel_limited_action(pursuer_action, self.fuel_c)
        escaper_action = self._fuel_limited_action(escaper_action, self.fuel_t)

        # 记录动作执行前距离，用于判断这一步是否更接近目标。
        old_distance = np.linalg.norm(self.Pursuer_position - self.Escaper_position)

        # 简化燃料消耗：三个轴向控制量绝对值之和。
        self.fuel_c -= float(np.abs(pursuer_action).sum())
        self.fuel_t -= float(np.abs(escaper_action).sum())

        # 简化控制：动作直接作为速度增量。
        self.Pursuer_vector += pursuer_action
        self.Escaper_vector += escaper_action

        # 用 CW 状态转移矩阵推进双方位置和速度。
        s_pursuer, s_escaper = sf.Clohessy_Wiltshire(
            R0_c=self.Pursuer_position,
            V0_c=self.Pursuer_vector,
            R0_t=self.Escaper_position,
            V0_t=self.Escaper_vector,
        ).State_transition_matrix(self.dt)

        self.Pursuer_position, self.Pursuer_vector = s_pursuer[:3], s_pursuer[3:]
        self.Escaper_position, self.Escaper_vector = s_escaper[:3], s_escaper[3:]
        self.dis = np.linalg.norm(self.Pursuer_position - self.Escaper_position)
        self.calculate_number_danger_area()
        evader_reward = self.evader_reward(old_distance, escaper_action)

        # 捕获成功：直接返回高奖励并结束回合。
        if self.dis <= self.d_capture:
            self.last_evader_reward = self.evader_capture_penalty
            return self._get_state(), self.win_reward, True, False

        # 追踪方燃料耗尽属于任务真正终止。
        if self.fuel_c <= 0:
            self.last_evader_reward = evader_reward
            return self._get_state(), self.burn_reward, True, False

        # The time limit truncates the sampled trajectory, but the physical
        # state remains valid and can still be used for value bootstrapping.
        if epsiode_count >= self.max_episode_steps:
            self.last_evader_reward = evader_reward
            return self._get_state(), self.burn_reward, False, True

        # 奖励项 1：连续距离变化奖励。比上一时刻更接近则为正，远离则为负。
        distance_delta = old_distance - self.dis
        reward = 1.0 if distance_delta > 0.0 else -1.0
        reward += 4.0 * float(np.clip(distance_delta / max(self.d_capture, 1e-8), -2.0, 2.0))

        # 奖励项 2：鼓励进入目标附近的捕获外壳区域。
        reward += -1.0 if self.d_capture <= self.dis <= 4.0 * self.d_capture else -2.0
        far_penalty = float(np.clip((self.dis - 4.0 * self.d_capture) / max(self.d_capture, 1e-8), 0.0, 10.0))
        reward -= 0.3 * far_penalty

        # 奖励项 3：鼓励进入“接近且较近”的危险区/有效追踪区。
        reward += -1.0 if self.dangerous_zone == 0 else self.dangerous_zone * 0.5

        # 奖励项 4-7：更细的启发式 shaping，帮助 PPO 在稀疏捕获奖励之外获得学习信号。
        reward += self.distance_reward(self.Pursuer_position)
        reward += 0.6 * self.velocity_penalty(self.Pursuer_vector)
        reward += 0.2 * self.direction_reward(self.Pursuer_position, self.Pursuer_vector)
        reward += 2.0 * self.fuel_conservation_reward()
        self.pursuer_reward = float(reward)
        self.last_evader_reward = evader_reward
        return self._get_state(), self.pursuer_reward, False, False

    @staticmethod
    def _fuel_limited_action(action, fuel_remaining):
        """Scale action to the remaining fuel budget; return zero if fuel is gone."""
        requested_fuel = float(np.abs(action).sum())
        if fuel_remaining <= 0.0 or requested_fuel <= 1e-12:
            return np.zeros_like(action, dtype=np.float64)
        if requested_fuel <= fuel_remaining:
            return action
        return action * (fuel_remaining / requested_fuel)

    def _get_state(self):
        """组装 18 维状态向量。

        状态排列：
        0:3   相对位置 r_p - r_e
        3:6   相对速度 v_p - v_e
        6:9   追踪方位置
        9:12  追踪方速度
        12:15 逃逸方位置
        15:18 逃逸方速度
        """
        return np.array(
            [
                self.Pursuer_position - self.Escaper_position,
                self.Pursuer_vector - self.Escaper_vector,
                self.Pursuer_position,
                self.Pursuer_vector,
                self.Escaper_position,
                self.Escaper_vector,
            ],
            dtype=np.float32,
        ).ravel()

    def calculate_number_danger_area(self):
        """判断是否处于有效接近区。

        如果相对位置和相对速度点积小于 0，说明两者距离有缩小趋势；
        同时距离小于 4 倍捕获半径时，认为追踪方已经进入有效接近区。
        """
        rel_pos = self.Pursuer_position - self.Escaper_position
        rel_vel = self.Pursuer_vector - self.Escaper_vector
        closing = np.dot(rel_pos, rel_vel) < 0
        in_warning_shell = self.dis <= 4.0 * self.d_capture
        self.dangerous_zone = int(closing and in_warning_shell)
        return self.dangerous_zone

    def velocity_penalty(self, velocity):
        """速度惩罚项。

        速度越大，惩罚越接近 -1。这样可以抑制策略通过无限增大速度来刷接近奖励。
        """
        speed = np.linalg.norm(velocity)
        return -min(speed / 10000.0, 1.0)

    def direction_reward(self, position, velocity):
        """朝向奖励项。

        计算追踪方相对逃逸方的速度在“指向逃逸方方向”上的投影。
        投影越大，说明速度方向越有利于接近目标。
        """
        rel_pos = self.Escaper_position - position
        distance = np.linalg.norm(rel_pos) + 1e-8
        direction = rel_pos / distance
        closing_speed = np.dot(velocity - self.Escaper_vector, direction)
        return float(np.tanh(closing_speed / 1000.0))

    def distance_reward(self, position):
        """位置接近奖励项。

        距离进入 4 倍捕获半径内会逐渐得到更高奖励；
        距离太远则为负值，范围裁剪到 [-1, 1]。
        """
        distance = np.linalg.norm(position - self.Escaper_position)
        return float(np.clip((4.0 * self.d_capture - distance) / (4.0 * self.d_capture), -1.0, 1.0))

    def fuel_conservation_reward(self):
        """燃料保留奖励项。

        燃料比例越高奖励越高，鼓励策略不要无意义地大幅机动。
        """
        fuel_ratio = max(self.fuel_c, 0.0) / max(self.initial_fuel_c, 1e-8)
        return float(0.2 * fuel_ratio)

    def evader_reward(self, old_distance, escaper_action):
        """Reward for evader: separate from pursuer, save fuel and avoid capture."""
        distance_delta = self.dis - old_distance
        separation_reward = np.clip(distance_delta / max(self.d_capture, 1e-8), -1.0, 1.0)
        safe_distance_reward = np.clip((self.dis - self.d_capture) / max(3.0 * self.d_capture, 1e-8), -1.0, 1.0)
        capture_zone_penalty = -2.0 if self.dis <= self.d_capture else 0.0
        warning_zone_penalty = -0.5 if self.dis <= 4.0 * self.d_capture else 0.0
        fuel_penalty = 0.05 * float(np.abs(escaper_action).sum())
        fuel_bonus = 0.2 * max(self.fuel_t, 0.0) / max(self.initial_fuel_t, 1e-8)
        self.escaper_reward = float(
            2.0 * separation_reward
            + safe_distance_reward
            + capture_zone_penalty
            + warning_zone_penalty
            + fuel_bonus
            - fuel_penalty
        )
        return self.escaper_reward
