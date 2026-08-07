# 强化学习用于卫星/航天器追逃博弈的论文与材料综述

> 更新日期：2026-06-26
> 资料来源：公开网络检索 + 本地目录 `D:\空天信息大学\文献调研`
> 整理原则：不再区分“本地/线上”，统一按研究主题和对本项目的价值组织。

本文面向当前 PPO 卫星追逃项目，重点收集以下类型材料：

- 直接使用强化学习/深度强化学习解决卫星、航天器、轨道追逃博弈问题；
- 与追逃高度相关的主动防御、规避、拦截、多智能体轨道博弈；
- 对当前代码有直接参考价值的近邻方向，如安全 RL、航天器交会、检查、碰撞避免；
- 可作为 baseline 的传统微分博弈、最优控制、轨迹优化材料。

## 1. 直接相关论文总表

| 编号 | 论文/材料                                                                                                                  | 年份          | 场景                           | 方法                         | 相关度   |
| ---- | -------------------------------------------------------------------------------------------------------------------------- | ------------- | ------------------------------ | ---------------------------- | -------- |
| P1   | Two-Stage Pursuit Strategy for Incomplete-Information Impulsive Space Pursuit-Evasion Mission Using Reinforcement Learning | 2021          | 信息非完备脉冲空间追逃         | 远距离轨迹优化 + 近距离 DDPG | 直接相关 |
| P2   | Near-optimal interception strategy for orbital pursuit-evasion using deep reinforcement learning                           | 2022          | 一对一轨道追逃拦截             | 微分博弈 barrier + DNN/DRL   | 直接相关 |
| P3   | 基于多智能体强化学习的轨道追逃博弈方法                                                                                     | 2022          | 多星集群追逃非合作目标         | MADDPG                       | 直接相关 |
| P4   | 航天器轨道追逃博弈多阶段强化学习训练方法                                                                                   | 2022          | 双航天器追逃训练赋能           | 多阶段 RL + PPO              | 直接相关 |
| P5   | Cooperative Guidance Strategy for Active Defense Spacecraft with Imperfect Information via Deep Reinforcement Learning     | 2022          | 目标-防御器-拦截器主动防御     | Deep RL                      | 直接相关 |
| P6   | 基于终端诱导强化学习的航天器轨道追逃博弈                                                                                   | 2023          | 多脉冲航天器轨道追逃           | PPO 交替训练                 | 直接相关 |
| P7   | 信息非完备下多航天器轨道博弈强化学习方法                                                                                   | 2023          | 信息非完备多航天器轨道博弈     | PPO + LSTM + self-play       | 直接相关 |
| P8   | An efficient learning-based algorithm for orbital pursuit-evasion game with impulsive maneuvers                            | 2023          | 脉冲轨道追逃博弈               | PRD-MADDPG                   | 直接相关 |
| P9   | Reinforcement learning-based decision-making for spacecraft pursuit-evasion game in elliptical orbits                      | 2024          | 椭圆轨道航天器追逃             | DDPG + TH 方程               | 直接相关 |
| P10  | Impulsive maneuver strategy for multi-agent orbital pursuit-evasion game under sparse rewards                              | 2024          | 多智能体脉冲轨道追逃           | 分层 MARL + HER + 稀疏奖励   | 直接相关 |
| P11  | Satellite Chasers: Divergent Adversarial Reinforcement Learning to Engage Intelligent Adversaries on Orbit                 | 2024/2026     | 多追踪星追逐逃逸星             | DARL / 多智能体对抗 RL       | 直接相关 |
| P12  | Orbital Multi-Player Pursuit-Evasion Game with Deep Reinforcement Learning                                                 | 2025          | 多追踪方合围捕获逃逸方         | 分布式 distributional DDPG   | 直接相关 |
| P13  | I Can Hear You Coming: RF Sensing for Uncooperative Satellite Evasion                                                      | 2025          | 非合作卫星接近下的 RF 感知规避 | Constrained RL + RF 感知     | 直接相关 |
| P14  | 基于多智能体强化学习的追踪-防御轨道博弈决策方法研究                                                                        | 2026 网络首发 | 追踪-防御轨道博弈              | SA-MAPPO + masking           | 直接相关 |

## 2. 逐篇创新点与对本项目启发

### P1. Two-Stage Pursuit Strategy for Incomplete-Information Impulsive Space Pursuit-Evasion Mission Using Reinforcement Learning

**核心问题**
信息非完备、J2 摄动、脉冲推进条件下的空间追逃任务。

**主要方法**
将任务分为远距离交会阶段和近距离博弈阶段。远距离阶段做轨迹优化，近距离阶段用 DDPG 闭环追踪。

**创新点**

1. 按逃逸方感知范围将追逃任务拆分为远距离交会和近距离博弈两个阶段。
2. 远距离阶段构造新的终端追踪能力目标函数，使转入近距离博弈时追踪方更有优势。
3. 近距离阶段使用 DDPG 在线更新追踪策略，适应信息非完备和逃逸方机动。
4. 考虑 J2 摄动，较单纯 CW 模型更接近真实轨道环境。
5. Monte Carlo 仿真中多场景追踪成功率超过 91%。

**对本项目的启发**

- 当前代码可以加入“两阶段模式”：先远距离接近，再近距离追逃；
- 可以在 `env.py` 中加入 J2 摄动或更高保真轨道模型；
- 可以将 `d_capture` 外的接近过程单独设计奖励，提高早期训练效率。

### P2. Near-optimal interception strategy for orbital pursuit-evasion using deep reinforcement learning

**核心问题**
一对一轨道追逃中，追踪方需要把状态推进到捕获区，实现近似最优拦截。

**主要方法**
结合微分博弈 barrier solution、深度学习和 DRL。

**创新点**

1. 先用微分博弈推导闭式 barrier 解，保留传统理论可解释性。
2. 用 DNN 学习轨道追逃状态与 barrier 轨迹之间的隐含映射，实现快速在线求解。
3. 对 barrier 外的状态，使用 DRL 学习 capture-zone embedding strategy，帮助状态跨越 barrier 面。
4. 将解析解、神经网络近似和强化学习策略结合，而不是纯黑箱学习。
5. 面向实时拦截，避免大量离线轨迹库带来的存储和查询负担。

**对本项目的启发**

- 可以把传统 barrier/捕获域理论作为 reward 或 expert prior；
- 可以先用传统方法生成样本，再预训练 PPO actor；
- 可以在动画里显示“是否跨越捕获 barrier”作为博弈态势指标。

### P3. 基于多智能体强化学习的轨道追逃博弈方法

**核心问题**
多个追踪卫星协同追捕具有未知机动信息的非合作目标。

**主要方法**
MADDPG，集中训练、分布式执行。

**创新点**

1. 面向多星集群和非合作目标的轨道追逃，而不是单追踪星。
2. 奖励函数同时考虑最短时间、燃料最优和碰撞避免。
3. 采用 MADDPG 进行集中训练，使各追踪卫星能学习协同行为。
4. 执行阶段分布式决策，更符合星上自治需求。
5. 仿真中涌现出“拦截、围攻、渗透、捕获”等智能博弈行为。

**对本项目的启发**

- 可将当前 `pursuer` 扩展为多个追踪星；
- 需要为每个智能体建立独立观测、动作和奖励；
- 可以引入集中训练分布式执行框架，替换当前单智能体 PPO。

### P4. 航天器轨道追逃博弈多阶段强化学习训练方法

**核心问题**
直接自博弈训练效率低，且面对不确定对手时泛化能力不足。

**主要方法**
规则策略预训练 + 多阶段 PPO 互搏重训练。

**创新点**

1. 构造基于终端位置预测的规则追踪策略和规则逃逸策略。
2. 先让神经网络与规则对手对抗预训练，降低随机探索难度。
3. 再让追踪方和逃逸方 PPO 网络互搏训练，逐步提升双方能力。
4. 将“规则赋能”和“强化学习自博弈”结合，提高训练效率。
5. 使最终策略对不确定对手有更好的适应性。

**对本项目的启发**

- 当前 evader 未训练，可以先写一个规则逃逸策略作为预训练对手；
- 再加入 pursuer/evader 双 PPO self-play；
- 可在 `main.py` 中加入 `--opponent rule|learned|selfplay` 参数。

### P5. Cooperative Guidance Strategy for Active Defense Spacecraft with Imperfect Information via Deep Reinforcement Learning

**核心问题**
目标航天器面对拦截器威胁，同时释放主动防御飞行器协同规避。

**主要方法**
深度强化学习，多智能体主动防御指导律。

**创新点**

1. 将主动防御航天器制导问题从传统最优制导/微分博弈转为 DRL 求解。
2. 考虑拦截器机动能力不完全信息，更接近真实非合作对抗。
3. 使用 reward shaping 缓解稀疏奖励。
4. 采用逐渐增难训练，提高复杂场景下收敛稳定性。
5. 将目标规避和防御器协同统一到学习策略中。

**对本项目的启发**

- 可加入第三方“防御星”或“拦截星”，扩展成追踪-防御-逃逸三方博弈；
- 可借鉴课程学习，从简单捕获半径、大燃料裕度开始训练；
- 可将不完全信息作为默认设置，而不是默认全状态可观测。

### P6. 基于终端诱导强化学习的航天器轨道追逃博弈

**核心问题**
追踪方需要在指定终端时刻进入逃逸方邻域，而不是只追求当前距离变小。

**主要方法**
PPO，追踪方和逃逸方交替训练，终端诱导奖励。

**创新点**

1. 考虑燃料、推力、决策周期、运动范围等工程约束。
2. 建立锥形安全接近区和 CW 相对运动模型。
3. 采用 PPO 框架交替训练追踪星和逃逸星。
4. 提出终端诱导奖励函数，用 CW 方程预测终端相对误差。
5. 相比只基于当前误差的奖励，能提高指定时刻追击成功率。

**对本项目的启发**

- 当前奖励主要依赖即时距离变化，可以加入“终端预测误差”；
- 可在 `env.step()` 中计算若干步后的预测相对位置，作为 shaping 项；
- 对固定时间捕获任务比单纯最短距离奖励更合适。

### P7. 信息非完备下多航天器轨道博弈强化学习方法

**核心问题**
多航天器轨道博弈中，智能体无法获得完整位置/速度信息。

**主要方法**
多智能体 PPO + LSTM + self-play。

**创新点**

1. 明确信息非完备约束下单步观测不足的问题。
2. 在策略网络中引入 LSTM，用历史观测补偿位置、速度信息缺失。
3. 根据分布式系统架构设计网络输入输出，适配多航天器场景。
4. 采用红蓝左右互搏式 PPO 自博弈训练。
5. 提高训练稳定性、任务完成率，并降低燃料消耗。

**对本项目的启发**

- 可以给 PPO actor 加 LSTM/GRU，处理部分可观测状态；
- 可把完整状态改为“相对距离 + 方位 + 噪声速度估计”；
- 可加入历史帧堆叠，作为不改网络结构的轻量替代方案。

### P8. An efficient learning-based algorithm for orbital pursuit-evasion game with impulsive maneuvers

**核心问题**
双方均采用脉冲速度增量机动的轨道追逃博弈。

**主要方法**
PRD-MADDPG：Predict-Reward-Detect + MADDPG。

**创新点**

1. 将脉冲轨道追逃建模为带终端时间、机动能力、燃料和任务时间约束的 min-max 优化问题。
2. 基于 MADDPG 训练双方策略，适配多智能体对抗。
3. 提出 PRD 机制，预测相邻脉冲间隔内的博弈状态变化。
4. 将预测信息转化为 predicted reward 注入训练。
5. 在复杂约束下提升学习效率，并展示未见场景泛化能力。

**对本项目的启发**

- 当前动作每步连续施加，可改为“脉冲机动事件”；
- 可在 replay buffer 中加入预测奖励或模型辅助 rollout；
- 对脉冲推力航天器更贴近实际任务。

### P9. Reinforcement learning-based decision-making for spacecraft pursuit-evasion game in elliptical orbits

**核心问题**
椭圆轨道中的三维脉冲航天器追逃决策。

**主要方法**
DDPG + Tschauner-Hempel 椭圆轨道相对运动方程。

**创新点**

1. 将追逃博弈从常见圆轨道/CW 模型扩展到椭圆轨道。
2. 使用线性化 TH 方程描述三维椭圆轨道相对运动。
3. 建立完整三维脉冲机动模型。
4. 基于 DDPG 学习连续动作追逃策略。
5. 奖励函数综合最短时间、燃料最优和碰撞避免。

**对本项目的启发**

- 可把 `satellite_function.py` 从 CW 扩展到 TH；
- 可加入椭圆轨道参数作为环境配置；
- 可把燃料和碰撞约束提升为主奖励项，而不是辅助项。

### P10. Impulsive maneuver strategy for multi-agent orbital pursuit-evasion game under sparse rewards

**核心问题**
多智能体脉冲轨道追逃中，密集奖励设计主观性强，稀疏奖励又难训练。

**主要方法**
分层网络 + HER + 可达域约束 + 集中训练分层执行。

**创新点**

1. 用稀疏奖励减少人工 dense reward shaping 的主观性。
2. 引入基于 HER 的分层网络，提高稀疏奖励下探索效率。
3. 使用轨道可达域方法细化子目标空间，使子目标物理可达。
4. 采用集中训练、分层执行框架。
5. 在多智能体追逃中形成从长期子目标到短期子目标的层级行为。

**对本项目的启发**

- 可将当前奖励改成更稀疏的“捕获/未捕获”，再用 HER 辅助训练；
- 可加入高层网络输出子目标，低层网络输出控制动作；
- 对多阶段追逃和复杂任务目标非常有价值。

### P11. Satellite Chasers: Divergent Adversarial Reinforcement Learning to Engage Intelligent Adversaries on Orbit

**核心问题**
两个追踪星追逐一个逃逸星的近距离对抗任务，类似空间 capture-the-flag。

**主要方法**
DARL，多智能体对抗强化学习，强调对手多样性。

**创新点**

1. 提出 Divergent Adversarial Reinforcement Learning。
2. 先训练基础逃逸策略，再训练多个差异化追踪对手，再用对手池提升逃逸策略泛化。
3. 引入 divergent loss，避免多个对手收敛为相似策略。
4. 使用 CW 动力学、3DOF 连续控制和部分可观测设置。
5. 关注智能对手下的泛化能力，而不仅是固定规则对手。

**对本项目的启发**

- 可建立 pursuer/evader 策略池，训练时随机采样对手；
- 可加入策略多样性损失，避免自博弈坍缩；
- 可把当前单一 evader 改成多种逃逸风格。

### P12. Orbital Multi-Player Pursuit-Evasion Game with Deep Reinforcement Learning

**核心问题**
多个追踪方以合围构型捕获一个逃逸方，逃逸方试图突破包围。

**主要方法**
分布式 distributional DDPG，平行对抗学习。

**创新点**

1. 设计椭圆合围构型，利用多追踪星初始位置优势。
2. 将合围捕获建模为离散 Markov game。
3. 将 distributional DDPG 改造为平行对抗学习框架。
4. 修改策略网络和 policy-gradient 计算，实现多追踪方去中心化协同决策。
5. 训练后涌现主动协同追踪和多目标逃逸策略。

**对本项目的启发**

- 当前动画和环境可以扩展为多追踪星合围；
- 可加入 encirclement/capture 队形奖励；
- 可让每个 pursuer 只看到局部观测，执行分布式决策。

### P13. I Can Hear You Coming: RF Sensing for Uncooperative Satellite Evasion

**核心问题**
逃逸卫星在缺少高精度态势感知设备时，利用 RF 信号估计非合作卫星接近并规避。

**主要方法**
RF 感知 + 受约束 RL。

**创新点**

1. 将 RF 截获信号作为追逃规避观测输入。
2. 结合动力学状态和 RF 观测，形成多模态状态。
3. 建立 Cat & Mouse 仿真系统，考虑定位误差和控制约束。
4. 将 RL 策略与传统避碰优化方法进行对比。
5. 强调星上边缘自治和不依赖地面支援的实时规避。

**对本项目的启发**

- 可加入带噪声的传感器观测，而不是直接给真实相对状态；
- 可扩展动画展示“真实位置”和“估计位置”；
- 可尝试 constrained PPO，保证燃料和安全距离约束。

### P14. 基于多智能体强化学习的追踪-防御轨道博弈决策方法研究

**核心问题**
脉冲推力航天器集群中，追踪方要避开防御方拦截并接近目标。

**主要方法**
SA-MAPPO：Self-Attention + MAPPO + masking。

**创新点**

1. 建立追踪-防御轨道博弈模型，考虑终端条件、单次速度增量和燃料限制。
2. 在 MAPPO 中引入 self-attention，提高个体间空间关系感知。
3. 引入 masking 机制处理个体航天器 drop-out 场景。
4. 双方通过互搏训练同步提升。
5. 相比 MAPPO，收敛更快、奖励更稳定，捕获和拦截成功率均提升。

**对本项目的启发**

- 若扩展到多星，self-attention 是很自然的状态融合方式；
- masking 可用于处理卫星失效、燃料耗尽或被捕获后的队伍变化；
- MAPPO 比单智能体 PPO 更适合多星协同。

## 3. 近邻支撑材料

这些论文不是严格追逃博弈，但对当前项目的环境、奖励、安全约束和工程落地有参考价值。

| 编号 | 论文/材料                                                                                                                                   | 年份 | 场景               | 方法                       | 可借鉴点                                                     |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------- | ---- | ------------------ | -------------------------- | ------------------------------------------------------------ |
| N1   | Autonomous Six-Degree-of-Freedom Spacecraft Docking Maneuvers via Reinforcement Learning                                                    | 2020 | 6DOF 航天器对接    | PPO                        | PPO 连续控制、动作约束、控制成本奖励、航天器对接任务建模     |
| N2   | Deep Reinforcement Learning for Autonomous Spacecraft Inspection using Illumination                                                         | 2023 | 航天器自主检查     | DRL                        | 将光照/成像质量纳入奖励，适合扩展成“追踪成像/侦察质量”任务 |
| N3   | Spacecraft Autonomous Decision-Planning for Collision Avoidance: a Reinforcement Learning Approach                                          | 2023 | 碰撞规避           | POMDP + RL                 | 部分可观测、不确定目标状态、规避机动                         |
| N4   | Deep Reinforcement Learning for Scalable Multiagent Spacecraft Inspection                                                                   | 2024 | 多智能体航天器检查 | MARL                       | 可扩展观测表示，适合智能体数量变化                           |
| N5   | Neural-based Control for CubeSat Docking Maneuvers                                                                                          | 2024 | CubeSat 对接       | PPO/DRL                    | 课程学习、噪声注入、Monte Carlo、硬件推理测试                |
| N6   | Coupled Rendezvous and Docking Maneuver control of satellite using Reinforcement learning-based Adaptive Fixed-Time Sliding Mode Controller | 2025 | 交会对接控制       | RL + 滑模控制              | RL 与控制理论结合，提供稳定性保证                            |
| N7   | Safe Reinforcement Learning: Optimal Formation Control With Collision Avoidance of Multiple Satellite Systems                               | 2025 | 多卫星编队与避碰   | Safe RL + barrier function | 用 barrier function 保证安全，可用于多追踪星避碰             |

## 4. 传统非 RL baseline 材料

这些材料不属于强化学习，但适合作为论文对比或理论基础。

| 编号 | 材料                                                                                                     | 类型                   | 可参考点                                                                    |
| ---- | -------------------------------------------------------------------------------------------------------- | ---------------------- | --------------------------------------------------------------------------- |
| B1   | Interception Domain Approach to Orbital Multi-Player “Encirclement-Capture” Games                      | 多玩家合围捕获微分博弈 | interception domain、Nash equilibrium 分区结构，可作为多追踪方理论 baseline |
| B2   | Optimal Time-to-Entry Pursuit-Evasion Games Under Sun-Angle Constraints with Non-Smooth Terminal Regions | 太阳角约束追逃微分博弈 | 将太阳角/成像约束引入终端区域，适合扩展观测/成像任务                        |
| B3   | Impulsive guidance of optimal pursuit with conical imaging zone for the evader                           | 锥形成像区约束脉冲制导 | 成像捕获型任务、锥形安全接近区、脉冲制导 baseline                           |
| B4   | 基于行为预测和策略融合的轨道博弈决策方法                                                                 | 行为预测 + 策略融合    | 可为 RL 策略加入对手行为预测和策略融合模块                                  |

## 5. 推荐阅读顺序

### 第一阶段：直接服务当前 PPO 单追踪星代码

1. **P6 基于终端诱导强化学习的航天器轨道追逃博弈**与当前代码最接近：PPO、CW、追踪/逃逸交替训练、奖励设计。
2. **P4 航天器轨道追逃博弈多阶段强化学习训练方法**重点看规则策略预训练和多阶段训练，可直接解决当前 evader 未训练的问题。
3. **P7 信息非完备下多航天器轨道博弈强化学习方法**重点看 PPO + LSTM，可用于把当前全状态观测改成部分可观测。
4. **N1 6DOF Docking via PPO**
   重点看 PPO 在航天器连续控制任务中的工程化设计。

### 第二阶段：扩展到真正多智能体追逃

5. **P3 基于多智能体强化学习的轨道追逃博弈方法**入门 MADDPG 多星追逃。
6. **P8 PRD-MADDPG 脉冲轨道追逃**重点看预测奖励和脉冲机动。
7. **P12 Orbital Multi-Player Pursuit-Evasion Game with DRL**重点看合围捕获和去中心化协同。
8. **P14 SA-MAPPO 追踪-防御轨道博弈**
   重点看 self-attention、masking、MAPPO。

### 第三阶段：提高泛化、安全性和真实性

9. **P11 Satellite Chasers / DARL**重点看对手策略池、divergent loss、对抗泛化。
10. **P13 RF Sensing for Uncooperative Satellite Evasion**重点看传感器不确定性、RF 观测和受约束 RL。
11. **P10 稀疏奖励多智能体脉冲追逃**重点看 HER、分层网络、可达域子目标。
12. **N7 Safe RL 多卫星避碰**
    重点看 barrier function 和安全约束。

### 第四阶段：建立传统方法 baseline

13. **B1/B2/B3 传统微分博弈与脉冲制导材料**
    用于给 RL 方法提供理论对照和性能 baseline。

## 6. 当前代码项目改进方向

| 当前代码现状                  | 推荐参考         | 改进方向                                        | 预期收益                             |
| ----------------------------- | ---------------- | ----------------------------------------------- | ------------------------------------ |
| 只训练 pursuer，evader 未训练 | P4, P6, P11      | 加入 evader agent 的 replay buffer 和自博弈训练 | 从单方追踪变成真正追逃博弈           |
| 奖励主要依赖当前距离变化      | P6, P10          | 加入终端预测误差、稀疏奖励、HER                 | 提高训练目标一致性，减少人工 shaping |
| 动力学是简化 CW               | P1, P9, B3       | 加入 J2、TH、脉冲机动、锥形成像区               | 更接近真实轨道任务                   |
| 状态完全可观测                | P7, P13, N3      | 加噪声观测、部分观测、LSTM/GRU、传感器模型      | 提高真实感和策略鲁棒性               |
| 单追踪星单逃逸星              | P3, P8, P12, P14 | 扩展为多追踪星、多逃逸星、追踪-防御三方         | 支持集群博弈和协同策略               |
| PPO 单智能体框架              | P3, P8, P12, P14 | 引入 MADDPG、MAPPO、SA-MAPPO、分层 MARL         | 更适合多智能体协同/对抗              |
| 没有安全约束                  | P13, N7, N6      | 加 constrained RL、barrier function、滑模安全层 | 避免碰撞、燃料耗尽和不合理动作       |
| 没有对手泛化机制              | P11, P4          | 建立规则对手、学习对手、策略池和多样性损失      | 降低过拟合单一对手的风险             |
| 动画只展示真实状态            | P13              | 展示真实状态、估计状态、传感器误差、捕获域      | 更适合论文汇报和调试                 |

## 7. 建议的代码落地路线

### Step 1：补齐双智能体自博弈

- 给 evader 增加独立 replay buffer；
- 设计逃逸方奖励：远离 pursuer、减少燃料、避免进入捕获区；
- 支持 `--train-agent pursuer|evader|both`；
- 支持固定规则对手和学习型对手切换。

### Step 2：加入终端诱导奖励

- 在 `env.step()` 中用 CW 方程预测若干步后的终端相对状态；
- 增加终端预测误差奖励；
- 和当前即时距离奖励做消融对比。

### Step 3：信息非完备与 LSTM

- 将观测改为相对距离、方位角、带噪速度估计；
- 增加历史观测堆叠或 LSTM actor；
- 比较全观测 PPO 与部分观测 recurrent PPO。

### Step 4：多智能体扩展

- 将 `satellites` 环境扩展为 `n_pursuers + n_evaders`；
- 支持集中训练、分布式执行；
- 优先实现 MAPPO 或 MADDPG。

### Step 5：安全约束和工程化

- 加入燃料硬约束、碰撞硬约束；
- 引入 barrier function 作为安全惩罚或 safety shield；
- 动画中显示安全边界、捕获区、燃料余量和约束触发情况。

## 8. 可用于发论文的创新点设计

下面这些选题都围绕当前代码项目展开，尽量避免“单纯复现 PPO 追逃”这种创新性不足的问题。建议优先选择 **一个主创新点 + 一个辅助创新点 + 完整消融实验**，这样论文结构会更稳。

### 8.1 终端诱导 + 自博弈 PPO 的航天器追逃方法

**核心想法**

在当前 PPO 追逃代码基础上，同时训练追踪方和逃逸方，并引入终端诱导奖励。奖励不只看当前距离是否缩小，还预测若干步后的终端相对误差，引导追踪方在指定时间进入捕获区，引导逃逸方在终端时刻远离捕获区。

**可能创新点**

1. 将终端诱导思想从单方追踪扩展到双智能体自博弈训练。
2. 设计 pursuer/evader 对称但目标相反的终端预测奖励。
3. 证明或实验展示：终端诱导奖励比即时距离奖励更适合指定时间追逃任务。
4. 给出不同终端预测窗口长度下的性能对比。

**需要改的代码**

- 给 evader 增加独立 `ReplayBuffer` 和 `PPO_continuous` 更新；
- 在 `env.py` 中加入 `predict_terminal_state()`；
- 在奖励函数中加入终端预测误差项；
- 在 `main.py` 中加入 `--train-agent both` 或 `--self-play`。

**实验设计**

- Baseline 1：当前 PPO，只训练 pursuer；
- Baseline 2：双智能体 PPO，但无终端诱导；
- Proposed：双智能体 PPO + 终端诱导；
- 指标：捕获成功率、平均捕获时间、平均燃料消耗、最终距离、训练收敛速度。

**适合论文题目**

《基于终端诱导自博弈 PPO 的航天器轨道追逃博弈方法》

### 8.2 信息非完备下的 Recurrent PPO 轨道追逃决策

**核心想法**

当前环境默认给智能体完整状态，但真实非合作目标通常只有带噪声的距离、角度或相对方位信息。可以将全状态观测改成部分观测，并在策略网络中加入 LSTM/GRU，让智能体从历史观测中推断对手运动趋势。

**可能创新点**

1. 建立信息非完备轨道追逃 POMDP 模型；
2. 设计距离、方位角、带噪速度估计等受限观测；
3. 将 Recurrent PPO 用于追逃博弈，利用历史序列补偿状态不可观测；
4. 分析不同观测噪声、不同观测丢失率下策略鲁棒性。

**需要改的代码**

- 在 `env.py` 中新增 `get_observation(mode="full|partial|noisy")`；
- 在 PPO actor/critic 中加入 LSTM/GRU 版本；
- replay buffer 支持序列样本；
- 命令行增加 `--obs-mode partial`、`--noise-std`、`--recurrent`。

**实验设计**

- Full-state PPO；
- Partial-state MLP-PPO；
- Partial-state Recurrent-PPO；
- 不同噪声强度和观测间隔下对比成功率、燃料、最终距离。

**适合论文题目**

《信息非完备条件下基于循环近端策略优化的航天器轨道追逃博弈决策》

### 8.3 基于策略池和对手多样性的轨道追逃自博弈训练

**核心想法**

普通自博弈容易过拟合当前对手。可以维护 pursuer/evader 策略池，每隔若干轮保存历史策略，训练时随机抽取不同风格对手。同时加入策略多样性约束，使不同对手学习不同追逃风格。

**可能创新点**

1. 将策略池机制引入航天器轨道追逃 PPO 训练；
2. 设计对手采样机制，提高对未知对手的泛化能力；
3. 引入动作分布 KL 距离或轨迹差异度，鼓励策略多样性；
4. 形成“规则对手 + 历史学习对手 + 当前对手”的混合训练框架。

**需要改的代码**

- 增加 checkpoint pool 管理；
- 训练时随机加载历史 opponent；
- 增加 trajectory embedding 或动作 KL 多样性指标；
- 保存每个策略的胜率矩阵。

**实验设计**

- Fixed opponent；
- Naive self-play；
- Policy-pool self-play；
- Policy-pool + diversity loss；
- 指标：对未见规则逃逸策略的成功率、胜率矩阵、策略多样性、泛化测试。

**适合论文题目**

《基于多样化策略池自博弈的航天器轨道追逃强化学习方法》

### 8.4 稀疏奖励 + HER 的脉冲式轨道追逃强化学习

**核心想法**

当前奖励 shaping 较多，容易被质疑为人工调参。可以把任务改成更稀疏的“捕获成功/失败”奖励，再引入 HER（Hindsight Experience Replay）和子目标机制，提高稀疏奖励下训练效率。

**可能创新点**

1. 构建稀疏奖励轨道追逃任务，减少人工 dense reward 依赖；
2. 将 HER 用于轨道追逃，将失败轨迹中的中间状态重标记为子目标；
3. 融合轨道可达域约束，保证子目标物理可达；
4. 支持连续控制和脉冲控制两种动作形式对比。

**需要改的代码**

- 将动作改为固定间隔脉冲机动或保留连续动作作对比；
- replay buffer 支持 goal-conditioned state；
- 增加 HER 重标记逻辑；
- 奖励函数改为 goal-based sparse reward。

**实验设计**

- Dense PPO；
- Sparse PPO；
- Sparse PPO + HER；
- Sparse PPO + HER + reachable subgoal；
- 指标：成功率、样本效率、达到捕获区所需训练回合数。

**适合论文题目**

《面向稀疏奖励轨道追逃任务的目标重标记强化学习方法》

### 8.5 安全约束强化学习：带 Barrier Function 的卫星追逃策略

**核心想法**

追逃策略不能只追求捕获，还必须满足燃料、安全距离、追踪星间避碰、最大速度等约束。可以在 PPO 外层增加 barrier function 或 safety shield，对不安全动作进行投影/惩罚。

**可能创新点**

1. 建立带安全约束的轨道追逃 MDP；
2. 将控制 barrier function 引入追逃博弈奖励或动作修正层；
3. 同时保证捕获性能和安全约束满足率；
4. 对比“纯惩罚法”和“安全屏蔽法”的差异。

**需要改的代码**

- 在 `env.py` 中显式定义安全约束：最小安全距离、最大速度、燃料下界；
- 增加 `safety_filter(action, state)`；
- 记录 constraint violation；
- 动画中展示安全边界和违规时刻。

**实验设计**

- PPO without safety；
- PPO + penalty；
- PPO + barrier reward；
- PPO + safety shield；
- 指标：捕获成功率、约束违反次数、燃料消耗、动作修正次数。

**适合论文题目**

《融合控制障碍函数的安全强化学习轨道追逃博弈方法》

### 8.6 多追踪星合围捕获的 Attention-MAPPO 方法

**核心想法**

将当前单追踪星扩展为多追踪星，目标是协同合围一个逃逸星。用 MAPPO 做集中训练，并在 actor/critic 中引入 self-attention，让每颗卫星自动关注关键队友和目标。

**可能创新点**

1. 建立多追踪星合围捕获轨道博弈环境；
2. 用 self-attention 表征追踪星之间以及追踪星与逃逸星之间的关系；
3. 支持智能体数量变化或个体失效 masking；
4. 分析涌现的合围、拦截、包抄行为。

**需要改的代码**

- `env.py` 从固定两星扩展到 `n_pursuers + n_evaders`；
- 状态改为列表/张量形式；
- 实现 MAPPO 或 parameter-sharing PPO；
- 动画支持多星轨迹和合围区域展示。

**实验设计**

- Independent PPO；
- MADDPG；
- MAPPO；
- Attention-MAPPO；
- 指标：多追踪星捕获成功率、合围形成时间、队形保持误差、失效鲁棒性。

**适合论文题目**

《基于注意力多智能体近端策略优化的多航天器合围追逃博弈方法》

### 8.7 RF/光学观测不确定性下的追逃规避策略

**核心想法**

引入传感器模型，让智能体不能直接看到真实位置速度，而是看到带误差的 RF/光学观测。研究不同观测误差和探测范围下，强化学习策略如何完成追踪或逃逸。

**可能创新点**

1. 将 RF/光学观测误差模型加入轨道追逃环境；
2. 设计真实状态、估计状态、置信度共同输入的策略网络；
3. 训练对感知误差鲁棒的追逃/规避策略；
4. 动画可视化真实轨迹、估计轨迹和不确定性椭球。

**需要改的代码**

- 新增 sensor model；
- 支持 `--sensor rf|optical|perfect`；
- 状态中加入估计协方差或置信度；
- 动画显示估计误差。

**实验设计**

- Perfect observation；
- Noisy position/velocity；
- RF-like bearing/range observation；
- Observation dropout；
- 指标：成功率随噪声变化曲线、鲁棒性、误判导致的失败案例。

**适合论文题目**

《感知不确定条件下航天器轨道追逃博弈强化学习方法》

## 9. 推荐优先级

如果希望 **最快做出可投稿的小论文**，建议选择：

1. **终端诱导 + 双智能体 PPO 自博弈**改动适中，和当前代码最贴近，实验容易完成。
2. **信息非完备 + Recurrent PPO**创新性较好，容易讲清楚真实意义，适合中文核心/会议。
3. **安全约束 PPO + Barrier Function**
   工程价值强，实验指标清晰。

如果希望做 **更高水平、更完整的论文**，建议选择：

1. **Attention-MAPPO 多追踪星合围捕获**工作量较大，但创新性和展示效果最好。
2. **策略池 + 对手多样性自博弈**更偏 AI/MARL，适合强调泛化能力。
3. **稀疏奖励 + HER + 可达域子目标**
   算法味更强，但实现复杂度较高。

## 10. 最推荐的论文方案

综合当前代码基础、实现难度和创新性，最推荐方案是：

**题目方向：基于终端诱导自博弈 PPO 的信息非完备航天器轨道追逃博弈方法**

可以组合三个创新点：

1. **双智能体自博弈 PPO**同时训练追踪方和逃逸方，解决当前只训练追踪方的问题。
2. **终端诱导奖励**用预测终端误差替代单纯即时距离奖励，提高指定时间捕获能力。
3. **部分观测/噪声观测增强**
   加入观测噪声或历史观测堆叠，提高工程真实性。

推荐实验章节：

1. 环境建模：CW 动力学、状态、动作、捕获条件；
2. 方法：Self-play PPO、终端诱导奖励、部分观测增强；
3. 对比实验：原始 PPO、双 PPO、双 PPO + 终端诱导、双 PPO + 终端诱导 + 部分观测；
4. 消融实验：终端预测窗口、噪声强度、捕获距离、燃料约束；
5. 可视化分析：轨迹图、动画、距离曲线、奖励曲线、燃料曲线。

这条路线的优点是：和已有文献相关但不完全重复，代码改动可控，也能自然接上你现在已经有的训练、推理和动画展示功能。

## 11. 更新后的首选论文方案：捕获域感知课程自博弈

上一节给出的“终端诱导 + 自博弈 PPO + 信息非完备”方案实现风险较低，但创新性偏向已有方法组合。更适合作为工程增强方案，而不是最有辨识度的主创新。更推荐将论文主线调整为：

**基于捕获域感知课程自博弈的航天器轨道追逃强化学习方法**

英文题目可写为：

**Capture-Domain-Aware Curriculum Self-Play Reinforcement Learning for Spacecraft Orbital Pursuit-Evasion Games**

### 11.1 核心思想

传统 PPO 追逃训练通常依赖距离、燃料、速度方向等人工奖励项。这样虽然能训练出策略，但容易被质疑为 reward shaping 堆叠，且没有充分利用轨道追逃问题本身的结构。

本方案的核心是：把轨道追逃微分博弈中的 **捕获域、可达域、屏障面、可捕获性** 这些物理和博弈结构引入强化学习训练，使智能体不是盲目学习“距离变小”，而是围绕“当前状态是否可被捕获、如何跨越捕获域边界、如何从争夺区进入捕获区”进行学习。

具体来说，可以构造一个 `capture_domain_score`，根据当前相对位置、相对速度、燃料余量、捕获半径、动力学预测窗口等信息，评估当前状态对追踪方的可捕获难度。然后用这个指标同时驱动：

1. 状态难度分层；
2. 课程训练采样；
3. 区域自适应奖励；
4. 自博弈对手强度调度；
5. 训练和推理阶段的态势解释。

### 11.2 主要创新点

#### 创新点 1：捕获域感知态势评估指标

设计一个面向轨道追逃博弈的 `capture_domain_score`，不再只用欧氏距离描述态势，而是融合以下因素：

- 相对位置；
- 相对速度；
- 捕获半径；
- 追踪方和逃逸方燃料余量；
- 预测窗口内的动力学可达性；
- 追踪方速度方向是否有利于进入捕获域；
- 逃逸方是否具备逃离当前捕获邻域的能力。

该指标可将状态划分为：

- **easy / pursuer-dominant**：追踪方明显处于可捕获优势；
- **medium / contested**：双方都有机会，属于关键博弈区；
- **hard / evader-dominant**：逃逸方优势明显，追踪方需要先改变可达性。

这比单纯的距离指标更符合轨道追逃任务，因为远近并不完全决定可捕获性，相对速度方向、燃料、动力学可达性同样重要。

#### 创新点 2：捕获域感知的区域自适应奖励

根据 `capture_domain_score` 所处区域，动态调整奖励目标：

| 区域       | 训练重点     | 奖励设计                                                     |
| ---------- | ------------ | ------------------------------------------------------------ |
| 远离捕获域 | 提高可捕获性 | 奖励 capture_domain_score 改善、相对速度方向修正、可达性增强 |
| 可争夺区   | 进入捕获优势 | 奖励跨越捕获域边界、降低终端预测误差、控制燃料消耗           |
| 捕获邻域   | 稳定捕获     | 奖励进入并保持捕获区，惩罚过冲和高燃料消耗                   |

这样奖励函数不再是固定权重的人工拼接，而是和轨道追逃任务阶段绑定。它的优势是可解释性更强，也更容易做消融实验。

#### 创新点 3：捕获难度分层的课程自博弈训练

训练初期从 easy 状态开始，让追踪方先学会基本接近和捕获；随后逐步提高 medium 和 hard 状态比例，使策略学习更困难的争夺区和逃逸方优势区。

课程可以按如下方式调度：

```text
阶段 1：easy 70%, medium 25%, hard 5%
阶段 2：easy 40%, medium 45%, hard 15%
阶段 3：easy 20%, medium 50%, hard 30%
阶段 4：按验证集失败分布自适应采样
```

自博弈训练中，pursuer 和 evader 的初始状态不再随机生成，而是按照捕获难度分布采样。这样可以避免训练样本过多集中在“太容易”或“根本不可捕获”的状态，提高样本效率。

#### 创新点 4：失败诊断驱动的困难样本生成

进一步增强方案是：在每轮评估后分析失败轨迹，并按失败原因生成下一阶段训练样本。失败原因可分为：

- 追踪方燃料不足；
- 接近方向错误；
- 终端速度过大导致过冲；
- 过晚进入可争夺区；
- 逃逸方诱导追踪方偏离；
- 追踪方进入局部循环轨迹；
- 追踪方短期接近但长期可捕获性下降。

然后把这些失败状态加入 hard sample buffer，在后续训练中提高采样概率。这相当于形成：

```text
失败轨迹分析 -> 困难状态挖掘 -> 对抗样本生成 -> 自博弈策略修复
```

这比普通 self-play 更有针对性，也更容易体现“轨道追逃任务机理驱动”的特色。

### 11.3 与已有文献的区别

| 已有方向                    | 已有工作特点                 | 本方案区别                                                         |
| --------------------------- | ---------------------------- | ------------------------------------------------------------------ |
| 终端诱导奖励                | 强调终端时刻误差预测         | 本方案关注整个状态是否跨入捕获域，以及不同捕获难度区域下的训练调度 |
| barrier / capture zone 方法 | 多用于解析求解或辅助拦截策略 | 本方案把捕获域转化为 RL 的状态难度评估、课程采样和区域奖励机制     |
| 普通 self-play              | 让双方互相训练               | 本方案按捕获难度和失败类型组织 self-play 样本                      |
| 策略池对抗训练              | 强调对手策略多样性           | 本方案强调轨道动力学可捕获性和困难状态生成                         |
| 稀疏奖励/HER                | 强调目标重标记               | 本方案强调可达性、捕获域边界和博弈态势分区                         |

因此，本方案不是简单拼接“终端诱导 + PPO + 信息非完备”，而是提出一个围绕 **捕获域结构** 展开的训练框架。

### 11.4 可写成论文的贡献表述

论文贡献可以写成以下三点：

1. **提出一种轨道追逃捕获域感知态势评估指标。**该指标融合相对状态、燃料约束和动力学预测，用于刻画当前状态的可捕获性，并将追逃状态划分为追踪方优势区、双方争夺区和逃逸方优势区。
2. **设计一种捕获域感知的区域自适应奖励机制。**根据不同捕获难度区域动态调整奖励目标，使策略在远域接近、争夺区突破和近域捕获阶段学习不同的控制行为，从而提升奖励设计的物理可解释性。
3. **构建一种基于捕获难度分层和失败诊断的课程自博弈训练框架。**
   通过动态调整初始状态分布和困难样本采样比例，提高策略在复杂初始条件和未知对手策略下的训练效率与泛化能力。

### 11.5 对当前代码的具体改造

#### 需要新增的模块

1. `capture_domain.py`

   - `compute_capture_score(state, env_params)`
   - `classify_capture_region(score)`
   - `predict_reachable_gap(state, horizon)`
2. `curriculum.py`

   - `sample_initial_state_by_difficulty(level)`
   - `update_curriculum(success_rate)`
   - `hard_case_buffer`
3. `failure_diagnosis.py`

   - `diagnose_failure(trajectory)`
   - `extract_hard_states(trajectory)`

#### 需要修改的现有文件

| 文件                  | 修改点                                       |
| --------------------- | -------------------------------------------- |
| `env.py`            | 增加捕获域评分、状态难度分类、区域自适应奖励 |
| `main.py`           | 增加课程训练参数、难度采样、自博弈训练入口   |
| `ppo_continuous.py` | 可暂时不改，先复用现有 PPO                   |
| `plot_function.py`  | 动画中显示 capture score、当前区域、失败原因 |
| `replaybuffer.py`   | 如果加入困难样本回放，再扩展优先采样         |

#### 建议命令行参数

```powershell
--use-capture-domain
--curriculum
--difficulty easy|medium|hard|adaptive
--capture-horizon 10
--hard-case-ratio 0.3
--diagnose-failure
```

### 11.6 实验设计

#### 对比方法

| 方法                | 说明                                                 |
| ------------------- | ---------------------------------------------------- |
| PPO-Baseline        | 当前代码，固定奖励，只训练追踪方                     |
| Self-Play PPO       | 同时训练追踪方和逃逸方                               |
| Terminal-Guided PPO | 加入终端预测误差奖励                                 |
| Curriculum PPO      | 使用普通距离难度课程                                 |
| Proposed            | 捕获域感知区域奖励 + 捕获难度课程 + 失败诊断困难样本 |

#### 消融实验

| 消融项                   | 目的                                    |
| ------------------------ | --------------------------------------- |
| 去掉 capture score       | 验证捕获域态势指标是否有效              |
| 去掉区域自适应奖励       | 验证动态奖励是否优于固定奖励            |
| 去掉课程采样             | 验证难度分层是否提高样本效率            |
| 去掉失败诊断 hard buffer | 验证失败样本挖掘是否提升 hard case 性能 |
| 不同预测窗口             | 分析 capture horizon 对性能的影响       |

#### 评价指标

- 捕获成功率；
- 平均捕获步数；
- 最终相对距离；
- 平均燃料消耗；
- hard case 成功率；
- 训练收敛速度；
- 不同初始条件下泛化能力；
- 捕获域跨越成功率；
- 约束违反次数；
- 策略对未知逃逸策略的胜率。

### 11.7 论文结构建议

1. 引言强调轨道追逃博弈中仅依赖距离奖励的局限，以及捕获域结构对策略学习的重要性。
2. 问题建模CW/TH 动力学、状态动作空间、捕获条件、燃料约束、博弈目标。
3. 捕获域感知课程自博弈方法介绍 capture score、区域自适应奖励、课程采样、失败诊断。
4. 强化学习训练框架PPO/self-play 训练流程，算法伪代码。
5. 实验与分析对比实验、消融实验、泛化实验、可视化轨迹分析。
6. 结论
   总结捕获域先验对训练效率和泛化性能的提升。

### 11.8 为什么这个方案更有独特性

这个方案的独特性在于，它不是把已有强化学习技巧简单组合，而是把 **轨道追逃博弈的领域结构** 变成 RL 训练机制：

- 捕获域不是只用于理论分析，而是进入训练样本组织；
- 可捕获性不是只用于评价，而是进入奖励和课程；
- 失败轨迹不是只用于展示，而是反过来生成困难训练样本；
- 动画和推理日志不只是可视化，而是可以作为失败诊断和课程更新的数据来源。

因此，它更容易形成一条清晰的论文主线：

```text
轨道追逃理论结构 -> 捕获域态势评估 -> 课程自博弈训练 -> 策略泛化提升
```

这比“终端诱导 + 信息非完备 + PPO”更像一个独立方法，也更能体现本项目自己的贡献。
