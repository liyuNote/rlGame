# PPO 卫星追逃博弈项目说明

## 1. 项目在做什么

本项目实现了一个基于 PPO（Proximal Policy Optimization，近端策略优化）的连续动作强化学习实验，用于模拟二维/三维近似轨道环境中的“追踪卫星”和“逃逸卫星”博弈。

核心任务是：

- 追踪方（pursuer）观察双方相对位置、相对速度以及各自绝对状态；
- PPO 策略网络输出追踪方在 3 个轴向上的加速度/速度增量控制量；
- 环境用 Clohessy-Wiltshire（CW）相对运动模型推进下一时刻状态；
- 如果追踪方进入捕获距离 `d_capture` 内，则判定追踪成功并给予高奖励；
- 如果达到最大回合步数或燃料耗尽，则回合结束；
- 训练完成后保存追踪方 actor/critic 模型，用于后续推理。

当前代码主要训练追踪方。逃逸方也实例化了一个策略网络，但训练循环只把追踪方经验写入 replay buffer 并更新追踪方 PPO；逃逸方在当前版本中相当于一个未训练的扰动策略。

## 2. 文件结构

| 文件 | 作用 |
| --- | --- |
| `main.py` | 项目入口；解析命令行参数，构建环境，执行训练或推理 |
| `env.py` | 卫星追逃环境；定义状态空间、动作空间、奖励函数、终止条件 |
| `ppo_continuous.py` | PPO 连续动作算法；包含 actor、critic、动作采样、GAE 和 PPO 更新 |
| `replaybuffer.py` | 固定批量经验缓存；保存状态、动作、奖励、下一状态、终止标记 |
| `satellite_function.py` | 轨道动力学推进；当前主要使用 CW 状态转移矩阵 |
| `normalization.py` | 状态归一化和奖励缩放工具；当前主流程未启用 |
| `plot_function.py` | 训练奖励曲线和三维轨迹绘图 |
| `environment.yml` | conda 环境定义 |
| `setup_env.ps1` | Windows PowerShell 一键创建/更新环境脚本 |

## 3. 强化学习建模

### 3.1 智能体

- `pursuer_agent`：追踪方，使用 PPO 训练。
- `evader_agent`：逃逸方，当前只用于产生逃逸动作，不参与训练更新。

### 3.2 状态空间

环境返回 18 维连续状态：

| 切片 | 含义 | 维度 |
| --- | --- | --- |
| `s[0:3]` | 追踪方位置 - 逃逸方位置 | 3 |
| `s[3:6]` | 追踪方速度 - 逃逸方速度 | 3 |
| `s[6:9]` | 追踪方绝对位置 | 3 |
| `s[9:12]` | 追踪方绝对速度 | 3 |
| `s[12:15]` | 逃逸方绝对位置 | 3 |
| `s[15:18]` | 逃逸方绝对速度 | 3 |

### 3.3 动作空间

动作是 3 维连续向量：

```text
[ax, ay, az]
```

每个分量会被裁剪到 `[-1.6, 1.6]`。在环境中动作直接累加到对应卫星速度上，可理解为每个控制周期的速度增量或等效加速度控制量。

### 3.4 环境推进

每一步执行：

1. 裁剪追踪方和逃逸方动作；
2. 根据动作消耗燃料；
3. 将动作累加到速度；
4. 使用 CW 状态转移矩阵推进 `env_dt` 时间；
5. 计算新距离、危险区标记、奖励和终止条件。

### 3.5 终止条件

回合结束条件：

- `self.dis <= self.d_capture`：追踪方进入捕获距离，成功捕获；
- `epsiode_count >= self.max_episode_steps`：达到单回合最大步数；
- `self.fuel_c <= 0`：追踪方燃料耗尽。

## 4. 奖励函数

当前奖励由多个部分叠加：

| 项 | 代码位置 | 含义 |
| --- | --- | --- |
| 距离变化奖励 | `reward = 1.0 if self.dis < old_distance else -1.0` | 比上一时刻更接近目标给正奖励，否则负奖励 |
| 捕获区间奖励 | `d_capture <= dis <= 4*d_capture` | 鼓励进入目标附近范围 |
| 危险区奖励 | `calculate_number_hanger_area()` | 如果正在接近且距离小于 `4*d_capture`，认为处于有效接近区 |
| 位置接近奖励 | `distance_reward()` | 距离越接近 `d_capture` 附近，奖励越高 |
| 速度惩罚 | `velocity_penalty()` | 速度过大给惩罚，避免策略无限增速 |
| 朝向奖励 | `direction_reward()` | 速度方向越朝向逃逸方，奖励越高 |
| 燃料保留奖励 | `fuel_conservation_reward()` | 剩余燃料比例越高，奖励越高 |
| 捕获成功奖励 | `win_reward` | 成功捕获直接给 `100` |
| 失败/超时奖励 | `burn_reward` | 当前为 `0` |

## 5. 主要参数说明

### 5.1 训练参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `max_train_steps` | `400`（CLI 默认） | 训练回合数；代码变量名沿用原文，实际是 episode 数 |
| `max_episode_steps` | `64`（CLI 默认） | 单个回合最多环境步数 |
| `batch_size` | `64`（CLI 默认） | replay buffer 达到该条数后触发一次 PPO 更新 |
| `mini_batch_size` | `32`（CLI 默认） | PPO 更新时每个小批量样本数 |
| `K_epochs` / `--k-epochs` | `3`（CLI 默认） | 每批数据重复优化的轮数 |
| `policy_dist` | `Gaussian` | 策略分布；支持 `Gaussian` 和 `Beta` |
| `hidden_width` | `128`（CLI 默认） | actor/critic 隐藏层宽度 |
| `chkpt_dir` | `checkpoints` | 模型保存目录 |
| `pre_train` | `False` | 是否从已有 checkpoint 继续训练 |
| `no_plot` | `False` | 是否关闭绘图输出 |

### 5.2 PPO 超参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `lr_a` | `0.0002` | actor 学习率 |
| `lr_c` | `0.0002` | critic 学习率 |
| `gamma` | `0.99` | 奖励折扣因子 |
| `lamda` | `0.95` | GAE 的 lambda 参数 |
| `epsilon` | `0.1` | PPO clipped surrogate objective 的裁剪范围 |
| `entropy_coef` | `0.01` | 策略熵奖励系数，鼓励探索 |
| `use_adv_norm` | `True` | 是否对 advantage 标准化 |
| `use_lr_decay` | `True` | 是否按训练进度线性衰减学习率 |
| `use_grad_clip` | `True` | 是否裁剪梯度范数 |
| `use_orthogonal_init` | `True` | 是否对线性层做正交初始化 |
| `set_adam_eps` | `True` | Adam 优化器 eps 是否设为 `1e-5` |
| `use_tanh` | `True` | 隐藏层激活函数是否使用 Tanh；否则使用 ReLU |

### 5.3 环境参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `Pursuer_position` | `[200000, 0, 0]` | 追踪方初始位置 |
| `Pursuer_vector` | `[0, 0, 0]` | 追踪方初始速度 |
| `Escaper_position` | `[18000, 0, 0]` | 逃逸方初始位置 |
| `Escaper_vector` | `[0, 0, 0]` | 逃逸方初始速度 |
| `d_capture` | CLI 默认 `20000` | 捕获距离阈值 |
| `dis_safe` | `1000` | 安全距离参数，目前主要保留为扩展字段 |
| `fuel_c` | `320` | 追踪方初始燃料 |
| `fuel_t` | `320` | 逃逸方初始燃料 |
| `env_dt` | `100` | 每一步 CW 动力学推进时间 |
| `win_reward` | `100` | 捕获成功奖励 |
| `burn_reward` | `0` | 超时或燃料耗尽奖励 |

## 6. 如何运行

创建或更新环境：

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
```

激活环境：

```powershell
conda activate rlgame-ppo
```

快速训练：

```powershell
python main.py --mode train --max-train-steps 400 --batch-size 64 --mini-batch-size 32 --max-episode-steps 64
```

继续训练：

```powershell
python main.py --mode train --pre-train
```

推理：

```powershell
python main.py --mode test --max-episode-steps 64
```

不生成图片：

```powershell
python main.py --mode train --no-plot
python main.py --mode test --no-plot
```

## 7. 当前实现的局限和后续方向

- 当前只训练追踪方，逃逸方未训练；如果要做真正双智能体博弈，需要为逃逸方建立独立 replay buffer 并设计逃逸奖励。
- 动作目前直接加到速度上，是简化控制模型；如果要贴近真实轨道控制，应引入质量、推力、比冲和更真实的推进模型。
- CW 方程适用于近圆轨道和相对距离不太大的近似场景；复杂轨道场景可以切换到 `Numerical_calculation_method` 或更完整的轨道动力学模型。
- 奖励权重是工程化初值，后续可以通过消融实验、网格搜索或贝叶斯优化调整。
