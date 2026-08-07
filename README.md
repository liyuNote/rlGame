# PPO Satellite Pursuit-Evasion

这是一个基于 PPO（Proximal Policy Optimization）的卫星追逃博弈项目。环境中有两个航天器：

- `pursuer`：追踪方，目标是接近并捕获 `evader`。
- `evader`：逃逸方，目标是拉开距离、避免被捕获并节省燃料。

当前代码支持三种训练模式：只训练 `pursuer`、只训练 `evader`、或同时训练双方。未训练的一方可以使用固定规则控制器、零控制漂移，或已有 PPO checkpoint 作为学习型对手。

## 环境准备

首次使用时，在项目根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
```

然后激活 conda 环境：

```powershell
conda activate rlgame-ppo
```

如果你需要安装 CUDA 版 PyTorch，可以参考：

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_gpu_pytorch.ps1
```

## 快速开始

快速训练默认的追踪方模型：

```powershell
python main.py --mode train
```

快速验证训练流程：

```powershell
python main.py --mode train --max-train-steps 2 --batch-size 4 --mini-batch-size 2 --max-episode-steps 4 --k-epochs 1 --no-plot
```

加载 checkpoint 做确定性推理：

```powershell
python main.py --mode test
```

只运行推理逻辑、不生成图片或 GIF：

```powershell
python main.py --mode test --no-plot
```

查看全部命令行参数：

```powershell
python main.py --help
```

## 当前运行逻辑

### 环境

`main.py` 默认构造一个沿 X 轴分离的追逃场景：

- pursuer 初始位置：`[initial_distance, 0, 0]`，默认 `initial_distance=80000`
- evader 初始位置：`[0, 0, 0]`
- 双方初始速度：`[0, 0, 0]`
- 单步动力学推进时间：`env_dt=100`
- 捕获距离：`d_capture=20000`

环境状态是 18 维向量：

| 切片 | 含义 |
| --- | --- |
| `s[0:3]` | pursuer 与 evader 的相对位置 `r_p - r_e` |
| `s[3:6]` | pursuer 与 evader 的相对速度 `v_p - v_e` |
| `s[6:9]` | pursuer 绝对位置 |
| `s[9:12]` | pursuer 绝对速度 |
| `s[12:15]` | evader 绝对位置 |
| `s[15:18]` | evader 绝对速度 |

动作是 3 维连续向量，每个分量会裁剪到 `[-1.6, 1.6]`。当前简化模型中，动作直接作为速度增量加入对应航天器速度，然后使用 Clohessy-Wiltshire 状态转移推进位置和速度。

### PPO 策略

`--policy-dist Gaussian` 是默认策略分布。Gaussian actor 直接输出 `[-max_action, max_action]` 范围内的连续动作均值，并在训练时采样。

`--policy-dist Beta` 会先输出 `[0, 1]` 区间动作，再映射到环境动作范围：

```text
env_action = 2 * (policy_action - 0.5) * max_action
```

训练和推理必须使用同样的 `--policy-dist`、`--hidden-width`、状态维度和动作维度，否则 checkpoint 无法加载。

### 状态缩放

默认启用状态缩放：

- 位置相关维度除以 `--state-position-scale`，默认 `200000`
- 速度相关维度除以 `--state-velocity-scale`，默认 `10000`

可以用 `--no-state-norm` 关闭。

### 对手策略选项

`--opponent-policy` 只影响未训练的一方。可选值如下：

| 取值 | 行为 | 是否加载 checkpoint |
| --- | --- | --- |
| `fixed` | 使用几何固定规则，pursuer 追击、evader 远离 | 否 |
| `learning` | 使用对应 PPO 策略作为对手 | 是 |
| `idle` | 动作恒为 `[0, 0, 0]`，只随 CW 动力学自然漂移 | 否 |

### 固定规则对手

`--opponent-policy fixed` 是默认对手策略。当前 `fixed_rule_action()` 使用状态中的相对位置 `raw_state[0:3]` 计算方向，并输出固定强度动作：

```text
action = fixed_rule_action_scale * max_action * direction / norm
```

这里的 `direction` 是从 pursuer 指向 evader 的全局方向。因此同一个动作方向给 pursuer 使用时表现为追击，给 evader 使用时表现为远离 pursuer。

### 零控制漂移对手

`--opponent-policy idle` 表示未训练的一方不做任何主动机动，动作恒为 `[0, 0, 0]`。该航天器仍然会经过环境中的 Clohessy-Wiltshire 动力学推进，所以它不是“冻结在原地”，而是只做自然轨道运动。

注意：默认 `--initial-distance 80000` 在 CW 动力学下仍然可能自然发散；如果从零开始训练不稳定，可以先用较短初始距离做课程训练。

## 常用训练命令

训练 pursuer，evader 使用固定规则：

```powershell
python main.py --mode train --train-agent pursuer --opponent-policy fixed
```

训练 pursuer，evader 只做零控制轨道漂移：

```powershell
python main.py --mode train --train-agent pursuer --opponent-policy idle
```

更容易的 idle 课程训练起点：

```powershell
python main.py --mode train --train-agent pursuer --opponent-policy idle --initial-distance 40000
```

训练 evader，pursuer 使用固定规则：

```powershell
python main.py --mode train --train-agent evader --opponent-policy fixed
```

同时训练 pursuer 和 evader：

```powershell
python main.py --mode train --train-agent both
```

使用已有 checkpoint 继续训练当前学习策略角色：

```powershell
python main.py --mode train --pre-train
```

训练 pursuer，并加载 evader checkpoint 作为学习型对手：

```powershell
python main.py --mode train --train-agent pursuer --opponent-policy learning --pre-train
```

训练更久：

```powershell
python main.py --mode train --max-train-steps 5000 --max-episode-steps 500
```

指定计算设备：

```powershell
python main.py --mode train --device cuda
```

`--device auto` 会按当前 PyTorch 可用性自动选择设备。

## 训练可视化

训练时默认会保存两张图：

- `outputs/train_reward.png`：每个 episode 的 reward 和累计平均 reward。
- `outputs/training_live_dashboard.png`：reward、距离、捕获率、loss、学习率和最近一轮 XY 追逃轨迹。

直接训练并生成可视化：

```powershell
python main.py --mode train
```

训练过程中每隔 N 个 episode 刷新一次 dashboard，默认是 `5`：

```powershell
python main.py --mode train --live-dashboard-freq 10
```

弹出实时训练 dashboard 窗口：

```powershell
python main.py --mode train --live-dashboard-window
```

自定义 dashboard 保存路径：

```powershell
python main.py --mode train --live-dashboard-output outputs\my_training_dashboard.png
```

关闭训练过程中的 dashboard 刷新：

```powershell
python main.py --mode train --live-dashboard-freq 0
```

完全不生成训练图片：

```powershell
python main.py --mode train --no-plot
```

## 推理可视化

推理阶段使用策略均值动作，不再随机采样。默认会输出 XY 静态轨迹图和 GIF 动画：

```powershell
python main.py --mode test
```

推理图会自动缩放坐标轴，确保 pursuer/evader 的完整 XY 轨迹、初末位置和捕获半径都在画面内。GIF 和弹窗右侧状态面板会实时显示双方剩余燃料；推理结束时终端也会打印 pursuer 和 evader 的最终剩余燃料。

推理阶段默认生成：

- `outputs/trajectory.png`
- `outputs/satellite_game.gif`

自定义动画文件和帧率：

```powershell
python main.py --mode test --animation-output outputs\satellite_game.gif --animation-fps 10
```

推理时弹出追逃动画窗口：

```powershell
python main.py --mode test --inference-window
```

## Checkpoint

checkpoint 目录由 `--chkpt-dir` 控制，默认是：

```text
checkpoints/
```

文件名由 agent 名称和策略分布拼接：

| 角色 | Gaussian actor | Beta actor | critic |
| --- | --- | --- | --- |
| pursuer | `agent_pursuer_actor_Gaussian` | `agent_pursuer_actor_Beta` | `agent_pursuer_critic` |
| evader | `agent_evader_actor_Gaussian` | `agent_evader_actor_Beta` | `agent_evader_critic` |

训练结束时只保存被训练的角色：

- `--train-agent pursuer`：保存 pursuer actor/critic
- `--train-agent evader`：保存 evader actor/critic
- `--train-agent both`：保存双方 actor/critic

`--pre-train` 会加载当前运行中使用学习策略的角色：

- 被训练角色总是使用学习策略，会尝试加载其 checkpoint。
- 当 `--opponent-policy learning` 时，未训练对手也会尝试加载 checkpoint。
- 当 `--opponent-policy fixed` 时，未训练对手不会加载 checkpoint。

## 主要参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--mode` | `train` | `train` 训练，`test` 推理 |
| `--max-train-steps` | `1000` | 训练 episode 数 |
| `--max-episode-steps` | `500` | 单个 episode 最大环境步数 |
| `--batch-size` | `64` | buffer 收集多少 transition 后触发 PPO 更新 |
| `--mini-batch-size` | `32` | PPO 更新时的小批量样本数 |
| `--k-epochs` | `3` | 每批数据重复优化轮数 |
| `--hidden-width` | `128` | actor/critic 两层 MLP 的隐藏层宽度 |
| `--policy-dist` | `Gaussian` | 策略分布：`Gaussian` 或 `Beta` |
| `--chkpt-dir` | `checkpoints` | checkpoint 保存和读取目录 |
| `--d-capture` | `20000.0` | 捕获距离阈值 |
| `--train-agent` | `pursuer` | 训练 `pursuer`、`evader` 或 `both` |
| `--opponent-policy` | `fixed` | 可选 `{fixed, learning, idle}`，控制未训练对手策略 |
| `--fixed-rule-action-scale` | `0.8` | 固定规则动作强度占 `max_action` 的比例 |
| `--initial-distance` | `80000.0` | pursuer 与 evader 初始 X 轴距离 |
| `--state-position-scale` | `200000.0` | 状态中位置维度的缩放尺度 |
| `--state-velocity-scale` | `10000.0` | 状态中速度维度的缩放尺度 |
| `--no-state-norm` | 关闭 | 不对输入 PPO 的状态做缩放 |
| `--no-plot` | 关闭 | 不生成图像或动画 |
| `--inference-window` | 关闭 | 推理时弹出追逃动画窗口，关闭窗口后命令结束 |
| `--device` | `auto` | PyTorch 设备：`auto`、`cpu`、`cuda`、`cuda:0`、`mps`、`xpu` |

## 常见问题

### checkpoint 加载时报 size mismatch

这通常说明当前命令创建的网络结构和 checkpoint 不一致。重点检查：

- `--hidden-width`
- `--policy-dist`
- `--train-agent` 和 checkpoint 文件名
- 状态维度、动作维度

解决方式是推理时使用训练时同样的参数，或用当前参数重新训练并覆盖 checkpoint。

### test 模式找不到 evader checkpoint

`test` 会根据 `uses_learning_policy()` 判断哪些角色使用 PPO。默认 `--train-agent pursuer`，所以默认只加载 pursuer checkpoint，evader 用固定规则。如果你设置了 `--opponent-policy learning`，则 evader 也需要有对应 checkpoint；如果设置 `--opponent-policy idle`，evader 不加载 checkpoint，动作恒为零。
