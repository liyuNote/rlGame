# PPO Satellite Pursuit-Evasion

这是一个基于 PPO（Proximal Policy Optimization）的卫星追逃博弈项目。项目训练追踪卫星 `pursuer` 在连续动作空间中机动，目标是在轨道近似动力学环境中接近并捕获逃逸卫星 `evader`。推理阶段可以输出静态轨迹图，也可以生成带状态信息面板的卫星博弈动画。

## 环境准备

首次使用时，在项目目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
```

激活 conda 环境：

```powershell
conda activate rlgame-ppo
```

如果你使用 VS Code，也可以直接使用 `.vscode/launch.json` 中的调试配置。

## 快速训练

使用默认推荐参数训练：

```powershell
python main.py --mode train --max-train-steps 400 --batch-size 64 --mini-batch-size 32 --max-episode-steps 64
```

训练完成后，模型会保存到：

```text
checkpoints/
```

默认会生成训练奖励曲线：

```text
outputs/train_reward.png
```

如果只想快速验证训练流程：

```powershell
python main.py --mode train --max-train-steps 2 --batch-size 4 --mini-batch-size 2 --max-episode-steps 4 --k-epochs 1 --no-plot
```

继续从已有 checkpoint 训练：

```powershell
python main.py --mode train --pre-train
```

注意：继续训练时，网络结构参数必须和保存 checkpoint 时一致，尤其是 `--hidden-width` 和 `--policy-dist`。

## 快速推理

不生成图片，只打印测试结果：

```powershell
python main.py --mode test --max-episode-steps 64 --no-plot
```

生成静态轨迹图和动画：

```powershell
python main.py --mode test --max-episode-steps 64
```

默认输出：

```text
outputs/trajectory.png
outputs/satellite_game.gif
```

自定义动画文件和帧率：

```powershell
python main.py --mode test --max-episode-steps 64 --animation-output outputs\game.gif --animation-fps 10
```

## 命令行参数说明

所有参数都可以通过下面命令查看：

```powershell
python main.py --help
```

### 通用参数

| 参数 | 可选值/类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--mode` | `train` 或 `test` | `train` | 运行模式。`train` 表示训练追踪方 PPO；`test` 表示加载 checkpoint 做推理。 |
| `--max-episode-steps` | 整数 | `64` | 单个 episode 最多运行多少个环境步。训练和推理都会使用。 |
| `--hidden-width` | 整数 | `128` | Actor 和 Critic 隐藏层宽度。训练和推理必须保持一致，否则 checkpoint 无法加载。 |
| `--policy-dist` | `Gaussian` 或 `Beta` | `Gaussian` | 策略分布类型。`Gaussian` 直接输出连续动作；`Beta` 输出 `[0,1]` 后映射到动作范围。训练和推理必须保持一致。 |
| `--chkpt-dir` | 路径字符串 | `checkpoints` | checkpoint 保存/读取目录。训练时写入该目录，推理时从该目录加载。 |
| `--d-capture` | 浮点数 | `20000.0` | 捕获距离阈值。追踪星与逃逸星距离小于该值时判定捕获成功。 |
| `--no-plot` | 开关 | 关闭 | 加上该参数后不生成图片或动画，只执行训练/推理逻辑。 |

### 训练参数

| 参数 | 可选值/类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--max-train-steps` | 整数 | `400` | 训练 episode 数。变量名沿用原代码，但当前实现中它表示训练多少个回合。 |
| `--batch-size` | 整数 | `64` | ReplayBuffer 收集多少条 transition 后触发一次 PPO 更新。 |
| `--mini-batch-size` | 整数 | `32` | PPO 更新时，每次随机小批量优化使用多少条样本。通常应小于等于 `--batch-size`。 |
| `--k-epochs` | 整数 | `3` | 每收集一个 batch 后，对同一批数据重复优化多少轮。值越大更新越充分，但也更慢，且可能过拟合当前 batch。 |
| `--pre-train` | 开关 | 关闭 | 从 `--chkpt-dir` 中已有 checkpoint 继续训练。要求 checkpoint 和当前网络结构参数一致。 |

### 推理与动画参数

| 参数 | 可选值/类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--animation-output` | 路径字符串 | `outputs/satellite_game.gif` | 推理动画 GIF 的保存路径。仅在 `--mode test` 且未使用 `--no-plot` 时生效。 |
| `--animation-fps` | 整数 | `8` | 动画帧率。值越大动画越流畅，但 GIF 体积通常也越大。 |

## 常见用法组合

### 训练一个默认模型

```powershell
python main.py --mode train
```

等价于使用默认参数训练 `400` 个 episode。

### 训练更久

```powershell
python main.py --mode train --max-train-steps 5000 --max-episode-steps 128
```

### 使用更小网络快速调试

```powershell
python main.py --mode train --hidden-width 32 --max-train-steps 10 --batch-size 8 --mini-batch-size 4 --max-episode-steps 8 --no-plot
```

对应推理也必须带相同的 `--hidden-width 32`：

```powershell
python main.py --mode test --hidden-width 32 --max-episode-steps 8 --no-plot
```

### 只推理并生成动画

```powershell
python main.py --mode test --max-episode-steps 64 --animation-output outputs\satellite_game.gif --animation-fps 8
```

## Checkpoint 兼容性提醒

推理时会先根据命令行参数创建网络，再加载 checkpoint。因此这些参数必须和训练时一致：

- `--hidden-width`
- `--policy-dist`
- 状态维度 `state_dim`
- 动作维度 `action_dim`

如果出现类似下面的报错：

```text
size mismatch for fc1.weight
```

说明当前命令创建的网络结构和 checkpoint 不一致。解决方法：

```powershell
# 方法 1：推理时使用训练时相同的结构参数
python main.py --mode test --hidden-width 32

# 方法 2：用当前默认参数重新训练并覆盖 checkpoint
python main.py --mode train
```

## 输出文件

| 文件 | 生成条件 | 说明 |
| --- | --- | --- |
| `checkpoints/agent_pursuer_actor_Gaussian` | 训练完成 | 追踪方 Actor 模型参数 |
| `checkpoints/agent_pursuer_critic` | 训练完成 | 追踪方 Critic 模型参数 |
| `outputs/train_reward.png` | 训练且未使用 `--no-plot` | 训练奖励曲线 |
| `outputs/trajectory.png` | 推理且未使用 `--no-plot` | 追踪星和逃逸星三维轨迹图 |
| `outputs/satellite_game.gif` | 推理且未使用 `--no-plot` | 带状态信息的卫星博弈动画 |
