import argparse
import copy

import numpy as np
import torch

from main import args_param, build_env, init_agent_dims, scale_state
from ppo_continuous import PPO_continuous, resolve_device


def sample_policy(args, env, samples, checkpoint=None):
    agent = PPO_continuous(args, "pursuer")
    if checkpoint:
        agent.load_checkpoint()
    state = scale_state(env.reset(0), args)
    state_tensor = torch.unsqueeze(torch.tensor(state, dtype=torch.float32, device=agent.device), 0)

    with torch.no_grad():
        dist = agent.actor.get_dist(state_tensor)
        mean = dist.mean.detach().cpu().numpy().reshape(-1)
        std = dist.stddev.detach().cpu().numpy().reshape(-1)
        raw = dist.sample((samples,)).squeeze(1)
        clipped = torch.clamp(raw, -args.max_action, args.max_action)
        raw_np = raw.cpu().numpy()
        clipped_np = clipped.cpu().numpy()

    saturated = np.isclose(np.abs(clipped_np), args.max_action, rtol=0.0, atol=1e-7)
    any_saturated = saturated.any(axis=1)
    changed = np.abs(raw_np - clipped_np) > 1e-9
    l1 = np.abs(clipped_np).sum(axis=1)

    title = "Loaded Gaussian policy distribution" if checkpoint else "Initial Gaussian policy distribution"
    print(f"\n{title}")
    print(f"  mean action: {mean}")
    print(f"  std action:  {std}")
    print(f"  max_action:  {args.max_action}")
    print(f"  per-axis saturation rate: {saturated.mean(axis=0)}")
    print(f"  any-axis saturation rate: {any_saturated.mean():.6f}")
    print(f"  sample changed by clamp rate: {changed.any(axis=1).mean():.6f}")
    print(f"  clipped action L1 mean/max: {l1.mean():.6f} / {l1.max():.6f} m/s")

    return agent


def rollout_with_agent(args, deterministic, max_steps, checkpoint=None):
    local_args = copy.copy(args)
    env = build_env(local_args, d_capture=2000.0)
    init_agent_dims(local_args, env)
    local_args.device = args.device
    agent = PPO_continuous(local_args, "pursuer")
    if checkpoint:
        agent.load_checkpoint()
    raw_state = env.reset(0)
    state = scale_state(raw_state, local_args)

    distances = []
    rewards = []
    actions = []
    saturated_steps = 0
    for step in range(1, max_steps + 1):
        if deterministic:
            action = agent.evaluate(state)
        else:
            action, _logprob = agent.choose_action(state)
        raw_state, reward, terminated, truncated = env.step(action, np.zeros(3), step)
        state = scale_state(raw_state, local_args)
        distances.append(float(env.dis))
        rewards.append(float(reward))
        actions.append(np.asarray(action, dtype=np.float64))
        saturated_steps += int(np.any(np.isclose(np.abs(action), local_args.max_action, rtol=0.0, atol=1e-7)))
        if terminated or truncated:
            break

    distances = np.asarray(distances, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    source = "loaded" if checkpoint else "untrained"
    label = "deterministic mean policy" if deterministic else "sampled clipped policy"
    print(f"\nRollout with {source} {label}")
    print(f"  steps: {len(distances)}")
    print(f"  distance km first/min/final/max: {distances[0] / 1000:.3f} / {distances.min() / 1000:.3f} / {distances[-1] / 1000:.3f} / {distances.max() / 1000:.3f}")
    print(f"  reward sum: {sum(rewards):.3f}")
    print(f"  fuel use proxy action L1 sum: {np.abs(actions).sum():.6f} m/s")
    print(f"  steps with any saturated axis: {saturated_steps} / {len(distances)}")
    print(f"  action mean: {actions.mean(axis=0)}")
    print(f"  action std:  {actions.std(axis=0)}")


def rollout_zero_action(args, max_steps):
    env = build_env(args, d_capture=2000.0)
    init_agent_dims(args, env)
    state = env.reset(0)
    distances = []
    rewards = []
    for step in range(1, max_steps + 1):
        state, reward, terminated, truncated = env.step(np.zeros(3), np.zeros(3), step)
        distances.append(float(env.dis))
        rewards.append(float(reward))
        if terminated or truncated:
            break
    distances = np.asarray(distances, dtype=np.float64)
    print("\nRollout with zero action")
    print(f"  steps: {len(distances)}")
    print(f"  distance km first/min/final/max: {distances[0] / 1000:.3f} / {distances.min() / 1000:.3f} / {distances[-1] / 1000:.3f} / {distances.max() / 1000:.3f}")
    print(f"  reward sum: {sum(rewards):.3f}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose PPO action sampling against the current satellite environment.")
    parser.add_argument("--initial-distance", type=float, default=20000.0)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--init-log-std", type=float, default=-2.0)
    parser.add_argument("--max-delta-v", type=float, default=0.2)
    parser.add_argument("--hidden-width", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint directory to load before diagnostics.")
    args_cli = parser.parse_args()

    args = args_param(
        max_episode_steps=args_cli.max_episode_steps,
        initial_distance=args_cli.initial_distance,
        max_delta_v=args_cli.max_delta_v,
        hidden_width=args_cli.hidden_width,
        init_log_std=args_cli.init_log_std,
        opponent_policy="idle",
        train_agent="pursuer",
        device=resolve_device(args_cli.device),
    )
    if args_cli.checkpoint:
        args.chkpt_dir = args_cli.checkpoint
    env = build_env(args, d_capture=2000.0)
    init_agent_dims(args, env)

    print("PPO action diagnostics")
    print(f"  initial_distance: {args.initial_distance} m")
    print(f"  max_episode_steps: {args.max_episode_steps}")
    print(f"  init_log_std: {args.init_log_std}")
    print(f"  max_delta_v: {args.max_delta_v} m/s")
    print(f"  device: {args.device}")

    if args_cli.checkpoint:
        print(f"  checkpoint: {args_cli.checkpoint}")

    sample_policy(args, env, args_cli.samples, checkpoint=args_cli.checkpoint)
    rollout_zero_action(args, args.max_episode_steps)
    rollout_with_agent(args, deterministic=True, max_steps=args.max_episode_steps, checkpoint=args_cli.checkpoint)
    rollout_with_agent(args, deterministic=False, max_steps=args.max_episode_steps, checkpoint=args_cli.checkpoint)


if __name__ == "__main__":
    main()
