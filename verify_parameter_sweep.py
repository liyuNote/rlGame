import argparse
import math

import numpy as np

from main import args_param, build_env, init_agent_dims
from satellite_function import EARTH_MU, GEO_ORBIT_RADIUS


def rollout(initial_distance, mode, max_steps, max_delta_v, d_capture):
    args = args_param(
        initial_distance=initial_distance,
        max_episode_steps=max_steps,
        max_delta_v=max_delta_v,
        opponent_policy="idle",
    )
    env = build_env(args, d_capture=d_capture)
    init_agent_dims(args, env)
    state = env.reset(0)
    distances = []
    rewards = []
    terminated = False
    truncated = False
    for step in range(1, max_steps + 1):
        if mode == "zero":
            action = np.zeros(3, dtype=np.float64)
        elif mode == "rule":
            direction = -np.asarray(state[0:3], dtype=np.float64)
            action = max_delta_v * direction / max(np.linalg.norm(direction), 1e-9)
        elif mode == "half_rule":
            direction = -np.asarray(state[0:3], dtype=np.float64)
            action = 0.5 * max_delta_v * direction / max(np.linalg.norm(direction), 1e-9)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        state, reward, terminated, truncated = env.step(action, np.zeros(3, dtype=np.float64), step)
        distances.append(float(env.dis))
        rewards.append(float(reward))
        if terminated or truncated:
            break

    distances = np.asarray(distances, dtype=np.float64)
    return {
        "steps": len(distances),
        "min_km": distances.min() / 1000.0,
        "final_km": distances[-1] / 1000.0,
        "max_km": distances.max() / 1000.0,
        "reward": sum(rewards),
        "fuel_left": env.fuel_c,
        "terminated": terminated,
        "truncated": truncated,
    }


def main():
    parser = argparse.ArgumentParser(description="Sweep pursuit/evasion parameter scales.")
    parser.add_argument("--distances-km", nargs="+", type=float, default=[10, 15, 20, 30, 40])
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-delta-v", type=float, default=0.2)
    parser.add_argument("--d-capture", type=float, default=2000.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    args = parser.parse_args()

    mean_motion = math.sqrt(EARTH_MU / (GEO_ORBIT_RADIUS**3))
    print(
        "dist_km init_vy_mps gamma_capture rule_steps rule_min_km "
        "rule_final_km rule_reward rule_fuel zero_final_km zero_max_km zero_reward "
        "half_steps half_final_km half_reward half_fuel"
    )
    for distance_km in args.distances_km:
        initial_distance = distance_km * 1000.0
        zero = rollout(initial_distance, "zero", args.max_steps, args.max_delta_v, args.d_capture)
        rule = rollout(initial_distance, "rule", args.max_steps, args.max_delta_v, args.d_capture)
        half = rollout(initial_distance, "half_rule", args.max_steps, args.max_delta_v, args.d_capture)
        discounted_capture = 500.0 * (args.gamma ** rule["steps"])
        print(
            f"{distance_km:.1f} "
            f"{-2.0 * mean_motion * initial_distance:.3f} "
            f"{discounted_capture:.2f} "
            f"{rule['steps']} {rule['min_km']:.3f} {rule['final_km']:.3f} "
            f"{rule['reward']:.1f} {rule['fuel_left']:.1f} "
            f"{zero['final_km']:.3f} {zero['max_km']:.3f} {zero['reward']:.1f} "
            f"{half['steps']} {half['final_km']:.3f} {half['reward']:.1f} {half['fuel_left']:.1f}"
        )


if __name__ == "__main__":
    main()
