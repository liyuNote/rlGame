from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


def _prepare_output(path):
    if path is None:
        return None
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def plot_train_reward(episode_rewards, episode_mean_rewards, save_path="outputs/train_reward.png"):
    output = _prepare_output(save_path)
    plt.figure(figsize=(9, 5))
    plt.plot(episode_rewards, label="episode reward", alpha=0.65)
    plt.plot(episode_mean_rewards, label="mean reward", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    if output:
        plt.savefig(output, dpi=150)
    else:
        plt.show()
    plt.close()


def plot_trajectory(pursuer_position, escaper_position, save_path="outputs/trajectory.png"):
    pursuer_position = np.asarray(pursuer_position)
    escaper_position = np.asarray(escaper_position)
    output = _prepare_output(save_path)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    if len(pursuer_position):
        ax.plot(pursuer_position[:, 0], pursuer_position[:, 1], pursuer_position[:, 2], label="pursuer")
    if len(escaper_position):
        ax.plot(escaper_position[:, 0], escaper_position[:, 1], escaper_position[:, 2], label="evader")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    plt.tight_layout()
    if output:
        plt.savefig(output, dpi=150)
    else:
        plt.show()
    plt.close()


def animate_satellite_game(history, save_path="outputs/satellite_game.gif", fps=8):
    """Generate a 3D pursuit/evasion animation with per-step state text.

    history is a dict produced by test_network(). It contains positions, velocities,
    actions, rewards and distances for every inference step.
    """
    output = _prepare_output(save_path)
    pursuer_pos = np.asarray(history["pursuer_position"], dtype=float)
    evader_pos = np.asarray(history["evader_position"], dtype=float)
    pursuer_vel = np.asarray(history["pursuer_velocity"], dtype=float)
    evader_vel = np.asarray(history["evader_velocity"], dtype=float)
    pursuer_action = np.asarray(history["pursuer_action"], dtype=float)
    evader_action = np.asarray(history["evader_action"], dtype=float)
    rewards = np.asarray(history["reward"], dtype=float)
    distances = np.asarray(history["distance"], dtype=float)
    cumulative_rewards = np.asarray(history["cumulative_reward"], dtype=float)

    if len(pursuer_pos) == 0:
        raise ValueError("No inference states were recorded; cannot create animation.")

    all_pos = np.vstack([pursuer_pos, evader_pos])
    center = all_pos.mean(axis=0)
    span = np.ptp(all_pos, axis=0)
    radius = max(float(span.max()) * 0.6, 1.0)

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(121, projection="3d")
    info_ax = fig.add_subplot(122)
    info_ax.axis("off")

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Satellite Pursuit-Evasion Inference")

    pursuer_line, = ax.plot([], [], [], color="#1f77b4", linewidth=2, label="Pursuer path")
    evader_line, = ax.plot([], [], [], color="#d62728", linewidth=2, label="Evader path")
    link_line, = ax.plot([], [], [], color="#555555", linestyle="--", linewidth=1, label="Distance")
    pursuer_dot, = ax.plot([], [], [], "o", color="#1f77b4", markersize=8, label="Pursuer")
    evader_dot, = ax.plot([], [], [], "o", color="#d62728", markersize=8, label="Evader")
    ax.legend(loc="upper left")

    info_text = info_ax.text(
        0.02,
        0.98,
        "",
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
        transform=info_ax.transAxes,
    )

    def fmt_vec(name, vec):
        return f"{name}: [{vec[0]:9.2f}, {vec[1]:9.2f}, {vec[2]:9.2f}]"

    def update(frame):
        i = frame
        pursuer_line.set_data(pursuer_pos[: i + 1, 0], pursuer_pos[: i + 1, 1])
        pursuer_line.set_3d_properties(pursuer_pos[: i + 1, 2])
        evader_line.set_data(evader_pos[: i + 1, 0], evader_pos[: i + 1, 1])
        evader_line.set_3d_properties(evader_pos[: i + 1, 2])

        pursuer_dot.set_data([pursuer_pos[i, 0]], [pursuer_pos[i, 1]])
        pursuer_dot.set_3d_properties([pursuer_pos[i, 2]])
        evader_dot.set_data([evader_pos[i, 0]], [evader_pos[i, 1]])
        evader_dot.set_3d_properties([evader_pos[i, 2]])
        link_line.set_data([pursuer_pos[i, 0], evader_pos[i, 0]], [pursuer_pos[i, 1], evader_pos[i, 1]])
        link_line.set_3d_properties([pursuer_pos[i, 2], evader_pos[i, 2]])

        info_text.set_text(
            "\n".join(
                [
                    "Inference State",
                    f"Step: {i + 1}/{len(pursuer_pos)}",
                    f"Distance: {distances[i]:.3f}",
                    f"Reward: {rewards[i]:.3f}",
                    f"Cumulative reward: {cumulative_rewards[i]:.3f}",
                    "",
                    "Pursuer",
                    fmt_vec("pos", pursuer_pos[i]),
                    fmt_vec("vel", pursuer_vel[i]),
                    fmt_vec("act", pursuer_action[i]),
                    "",
                    "Evader",
                    fmt_vec("pos", evader_pos[i]),
                    fmt_vec("vel", evader_vel[i]),
                    fmt_vec("act", evader_action[i]),
                ]
            )
        )
        return pursuer_line, evader_line, link_line, pursuer_dot, evader_dot, info_text

    animation = FuncAnimation(fig, update, frames=len(pursuer_pos), interval=1000 / fps, blit=False)
    animation.save(output, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return output
