from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

_DASHBOARD_SAVE_WARNING_SHOWN = False


def _prepare_output(path):
    if path is None:
        return None
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _save_figure_safely(fig, output, dpi=150, warn_only=False):
    """Save through a temporary file so live refresh does not crash on file locks."""
    global _DASHBOARD_SAVE_WARNING_SHOWN
    if output is None:
        return
    temp_output = output.with_name(f"{output.stem}.tmp{output.suffix}")
    try:
        fig.savefig(str(temp_output), dpi=dpi)
        temp_output.replace(output)
    except OSError as exc:
        if temp_output.exists():
            try:
                temp_output.unlink()
            except OSError:
                pass
        if warn_only:
            if not _DASHBOARD_SAVE_WARNING_SHOWN:
                print(f"Warning: dashboard image could not be saved to {output}: {exc}")
                _DASHBOARD_SAVE_WARNING_SHOWN = True
            return
        raise


def _save_animation_safely(animation, output, writer):
    """Save animation through a temporary file, then replace the target."""
    temp_output = output.with_name(f"{output.stem}.tmp{output.suffix}")
    try:
        animation.save(str(temp_output), writer=writer)
        temp_output.replace(output)
    except OSError:
        if temp_output.exists():
            try:
                temp_output.unlink()
            except OSError:
                pass
        raise


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
        _save_figure_safely(plt.gcf(), output, dpi=150)
    else:
        plt.show()
    plt.close()


def _moving_average(values, window=20):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    window = max(1, min(int(window), len(values)))
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def analyze_training(metrics, window=20):
    """Return short text observations for the live training dashboard."""
    rewards = metrics.get("episode_rewards", [])
    final_distances = metrics.get("final_distances", [])
    min_distances = metrics.get("min_distances", [])
    captures = metrics.get("captures", [])
    actor_losses = metrics.get("actor_losses", [])
    critic_losses = metrics.get("critic_losses", [])

    lines = ["Training analysis"]
    if rewards:
        recent_rewards = rewards[-window:]
        reward_mean = float(np.mean(recent_rewards))
        reward_delta = float(np.mean(recent_rewards[-max(1, len(recent_rewards) // 2):]) - np.mean(rewards[: min(len(rewards), window)]))
        lines.append(f"Recent reward mean: {reward_mean:.2f}")
        lines.append(f"Reward trend: {'up' if reward_delta >= 0 else 'down'} ({reward_delta:+.2f})")
    if final_distances:
        recent_final = float(np.mean(final_distances[-window:]))
        recent_min = float(np.mean(min_distances[-window:])) if min_distances else recent_final
        lines.append(f"Recent final distance: {recent_final:.2f}")
        lines.append(f"Recent closest distance: {recent_min:.2f}")
    if captures:
        capture_rate = float(np.mean(captures[-window:]))
        lines.append(f"Recent capture rate: {capture_rate:.1%}")
    if actor_losses:
        lines.append(f"Actor loss latest: {actor_losses[-1]:.4f}")
    if critic_losses:
        lines.append(f"Critic loss latest: {critic_losses[-1]:.4f}")

    if rewards and final_distances:
        if len(rewards) >= window and np.mean(rewards[-window:]) > np.mean(rewards[:window]):
            lines.append("Policy is earning more reward than early training.")
        if len(final_distances) >= window and np.mean(final_distances[-window:]) < np.mean(final_distances[:window]):
            lines.append("Pursuer is ending closer to the evader.")
        if captures and np.mean(captures[-window:]) > 0:
            lines.append("Captures appeared in the recent window.")
        elif min_distances and np.mean(min_distances[-window:]) < np.mean(final_distances[-window:]):
            lines.append("Pursuer gets closer during episodes but may not finish capture.")
    return "\n".join(lines)


def plot_training_dashboard(metrics, save_path="outputs/training_live_dashboard.png", window=20, live_window=False):
    """Save and optionally refresh a live dashboard window."""
    output = _prepare_output(save_path)
    episodes = np.arange(1, len(metrics.get("episode_rewards", [])) + 1)
    updates = np.arange(1, len(metrics.get("actor_losses", [])) + 1)

    if live_window:
        plt.ion()
        fig = plt.figure("Training Live Dashboard", figsize=(15, 8))
        fig.clf()
        axes = fig.subplots(2, 3)
    else:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ax_reward, ax_loss, ax_distance, ax_capture, ax_entropy, ax_text = axes.ravel()

    rewards = metrics.get("episode_rewards", [])
    mean_rewards = metrics.get("episode_mean_rewards", [])
    if rewards:
        ax_reward.plot(episodes, rewards, alpha=0.35, label="episode")
        ax_reward.plot(episodes, mean_rewards, linewidth=2, label="mean")
        ma = _moving_average(rewards, window)
        ax_reward.plot(
            episodes[-len(ma):],
            ma,
            linewidth=2,
            label=f"moving avg ({min(window, len(rewards))} episodes)",
        )
    ax_reward.set_title("Reward")
    ax_reward.set_xlabel("Episode")
    ax_reward.legend(loc="best")

    actor_losses = metrics.get("actor_losses", [])
    critic_losses = metrics.get("critic_losses", [])
    loss_spec = ax_loss.get_subplotspec()
    fig.delaxes(ax_loss)
    loss_grid = loss_spec.subgridspec(2, 1, hspace=0.55)
    actor_ax = fig.add_subplot(loss_grid[0])
    critic_ax = fig.add_subplot(loss_grid[1])
    if actor_losses:
        actor_ax.plot(updates, actor_losses, color="tab:blue")
    if critic_losses:
        critic_ax.plot(updates, critic_losses, color="tab:orange")
    actor_ax.set_title("Actor Loss", fontsize=10)
    critic_ax.set_title("Critic Loss", fontsize=10)
    critic_ax.set_xlabel("Update")
    actor_ax.tick_params(labelsize=8)
    critic_ax.tick_params(labelsize=8)
    actor_ax.grid(alpha=0.2)
    critic_ax.grid(alpha=0.2)

    final_distances = metrics.get("final_distances", [])
    min_distances = metrics.get("min_distances", [])
    if final_distances:
        ax_distance.plot(episodes, final_distances, label="final distance")
    if min_distances:
        ax_distance.plot(episodes, min_distances, label="closest distance")
    capture_distance = metrics.get("capture_distance")
    if capture_distance is not None:
        ax_distance.axhline(capture_distance, color="tab:red", linestyle="--", linewidth=1, label="capture radius")
    ax_distance.set_title("Pursuit Distance")
    ax_distance.set_xlabel("Episode")
    ax_distance.legend(loc="best")

    pursuer_path = np.asarray(metrics.get("latest_pursuer_path", []), dtype=float)
    evader_path = np.asarray(metrics.get("latest_evader_path", []), dtype=float)
    if len(pursuer_path) and len(evader_path):
        ax_capture.plot(pursuer_path[:, 0], pursuer_path[:, 1], color="tab:blue", label="pursuer")
        ax_capture.plot(evader_path[:, 0], evader_path[:, 1], color="tab:red", label="evader")
        ax_capture.scatter(pursuer_path[0, 0], pursuer_path[0, 1], color="tab:blue", marker="o", s=30)
        ax_capture.scatter(evader_path[0, 0], evader_path[0, 1], color="tab:red", marker="o", s=30)
        ax_capture.scatter(pursuer_path[-1, 0], pursuer_path[-1, 1], color="tab:blue", marker="x", s=45)
        ax_capture.scatter(evader_path[-1, 0], evader_path[-1, 1], color="tab:red", marker="x", s=45)
        ax_capture.plot(
            [pursuer_path[-1, 0], evader_path[-1, 0]],
            [pursuer_path[-1, 1], evader_path[-1, 1]],
            color="0.4",
            linestyle="--",
            linewidth=1,
        )
    xy_xlim = metrics.get("xy_xlim")
    xy_ylim = metrics.get("xy_ylim")
    if xy_xlim is not None and xy_ylim is not None:
        ax_capture.set_xlim(xy_xlim)
        ax_capture.set_ylim(xy_ylim)
        ax_capture.set_aspect("equal", adjustable="box")
    elif len(pursuer_path) and len(evader_path):
        ax_capture.axis("equal")
    ax_capture.set_title("Latest Pursuit-Evasion XY")
    ax_capture.set_xlabel("X")
    ax_capture.set_ylabel("Y")
    ax_capture.legend(loc="best")

    entropy = metrics.get("entropies", [])
    ratios = metrics.get("ratios", [])
    loss_updates = np.arange(1, len(entropy) + 1)
    if entropy:
        ax_entropy.plot(loss_updates, entropy, label="entropy")
    if ratios:
        ratio_ax = ax_entropy.twinx()
        ratio_ax.plot(loss_updates, ratios, color="tab:purple", alpha=0.55, label="ratio")
        ratio_ax.axhline(1.0, color="tab:purple", linestyle="--", linewidth=1)
        ratio_ax.set_ylabel("Ratio")
    ax_entropy.set_title("Exploration / PPO Ratio")
    ax_entropy.set_xlabel("Update")
    ax_entropy.legend(loc="best")

    ax_text.axis("off")
    ax_text.text(
        0.02,
        0.98,
        analyze_training(metrics, window=window),
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
        transform=ax_text.transAxes,
    )

    fig.tight_layout()
    if output:
        _save_figure_safely(fig, output, dpi=150, warn_only=live_window)
    if live_window:
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.001)
    elif not output:
        plt.show()
    if not live_window:
        plt.close(fig)
    return output


def keep_live_dashboard_open():
    """Keep the final live dashboard visible until the user closes it."""
    plt.ioff()
    plt.show()


def _xy_limits_for_paths(*paths, capture_distance=None):
    """Return padded XY limits that include all paths and optional capture circles."""
    valid_paths = [np.asarray(path, dtype=float)[:, :2] for path in paths if len(path)]
    if not valid_paths:
        return (-1.0, 1.0), (-1.0, 1.0)

    xy = np.vstack(valid_paths)
    min_xy = xy.min(axis=0)
    max_xy = xy.max(axis=0)

    if capture_distance is not None:
        capture_radius = float(capture_distance)
        if capture_radius > 0:
            min_xy -= capture_radius
            max_xy += capture_radius

    center = 0.5 * (min_xy + max_xy)
    span = max_xy - min_xy
    radius = max(float(span.max()) * 0.55, 1.0)
    margin = max(radius * 0.08, 1.0)
    radius += margin
    return (center[0] - radius, center[0] + radius), (center[1] - radius, center[1] + radius)


def plot_trajectory(pursuer_position, escaper_position, save_path="outputs/trajectory.png", capture_distance=None):
    pursuer_position = np.asarray(pursuer_position)
    escaper_position = np.asarray(escaper_position)
    output = _prepare_output(save_path)
    fig, ax = plt.subplots(figsize=(7, 6))
    if len(pursuer_position):
        ax.plot(pursuer_position[:, 0], pursuer_position[:, 1], color="tab:blue", label="pursuer")
        ax.scatter(pursuer_position[0, 0], pursuer_position[0, 1], color="tab:blue", marker="o", s=30)
        ax.scatter(pursuer_position[-1, 0], pursuer_position[-1, 1], color="tab:blue", marker="x", s=45)
    if len(escaper_position):
        ax.plot(escaper_position[:, 0], escaper_position[:, 1], color="tab:red", label="evader")
        ax.scatter(escaper_position[0, 0], escaper_position[0, 1], color="tab:red", marker="o", s=30)
        ax.scatter(escaper_position[-1, 0], escaper_position[-1, 1], color="tab:red", marker="x", s=45)
        if capture_distance is not None:
            capture_zone = plt.Circle(
                (escaper_position[-1, 0], escaper_position[-1, 1]),
                capture_distance,
                color="tab:red",
                fill=False,
                linestyle="--",
                linewidth=1,
                alpha=0.6,
                label="capture radius",
            )
            ax.add_patch(capture_zone)
    if len(pursuer_position) and len(escaper_position):
        ax.plot(
            [pursuer_position[-1, 0], escaper_position[-1, 0]],
            [pursuer_position[-1, 1], escaper_position[-1, 1]],
            color="0.4",
            linestyle="--",
            linewidth=1,
            label="final distance",
        )
    ax.set_title("Pursuit-Evasion XY Trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    xlim, ylim = _xy_limits_for_paths(pursuer_position, escaper_position, capture_distance=capture_distance)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    plt.tight_layout()
    if output:
        _save_figure_safely(plt.gcf(), output, dpi=150)
    else:
        plt.show()
    plt.close()


def animate_satellite_game(history, save_path="outputs/satellite_game.gif", fps=8, show_window=False):
    """Generate a 2D XY pursuit/evasion animation with per-step state text."""
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
    pursuer_fuel = np.asarray(history.get("pursuer_fuel", []), dtype=float)
    evader_fuel = np.asarray(history.get("evader_fuel", []), dtype=float)
    capture_distance = history.get("capture_distance")

    if len(pursuer_pos) == 0:
        raise ValueError("No inference states were recorded; cannot create animation.")

    xlim, ylim = _xy_limits_for_paths(pursuer_pos, evader_pos, capture_distance=capture_distance)

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(121)
    info_ax = fig.add_subplot(122)
    info_ax.axis("off")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Satellite Pursuit-Evasion XY Inference")

    pursuer_line, = ax.plot([], [], color="#1f77b4", linewidth=2, label="Pursuer path")
    evader_line, = ax.plot([], [], color="#d62728", linewidth=2, label="Evader path")
    link_line, = ax.plot([], [], color="#555555", linestyle="--", linewidth=1, label="Distance")
    pursuer_dot, = ax.plot([], [], "o", color="#1f77b4", markersize=8, label="Pursuer")
    evader_dot, = ax.plot([], [], "o", color="#d62728", markersize=8, label="Evader")
    capture_zone = None
    if capture_distance is not None:
        capture_zone = plt.Circle(
            (evader_pos[0, 0], evader_pos[0, 1]),
            capture_distance,
            color="#d62728",
            fill=False,
            linestyle="--",
            linewidth=1,
            alpha=0.55,
            label="Capture radius",
        )
        ax.add_patch(capture_zone)
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
        evader_line.set_data(evader_pos[: i + 1, 0], evader_pos[: i + 1, 1])

        pursuer_dot.set_data([pursuer_pos[i, 0]], [pursuer_pos[i, 1]])
        evader_dot.set_data([evader_pos[i, 0]], [evader_pos[i, 1]])
        link_line.set_data([pursuer_pos[i, 0], evader_pos[i, 0]], [pursuer_pos[i, 1], evader_pos[i, 1]])
        if capture_zone is not None:
            capture_zone.center = (evader_pos[i, 0], evader_pos[i, 1])

        info_text.set_text(
            "\n".join(
                [
                    "Inference State",
                    f"Step: {i + 1}/{len(pursuer_pos)}",
                    f"3D distance: {distances[i]:.3f}",
                    f"Capture radius: {capture_distance:.3f}" if capture_distance is not None else "Capture radius: n/a",
                    f"Reward: {rewards[i]:.3f}",
                    f"Cumulative reward: {cumulative_rewards[i]:.3f}",
                    f"Pursuer fuel: {pursuer_fuel[i]:.3f}" if len(pursuer_fuel) else "Pursuer fuel: n/a",
                    f"Evader fuel: {evader_fuel[i]:.3f}" if len(evader_fuel) else "Evader fuel: n/a",
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
        artists = [pursuer_line, evader_line, link_line, pursuer_dot, evader_dot, info_text]
        if capture_zone is not None:
            artists.append(capture_zone)
        return artists

    animation = FuncAnimation(fig, update, frames=len(pursuer_pos), interval=1000 / fps, blit=False)
    if output:
        _save_animation_safely(animation, output, writer=PillowWriter(fps=fps))
    if show_window:
        plt.show()
    plt.close(fig)
    return output
