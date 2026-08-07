import argparse
import os
import sys
from pathlib import Path

# Keep matplotlib cache inside the project so Windows user-directory permissions
# do not affect command-line rendering.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib
import numpy as np
from tqdm import tqdm

# Use a GUI backend only when live pop-up rendering is requested.
if "--live-dashboard-window" in sys.argv or "--inference-window" in sys.argv:
    matplotlib.use("TkAgg")
else:
    matplotlib.use("Agg")

from env import satellites
from ppo_continuous import PPO_continuous, get_device_report, resolve_device
from replaybuffer import ReplayBuffer
import plot_function as pf


class args_param:
    """Lightweight configuration object shared by env, buffer and PPO agents."""

    def __init__(
        self,
        max_train_steps=int(3e6),
        evaluate_freq=5e3,
        save_freq=20,
        policy_dist="Gaussian",
        batch_size=2048,
        mini_batch_size=64,
        hidden_width=256,
        hidden_width2=128,
        lr_a=0.0002,
        lr_c=0.0002,
        gamma=0.99,
        lamda=0.95,
        epsilon=0.1,
        K_epochs=10,
        max_episode_steps=500,
        use_adv_norm=True,
        use_state_norm=True,
        use_reward_norm=False,
        use_reward_scaling=True,
        entropy_coef=0.01,
        use_lr_decay=True,
        use_grad_clip=True,
        use_orthogonal_init=True,
        set_adam_eps=True,
        use_tanh=True,
        chkpt_dir="checkpoints",
        env_dt=100.0,
        device="auto",
        live_dashboard_freq=5,
        live_dashboard_output="outputs/training_live_dashboard.png",
        live_dashboard_window=False,
        state_position_scale=200000.0,
        state_velocity_scale=10000.0,
        train_agent="pursuer",
        opponent_policy="fixed",
        fixed_rule_action_scale=0.8,
        initial_distance=80000.0,
        dashboard_xy_margin=None,
    ):
        # Training loop controls. Despite the historical name, max_train_steps is
        # used as the number of training episodes in this project.
        self.max_train_steps = max_train_steps
        self.evaluate_freq = evaluate_freq
        self.save_freq = save_freq

        # Policy distribution: Gaussian directly outputs bounded continuous
        # actions; Beta outputs [0, 1] and is later mapped to action bounds.
        self.policy_dist = policy_dist

        # PPO collects batch_size transitions, then optimizes random mini-batches.
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size

        # Shared MLP width for actor and critic.
        self.hidden_width = hidden_width
        self.hidden_width2 = hidden_width2

        # Actor/critic optimization parameters.
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.gamma = gamma
        self.lamda = lamda
        self.epsilon = epsilon
        self.K_epochs = K_epochs

        # Common PPO stabilization switches.
        self.use_adv_norm = use_adv_norm
        self.use_state_norm = use_state_norm
        self.use_reward_norm = use_reward_norm
        self.use_reward_scaling = use_reward_scaling
        self.entropy_coef = entropy_coef
        self.use_lr_decay = use_lr_decay
        self.use_grad_clip = use_grad_clip
        self.use_orthogonal_init = use_orthogonal_init
        self.set_adam_eps = set_adam_eps
        self.use_tanh = use_tanh

        # Environment and output settings.
        self.max_episode_steps = max_episode_steps
        self.chkpt_dir = chkpt_dir
        self.env_dt = env_dt
        self.device = device
        self.live_dashboard_freq = live_dashboard_freq
        self.live_dashboard_output = live_dashboard_output
        self.live_dashboard_window = live_dashboard_window
        self.state_position_scale = state_position_scale
        self.state_velocity_scale = state_velocity_scale
        self.train_agent = train_agent
        self.opponent_policy = opponent_policy
        self.fixed_rule_action_scale = fixed_rule_action_scale
        self.initial_distance = initial_distance
        self.dashboard_xy_margin = dashboard_xy_margin

    def print_information(self):
        print("Maximum number of training episodes:", self.max_train_steps)
        print("Policy distribution:", self.policy_dist)
        print("Batch size:", self.batch_size)
        print("Minibatch size:", self.mini_batch_size)
        print("Hidden width:", self.hidden_width)
        print("Actor lr:", self.lr_a)
        print("Critic lr:", self.lr_c)
        print("Gamma:", self.gamma)
        print("GAE lambda:", self.lamda)
        print("PPO clip epsilon:", self.epsilon)
        print("PPO epochs:", self.K_epochs)
        print("Max episode steps:", self.max_episode_steps)
        print("Checkpoint dir:", self.chkpt_dir)
        print("Device:", self.device)
        print("Train agent:", self.train_agent)
        print("Opponent policy:", self.opponent_policy)
        print("State position scale:", self.state_position_scale)
        print("State velocity scale:", self.state_velocity_scale)
        print("Initial distance:", self.initial_distance)
        print("Dashboard XY margin:", self.dashboard_xy_margin)


def build_env(args, d_capture):
    """Build the default pursuit/evasion scenario."""
    return satellites(
        Pursuer_position=np.array([args.initial_distance, 0.0, 0.0]),
        Pursuer_vector=np.array([0.0, 0.0, 0.0]),
        Escaper_position=np.array([0.0, 0.0, 0.0]),
        Escaper_vector=np.array([0.0, 0.0, 0.0]),
        d_capture=d_capture,
        args=args,
    )


def init_agent_dims(args, env):
    """Fill dimensions that are only known after the environment is created."""
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = float(env.action_space[0][1])


def scale_state(state, args):
    """Scale raw environment state before feeding it to PPO networks."""
    if not args.use_state_norm:
        return np.asarray(state, dtype=np.float32)
    state = np.asarray(state, dtype=np.float32).copy()
    state[[0, 1, 2, 6, 7, 8, 12, 13, 14]] /= args.state_position_scale
    state[[3, 4, 5, 9, 10, 11, 15, 16, 17]] /= args.state_velocity_scale
    return state


def policy_to_env_action(args, policy_action):
    """Map policy distribution output to environment action units."""
    if args.policy_dist == "Beta":
        return 2 * (policy_action - 0.5) * args.max_action
    return policy_action


def choose_learning_action(args, agent, policy_state, deterministic=False):
    """Return an environment action and optional log probability from a PPO agent."""
    if deterministic:
        policy_action = agent.evaluate(policy_state)
        return policy_to_env_action(args, policy_action), None
    policy_action, logprob = agent.choose_action(policy_state)
    return policy_to_env_action(args, policy_action), logprob


def fixed_rule_action(args, raw_state, role):
    """Simple fixed opponent rule: pursuer chases, evader runs away."""
    rel_pos = np.asarray(raw_state[0:3], dtype=np.float64)
    direction = -rel_pos
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return np.zeros(args.action_dim, dtype=np.float32)
    action = args.fixed_rule_action_scale * args.max_action * direction / norm
    return np.asarray(action, dtype=np.float32)


def choose_opponent_action(args, raw_state, role):
    """Return the configured non-learning opponent action."""
    if args.opponent_policy == "idle":
        return np.zeros(args.action_dim, dtype=np.float32)
    return fixed_rule_action(args, raw_state, role)


def uses_learning_policy(args, role):
    """Whether this role should act through its PPO policy in the current run."""
    if args.train_agent in (role, "both"):
        return True
    return args.opponent_policy == "learning"


def dashboard_xy_limits(args, d_capture):
    """Fixed XY limits for the live pursuit/evasion panel."""
    margin = args.dashboard_xy_margin
    if margin is None:
        margin = max(2.5 * args.initial_distance, 8.0 * d_capture)
    center_x = 0.5 * args.initial_distance
    return (center_x - margin, center_x + margin), (-margin, margin)


def train_network(args, env, show_picture=True, pre_train=False, d_capture=15000.0):
    """Train the pursuer PPO agent.

    The evader agent is instantiated and produces actions, but it is not updated
    in the current single-agent training loop. It acts as an untrained opponent.
    """
    episode_rewards = []
    episode_mean_rewards = []
    train_metrics = {
        "episode_rewards": episode_rewards,
        "episode_mean_rewards": episode_mean_rewards,
        "episode_steps": [],
        "final_distances": [],
        "min_distances": [],
        "captures": [],
        "actor_losses": [],
        "critic_losses": [],
        "entropies": [],
        "ratios": [],
        "lr_actor": [],
        "lr_critic": [],
        "capture_distance": d_capture,
        "latest_pursuer_path": [],
        "latest_evader_path": [],
        "xy_xlim": dashboard_xy_limits(args, d_capture)[0],
        "xy_ylim": dashboard_xy_limits(args, d_capture)[1],
    }
    env.d_capture = d_capture
    init_agent_dims(args, env)
    args.device = resolve_device(args.device)
    print(get_device_report(args.device))

    train_pursuer = args.train_agent in ("pursuer", "both")
    train_evader = args.train_agent in ("evader", "both")
    pursuer_buffer = ReplayBuffer(args) if train_pursuer else None
    evader_buffer = ReplayBuffer(args) if train_evader else None
    pursuer_agent = PPO_continuous(args, "pursuer")
    evader_agent = PPO_continuous(args, "evader")
    if pre_train:
        if uses_learning_policy(args, "pursuer"):
            pursuer_agent.load_checkpoint()
        if uses_learning_policy(args, "evader"):
            evader_agent.load_checkpoint()

    pbar = tqdm(range(args.max_train_steps), desc=f"Training {args.train_agent}", unit="episode")
    for episode in pbar:
        episode_pursuer_reward = 0.0
        episode_evader_reward = 0.0
        episode_count = 0
        episode_distances = []
        episode_pursuer_path = []
        episode_evader_path = []
        raw_s = env.reset(0)
        s = scale_state(raw_s, args)

        while True:
            episode_count += 1

            if uses_learning_policy(args, "pursuer"):
                pursuer_action, pursuer_a_logprob = choose_learning_action(args, pursuer_agent, s)
            else:
                pursuer_action = choose_opponent_action(args, raw_s, "pursuer")
                pursuer_a_logprob = None

            if uses_learning_policy(args, "evader"):
                evader_action, evader_a_logprob = choose_learning_action(args, evader_agent, s)
            else:
                evader_action = choose_opponent_action(args, raw_s, "evader")
                evader_a_logprob = None

            raw_s_, r, terminated, truncated = env.step(
                pursuer_action,
                evader_action,
                episode_count,
            )
            s_ = scale_state(raw_s_, args)
            evader_reward = env.last_evader_reward
            episode_pursuer_reward += r
            episode_evader_reward += evader_reward
            episode_distances.append(float(env.dis))
            episode_pursuer_path.append(np.asarray(raw_s_[6:9], dtype=float))
            episode_evader_path.append(np.asarray(raw_s_[12:15], dtype=float))

            episode_done = terminated or truncated
            if train_pursuer:
                pursuer_buffer.store(
                    s,
                    pursuer_action,
                    pursuer_a_logprob,
                    r,
                    s_,
                    terminated,
                    episode_done,
                )
            if train_evader:
                evader_buffer.store(
                    s,
                    evader_action,
                    evader_a_logprob,
                    evader_reward,
                    s_,
                    terminated,
                    episode_done,
                )
            s = s_
            raw_s = raw_s_

            update_infos = []
            if train_pursuer and pursuer_buffer.count == args.batch_size:
                update_infos.append(pursuer_agent.update(pursuer_buffer, episode))
                pursuer_buffer.count = 0
            if train_evader and evader_buffer.count == args.batch_size:
                update_infos.append(evader_agent.update(evader_buffer, episode))
                evader_buffer.count = 0
            if update_infos:
                train_metrics["actor_losses"].append(float(np.mean([info["actor_loss"] for info in update_infos])))
                train_metrics["critic_losses"].append(float(np.mean([info["critic_loss"] for info in update_infos])))
                train_metrics["entropies"].append(float(np.mean([info["entropy"] for info in update_infos])))
                train_metrics["ratios"].append(float(np.mean([info["ratio"] for info in update_infos])))
                train_metrics["lr_actor"].append(float(np.mean([info["lr_actor"] for info in update_infos])))
                train_metrics["lr_critic"].append(float(np.mean([info["lr_critic"] for info in update_infos])))

            if episode_done:
                if args.train_agent == "evader":
                    episode_reward = episode_evader_reward
                elif args.train_agent == "both":
                    episode_reward = 0.5 * (episode_pursuer_reward + episode_evader_reward)
                else:
                    episode_reward = episode_pursuer_reward
                episode_rewards.append(episode_reward)
                episode_mean_rewards.append(float(np.mean(episode_rewards)))
                train_metrics["episode_steps"].append(episode_count)
                train_metrics["final_distances"].append(float(env.dis))
                train_metrics["min_distances"].append(float(np.min(episode_distances)) if episode_distances else float(env.dis))
                train_metrics["captures"].append(float(terminated and env.dis <= env.d_capture))
                train_metrics["latest_pursuer_path"] = episode_pursuer_path
                train_metrics["latest_evader_path"] = episode_evader_path
                should_refresh_dashboard = (
                    show_picture
                    and args.live_dashboard_freq > 0
                    and ((episode + 1) % args.live_dashboard_freq == 0 or episode + 1 == args.max_train_steps)
                )
                if should_refresh_dashboard:
                    pf.plot_training_dashboard(
                        train_metrics,
                        save_path=args.live_dashboard_output,
                        window=max(5, min(50, args.live_dashboard_freq * 2)),
                        live_window=args.live_dashboard_window,
                    )
                pbar.set_postfix(
                    {
                        "episode": episode,
                        "reward": f"{episode_reward:.1f}",
                        "mean_reward": f"{episode_mean_rewards[-1]:.1f}",
                        "distance": f"{env.dis:.1f}",
                    }
                )
                break

    if train_pursuer:
        pursuer_agent.save_checkpoint()
    if train_evader:
        evader_agent.save_checkpoint()
    if show_picture:
        pf.plot_train_reward(episode_rewards, episode_mean_rewards)
        pf.plot_training_dashboard(
            train_metrics,
            save_path=args.live_dashboard_output,
            live_window=args.live_dashboard_window,
        )
        print(f"Live training dashboard saved to: {args.live_dashboard_output}")
        if args.live_dashboard_window:
            print("Close the dashboard window to finish.")
            pf.keep_live_dashboard_open()
    return pursuer_agent


def test_network(
    args,
    env,
    show_pictures=True,
    d_capture=20000.0,
    max_steps=None,
    animation_path="outputs/satellite_game.gif",
    animation_fps=8,
    inference_window=False,
):
    """Run deterministic inference and optionally render trajectory + animation."""
    episode_reward = 0.0
    episode_count = 0
    pursuer_position = []
    escaper_position = []

    # history drives the animation: every rendered frame corresponds to one
    # inference step and displays positions, velocities, actions and rewards.
    history = {
        "pursuer_position": [],
        "evader_position": [],
        "pursuer_velocity": [],
        "evader_velocity": [],
        "pursuer_action": [],
        "evader_action": [],
        "reward": [],
        "cumulative_reward": [],
        "distance": [],
        "pursuer_fuel": [],
        "evader_fuel": [],
        "capture_distance": d_capture,
    }

    env.d_capture = d_capture
    init_agent_dims(args, env)
    args.device = resolve_device(args.device)
    print(get_device_report(args.device))
    pursuer_agent = PPO_continuous(args, "pursuer")
    evader_agent = PPO_continuous(args, "evader")
    if uses_learning_policy(args, "pursuer"):
        pursuer_agent.load_checkpoint()
    if uses_learning_policy(args, "evader"):
        evader_agent.load_checkpoint()

    raw_s = env.reset(2)
    s = scale_state(raw_s, args)
    max_steps = max_steps or args.max_episode_steps
    while episode_count < max_steps:
        episode_count += 1

        # evaluate uses the policy mean, so the same checkpoint and initial
        # state produce a stable inference trajectory.
        if uses_learning_policy(args, "pursuer"):
            pursuer_action, _pursuer_logprob = choose_learning_action(args, pursuer_agent, s, deterministic=True)
        else:
            pursuer_action = choose_opponent_action(args, raw_s, "pursuer")

        if uses_learning_policy(args, "evader"):
            evader_action, _evader_logprob = choose_learning_action(args, evader_agent, s, deterministic=True)
        else:
            evader_action = choose_opponent_action(args, raw_s, "evader")

        raw_s, r, terminated, truncated = env.step(
            pursuer_action,
            evader_action,
            episode_count,
        )
        s = scale_state(raw_s, args)
        episode_done = terminated or truncated
        episode_reward += r

        pursuer_position.append(raw_s[6:9])
        escaper_position.append(raw_s[12:15])
        history["pursuer_position"].append(raw_s[6:9])
        history["evader_position"].append(raw_s[12:15])
        history["pursuer_velocity"].append(raw_s[9:12])
        history["evader_velocity"].append(raw_s[15:18])
        history["pursuer_action"].append(np.asarray(pursuer_action, dtype=np.float32))
        history["evader_action"].append(np.asarray(evader_action, dtype=np.float32))
        history["reward"].append(float(r))
        history["cumulative_reward"].append(float(episode_reward))
        history["distance"].append(float(env.dis))
        history["pursuer_fuel"].append(float(env.fuel_c))
        history["evader_fuel"].append(float(env.fuel_t))
        if episode_done:
            break

    print(
        f"Test reward: {episode_reward:.3f}, steps: {episode_count}, "
        f"distance: {env.dis:.3f}, pursuer fuel: {env.fuel_c:.3f}, evader fuel: {env.fuel_t:.3f}"
    )
    if show_pictures:
        pf.plot_trajectory(pursuer_position, escaper_position, capture_distance=d_capture)
        animation_file = pf.animate_satellite_game(
            history,
            save_path=animation_path,
            fps=animation_fps,
            show_window=inference_window,
        )
        print(f"Animation saved to: {animation_file}")
    return episode_reward


def parse_args():
    """Command-line arguments for training, inference and rendering."""
    parser = argparse.ArgumentParser(description="PPO satellite pursuit/evasion")
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--max-train-steps", type=int, default=1000)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--mini-batch-size", type=int, default=32)
    parser.add_argument("--k-epochs", type=int, default=3)
    parser.add_argument("--hidden-width", type=int, default=128)
    parser.add_argument("--policy-dist", choices=["Gaussian", "Beta"], default="Gaussian")
    parser.add_argument("--chkpt-dir", default="checkpoints")
    parser.add_argument("--d-capture", type=float, default=20000.0)
    parser.add_argument("--pre-train", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--train-agent",
        choices=["pursuer", "evader", "both"],
        default="pursuer",
        help="Which agent to train. both enables self-play with separate replay buffers.",
    )
    parser.add_argument(
        "--opponent-policy",
        choices=["fixed", "learning", "idle"],
        default="fixed",
        help="Policy for any non-trained opponent: fixed rule, learning PPO checkpoint/current policy, or idle zero-control drift.",
    )
    parser.add_argument(
        "--fixed-rule-action-scale",
        type=float,
        default=0.8,
        help="Fraction of max_action used by the fixed chase/escape rule.",
    )
    parser.add_argument("--state-position-scale", type=float, default=200000.0)
    parser.add_argument("--state-velocity-scale", type=float, default=10000.0)
    parser.add_argument("--no-state-norm", action="store_true")
    parser.add_argument(
        "--initial-distance",
        type=float,
        default=80000.0,
        help="Initial X-axis distance between pursuer and evader. Old scenario was about 182000.",
    )
    parser.add_argument(
        "--dashboard-xy-margin",
        type=float,
        default=None,
        help="Fixed half-width of the pursuit/evasion XY dashboard. Default is max(2.5*initial_distance, 8*d_capture).",
    )
    parser.add_argument("--animation-output", default="outputs/satellite_game.gif")
    parser.add_argument("--animation-fps", type=int, default=8)
    parser.add_argument(
        "--inference-window",
        action="store_true",
        help="Show a pop-up animation window during inference. Close the window to finish.",
    )
    parser.add_argument(
        "--live-dashboard-freq",
        type=int,
        default=5,
        help="Refresh training dashboard every N episodes. Use 0 to disable live dashboard refresh.",
    )
    parser.add_argument("--live-dashboard-output", default="outputs/training_live_dashboard.png")
    parser.add_argument(
        "--live-dashboard-window",
        action="store_true",
        help="Show a live pop-up dashboard window during training.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device to use: auto, cpu, cuda, cuda:0, mps or xpu. Default: auto.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli = parse_args()
    Path(cli.chkpt_dir).mkdir(parents=True, exist_ok=True)
    args = args_param(
        max_episode_steps=cli.max_episode_steps,
        batch_size=cli.batch_size,
        mini_batch_size=cli.mini_batch_size,
        max_train_steps=cli.max_train_steps,
        K_epochs=cli.k_epochs,
        hidden_width=cli.hidden_width,
        policy_dist=cli.policy_dist,
        chkpt_dir=cli.chkpt_dir,
        device=cli.device,
        live_dashboard_freq=cli.live_dashboard_freq,
        live_dashboard_output=cli.live_dashboard_output,
        live_dashboard_window=cli.live_dashboard_window,
        train_agent=cli.train_agent,
        opponent_policy=cli.opponent_policy,
        fixed_rule_action_scale=cli.fixed_rule_action_scale,
        state_position_scale=cli.state_position_scale,
        state_velocity_scale=cli.state_velocity_scale,
        use_state_norm=not cli.no_state_norm,
        initial_distance=cli.initial_distance,
        dashboard_xy_margin=cli.dashboard_xy_margin,
    )
    env = build_env(args, d_capture=cli.d_capture)
    if cli.mode == "train":
        train_network(args, env, show_picture=not cli.no_plot, pre_train=cli.pre_train, d_capture=cli.d_capture)
    else:
        test_network(
            args,
            env,
            show_pictures=not cli.no_plot,
            d_capture=cli.d_capture,
            animation_path=cli.animation_output,
            animation_fps=cli.animation_fps,
            inference_window=cli.inference_window,
        )
