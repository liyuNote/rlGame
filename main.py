import argparse
import os
from pathlib import Path

# Keep matplotlib cache inside the project so Windows user-directory permissions
# do not affect command-line rendering.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib
import numpy as np
from tqdm import tqdm

from env import satellites
from ppo_continuous import PPO_continuous
from replaybuffer import ReplayBuffer
import plot_function as pf

# Headless backend: save images/animations without opening a GUI window.
matplotlib.use("Agg")


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
        max_episode_steps=1000,
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


def build_env(args, d_capture):
    """Build the default pursuit/evasion scenario."""
    return satellites(
        Pursuer_position=np.array([200000.0, 0.0, 0.0]),
        Pursuer_vector=np.array([0.0, 0.0, 0.0]),
        Escaper_position=np.array([18000.0, 0.0, 0.0]),
        Escaper_vector=np.array([0.0, 0.0, 0.0]),
        d_capture=d_capture,
        args=args,
    )


def init_agent_dims(args, env):
    """Fill dimensions that are only known after the environment is created."""
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = float(env.action_space[0][1])


def train_network(args, env, show_picture=True, pre_train=False, d_capture=15000.0):
    """Train the pursuer PPO agent.

    The evader agent is instantiated and produces actions, but it is not updated
    in the current single-agent training loop. It acts as an untrained opponent.
    """
    episode_rewards = []
    episode_mean_rewards = []
    env.d_capture = d_capture
    init_agent_dims(args, env)

    replay_buffer = ReplayBuffer(args)
    pursuer_agent = PPO_continuous(args, "pursuer")
    evader_agent = PPO_continuous(args, "evader")
    if pre_train:
        pursuer_agent.load_checkpoint()

    pbar = tqdm(range(args.max_train_steps), desc="Training pursuer", unit="episode")
    for episode in pbar:
        episode_reward = 0.0
        episode_count = 0
        s = env.reset(0)

        while True:
            episode_count += 1

            # choose_action samples from the policy and returns old log_prob,
            # which PPO later uses to compute the probability ratio.
            pursuer_a, pursuer_a_logprob = pursuer_agent.choose_action(s)
            evader_a, _evader_a_logprob = evader_agent.choose_action(s)

            if args.policy_dist == "Beta":
                pursuer_action = 2 * (pursuer_a - 0.5) * args.max_action
                evader_action = 2 * (evader_a - 0.5) * args.max_action
            else:
                pursuer_action = pursuer_a
                evader_action = evader_a

            s_, r, done = env.step(pursuer_action, evader_action, episode_count)
            episode_reward += r

            # dw marks transitions where there is no bootstrap value from s_.
            dw = bool(done or episode_count >= args.max_episode_steps)
            replay_buffer.store(s, pursuer_action, pursuer_a_logprob, r, s_, dw, done)
            s = s_

            if replay_buffer.count == args.batch_size:
                pursuer_agent.update(replay_buffer, episode)
                replay_buffer.count = 0

            if done:
                episode_rewards.append(episode_reward)
                episode_mean_rewards.append(float(np.mean(episode_rewards)))
                pbar.set_postfix(
                    {
                        "episode": episode,
                        "reward": f"{episode_reward:.1f}",
                        "mean_reward": f"{episode_mean_rewards[-1]:.1f}",
                    }
                )
                break

    pursuer_agent.save_checkpoint()
    if show_picture:
        pf.plot_train_reward(episode_rewards, episode_mean_rewards)
    return pursuer_agent


def test_network(
    args,
    env,
    show_pictures=True,
    d_capture=20000.0,
    max_steps=None,
    animation_path="outputs/satellite_game.gif",
    animation_fps=8,
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
    }

    env.d_capture = d_capture
    init_agent_dims(args, env)
    pursuer_agent = PPO_continuous(args, "pursuer")
    pursuer_agent.load_checkpoint()
    evader_agent = PPO_continuous(args, "evader")

    s = env.reset(2)
    max_steps = max_steps or args.max_episode_steps
    while episode_count < max_steps:
        episode_count += 1

        # evaluate uses the policy mean, so the same checkpoint and initial
        # state produce a stable inference trajectory.
        pursuer_a = pursuer_agent.evaluate(s)
        evader_a = evader_agent.evaluate(s)
        if args.policy_dist == "Beta":
            pursuer_action = 2 * (pursuer_a - 0.5) * args.max_action
            evader_action = 2 * (evader_a - 0.5) * args.max_action
        else:
            pursuer_action = pursuer_a
            evader_action = evader_a

        s, r, done = env.step(pursuer_action, evader_action, episode_count)
        episode_reward += r

        pursuer_position.append(s[6:9])
        escaper_position.append(s[12:15])
        history["pursuer_position"].append(s[6:9])
        history["evader_position"].append(s[12:15])
        history["pursuer_velocity"].append(s[9:12])
        history["evader_velocity"].append(s[15:18])
        history["pursuer_action"].append(np.asarray(pursuer_action, dtype=np.float32))
        history["evader_action"].append(np.asarray(evader_action, dtype=np.float32))
        history["reward"].append(float(r))
        history["cumulative_reward"].append(float(episode_reward))
        history["distance"].append(float(env.dis))
        if done:
            break

    print(f"Test reward: {episode_reward:.3f}, steps: {episode_count}, distance: {env.dis:.3f}")
    if show_pictures:
        pf.plot_trajectory(pursuer_position, escaper_position)
        animation_file = pf.animate_satellite_game(history, save_path=animation_path, fps=animation_fps)
        print(f"Animation saved to: {animation_file}")
    return episode_reward


def parse_args():
    """Command-line arguments for training, inference and rendering."""
    parser = argparse.ArgumentParser(description="PPO satellite pursuit/evasion")
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--max-train-steps", type=int, default=400)
    parser.add_argument("--max-episode-steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--mini-batch-size", type=int, default=32)
    parser.add_argument("--k-epochs", type=int, default=3)
    parser.add_argument("--hidden-width", type=int, default=128)
    parser.add_argument("--policy-dist", choices=["Gaussian", "Beta"], default="Gaussian")
    parser.add_argument("--chkpt-dir", default="checkpoints")
    parser.add_argument("--d-capture", type=float, default=20000.0)
    parser.add_argument("--pre-train", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--animation-output", default="outputs/satellite_game.gif")
    parser.add_argument("--animation-fps", type=int, default=8)
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
        )
