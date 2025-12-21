import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
import os
import imageio
import time

from ppo_agent import PPOAgent
from enhanced_env_wrapper import create_enhanced_env


def preprocess_observation(obs):
    return {
        'image': obs['image'],
        'direction': obs['direction'],
        'agent_pos': obs.get('agent_pos', [0, 0]),
        'agent_pos_normalized': obs.get('agent_pos_normalized', [0.0, 0.0]),
        'goal_direction': obs.get('goal_direction', [0.0, 0.0])
    }


def evaluate_ppo_agent(
        model_path,
        num_episodes=100,
        render=False,
        save_gif=False,
        use_curriculum=True,
        use_reward_shaping=True
):
    print("=" * 60)
    print("PPO AGENT EVALUATION")
    print("=" * 60)

    render_mode = 'rgb_array' if render or save_gif else None

    if use_curriculum or use_reward_shaping:
        env = create_enhanced_env(
            use_curriculum=use_curriculum,
            use_reward_shaping=use_reward_shaping
        )
        if render_mode:
            base_env = gym.make("MiniGrid-ObstructedMaze-Full-v1", render_mode=render_mode)
            if hasattr(env, 'env'):
                if hasattr(env.env, 'env'):
                    env.env.env = base_env
                else:
                    env.env = base_env
            else:
                env = base_env
    else:
        env = gym.make("MiniGrid-ObstructedMaze-Full-v1", render_mode=render_mode)

    obs_shape = env.observation_space['image'].shape
    action_size = env.action_space.n

    agent = PPOAgent(obs_shape, action_size)

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None

    agent.load(model_path)
    print("Model loaded")

    scores = []
    episode_lengths = []
    original_rewards = []
    bonus_rewards = []
    successes = 0
    total_time = 0

    best_reward = -float("inf")
    best_frames = None
    best_episode = -1

    for episode in range(num_episodes):
        start_time = time.time()
        obs, info = env.reset()
        state = preprocess_observation(obs)

        total_reward = 0
        original_reward = 0
        bonus_reward = 0
        steps = 0

        frames = [] if save_gif else None
        if frames is not None:
            frames.append(env.render())

        while True:
            action = agent.act(state, training=False)
            next_obs, reward, done, truncated, info = env.step(action)
            state = preprocess_observation(next_obs)

            if frames is not None:
                frames.append(env.render())

            total_reward += reward

            if 'original_reward' in info:
                original_reward += info['original_reward']
            else:
                original_reward = total_reward

            if 'bonus_reward' in info:
                bonus_reward += info['bonus_reward']

            steps += 1

            if done or truncated:
                break

        episode_time = time.time() - start_time
        total_time += episode_time

        scores.append(total_reward)
        episode_lengths.append(steps)
        original_rewards.append(original_reward)
        bonus_rewards.append(bonus_reward)

        if original_reward > 0:
            successes += 1

        if save_gif and total_reward > best_reward:
            best_reward = total_reward
            best_frames = frames
            best_episode = episode

        if episode % 10 == 0 or original_reward > 0:
            status = "SUCCESS" if original_reward > 0 else "FAILED"
            print(
                f"Episode {episode:3d} | {status} | "
                f"Reward {total_reward:.2f} | Steps {steps} | "
                f"Time {episode_time:.2f}s"
            )

    if save_gif and best_frames is not None:
        filename = f"ppo_best_episode_{best_episode}_reward_{best_reward:.2f}.gif"
        save_episode_gif(best_frames, filename)

    avg_time = total_time / num_episodes
    success_rate = successes / num_episodes

    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Episodes: {num_episodes}")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Average reward: {np.mean(scores):.3f}")
    print(f"Average episode length: {np.mean(episode_lengths):.1f}")
    print(f"Average time per episode: {avg_time:.2f}s")
    print("=" * 60)

    plot_evaluation_results(
        scores,
        episode_lengths,
        original_rewards,
        bonus_rewards,
        successes,
        num_episodes
    )

    env.close()

    return {
        'scores': scores,
        'episode_lengths': episode_lengths,
        'original_rewards': original_rewards,
        'bonus_rewards': bonus_rewards,
        'success_rate': success_rate
    }


def save_episode_gif(frames, filename):
    print(f"Saving best episode GIF: {filename}")
    imageio.mimsave(filename, frames, fps=8)
    print("GIF saved")


def plot_evaluation_results(scores, episode_lengths, original_rewards,
                            bonus_rewards, successes, num_episodes):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    ax1.hist(scores, bins=30)
    ax1.set_title("Total Reward Distribution")

    ax2.hist(episode_lengths, bins=30)
    ax2.set_title("Episode Length Distribution")

    success_flags = [1 if r > 0 else 0 for r in original_rewards]
    rolling = []
    window = min(20, len(success_flags))
    for i in range(len(success_flags)):
        rolling.append(np.mean(success_flags[max(0, i - window + 1):i + 1]))

    ax3.plot(rolling)
    ax3.set_ylim(0, 1)
    ax3.set_title("Rolling Success Rate")

    if any(bonus_rewards):
        ax4.scatter(original_rewards, bonus_rewards, alpha=0.6)
        ax4.set_title("Original vs Bonus Reward")
        ax4.set_xlabel("Original Reward")
        ax4.set_ylabel("Bonus Reward")
    else:
        ax4.plot(scores)
        ax4.set_title("Rewards Over Episodes")

    plt.tight_layout()
    plt.savefig("ppo_evaluation_results.png", dpi=300)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--save_gif", action="store_true")
    parser.add_argument("--no_curriculum", action="store_true")
    parser.add_argument("--no_reward_shaping", action="store_true")

    args = parser.parse_args()

    evaluate_ppo_agent(
        args.model_path,
        args.episodes,
        render=args.render,
        save_gif=args.save_gif,
        use_curriculum=not args.no_curriculum,
        use_reward_shaping=not args.no_reward_shaping
    )


if __name__ == "__main__":
    main()
