import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import argparse
import time

from ppo_agent import PPOAgent
from enhanced_env_wrapper import create_enhanced_env


def preprocess_observation(obs):
    """Preprocess the observation from the environment."""
    return {
        'image': obs['image'],
        'direction': obs['direction'],
        'agent_pos': obs.get('agent_pos', [0, 0]),
        'agent_pos_normalized': obs.get('agent_pos_normalized', [0.0, 0.0]),
        'goal_direction': obs.get('goal_direction', [0.0, 0.0])
    }


def train_ppo_agent(
        episodes=2000,
        max_steps=2500,
        update_frequency=512,  # ← CHANGE THIS from 2048 to 512
        save_interval=250,
        model_dir='models_ppo',
        log_dir='logs_ppo',
        use_curriculum=True,
        use_reward_shaping=True,
        load_checkpoint=None,
        render_episodes=0  # ADD THIS LINE
):
    """Train the PPO agent."""
    # Create directories
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f'ppo_maze_{timestamp}'
    writer = SummaryWriter(f'{log_dir}/{experiment_name}')

    print("=" * 60)
    print("🚀 PPO TRAINING FOR OBSTRUCTED MAZE")
    print("=" * 60)

    # Create environment
    env = create_enhanced_env(
        use_curriculum=use_curriculum,
        use_reward_shaping=use_reward_shaping
    )
    # if hasattr(env, 'max_steps'):
    #     env.max_steps = 500  # Override the wrapper's limit

    # Get observation and action spaces
    obs_shape = env.observation_space['image'].shape
    action_size = env.action_space.n

    print(f"\n📋 TRAINING CONFIGURATION:")
    print(f"  Environment: MiniGrid-ObstructedMaze-Full-v1")
    print(f"  Observation shape: {obs_shape}")
    print(f"  Action space: {action_size}")
    print(f"  Max episodes: {episodes}")
    print(f"  Update frequency: {update_frequency} steps")
    print(f"  Algorithm: PPO")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # Create agent
    agent = PPOAgent(
        obs_shape=obs_shape,
        action_size=action_size,
        lr=3e-4,        # ← Make sure this is 3e-4
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        c1=0.5,
        c2=0.01,
        epochs=10,      # ← CHANGE THIS from 4 to 10 (matches ppo_agent.py)
        batch_size=64
    )

    # Load checkpoint if specified
    start_episode = 0
    if load_checkpoint and os.path.exists(load_checkpoint):
        print(f"\n🔄 Loading checkpoint: {load_checkpoint}")
        agent.load(load_checkpoint)
        try:
            start_episode = int(load_checkpoint.split('episode_')[-1].split('.')[0])
            print(f"  Resuming from episode: {start_episode}")
        except:
            print("  Starting fresh episode count")

    # Training metrics
    scores = []
    policy_losses = []
    value_losses = []
    episode_lengths = []
    success_rate_window = []

    # Performance tracking
    best_success_rate = 0
    best_avg_reward = float('-inf')
    consecutive_successes = 0
    training_start_time = time.time()
    total_steps = 0

    print(f"\n🎬 TRAINING STARTED")

    # Training loop
    try:
        for episode in tqdm(range(start_episode, episodes), desc="Training",
                            initial=start_episode, total=episodes):
            obs, info = env.reset()
            state = preprocess_observation(obs)

            total_reward = 0
            original_reward = 0
            steps = 0

            for step in range(max_steps):
                # ADD THIS - Render if within first N episodes
                if episode < render_episodes:
                    env.render()

                # Select action
                action = agent.act(state, training=True)

                # Take action
                next_obs, reward, done, truncated, info = env.step(action)
                next_state = preprocess_observation(next_obs)

                # Store reward and done
                agent.store_reward_done(reward, done or truncated)

                # Track metrics
                original_reward += info.get('original_reward', 0)
                total_reward += reward
                steps += 1
                total_steps += 1

                state = next_state

                # Update policy at fixed intervals
                if total_steps % update_frequency == 0:
                    avg_loss, avg_policy_loss, avg_value_loss = agent.update(
                        next_state if not (done or truncated) else None)
                    policy_losses.append(avg_policy_loss)
                    value_losses.append(avg_value_loss)

                    writer.add_scalar('Training/Policy_Loss', avg_policy_loss, total_steps)
                    writer.add_scalar('Training/Value_Loss', avg_value_loss, total_steps)

                if done or truncated:
                    break

            # Update at end of episode if not already updated
            if len(agent.states) > 0:
                avg_loss, avg_policy_loss, avg_value_loss = agent.update()
                if avg_loss > 0:
                    policy_losses.append(avg_policy_loss)
                    value_losses.append(avg_value_loss)

            # Record metrics
            scores.append(total_reward)
            episode_lengths.append(steps)
            success_rate_window.append(1 if original_reward > 0 else 0)

            if len(success_rate_window) > 100:
                success_rate_window.pop(0)

            # Update consecutive successes
            if original_reward > 0:
                consecutive_successes += 1
            else:
                consecutive_successes = 0

            # Log to TensorBoard
            writer.add_scalar('Training/Total_Reward', total_reward, episode)
            writer.add_scalar('Training/Original_Reward', original_reward, episode)
            writer.add_scalar('Training/Episode_Length', steps, episode)
            writer.add_scalar('Training/Success_Rate', np.mean(success_rate_window), episode)
            writer.add_scalar('Training/Consecutive_Successes', consecutive_successes, episode)

            # Progress reporting
            if episode % 100 == 0 and episode > 0:
                avg_score = np.mean(scores[-100:])
                avg_length = np.mean(episode_lengths[-100:])
                success_rate = np.mean(success_rate_window)
                elapsed_time = time.time() - training_start_time
                eps_per_hour = episode / (elapsed_time / 3600) if elapsed_time > 0 else 0

                print(f"\n📊 EPISODE {episode} REPORT:")
                print(f"  Total Reward (avg last 100): {avg_score:.3f}")
                print(f"  Episode Length (avg last 100): {avg_length:.1f}")
                print(f"  Success Rate (last 100): {success_rate:.2%}")
                print(f"  Consecutive Successes: {consecutive_successes}")
                print(f"  Total Steps: {total_steps}")
                print(f"  Training Speed: {eps_per_hour:.1f} eps/hour")

                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    print(f"  🎯 NEW BEST SUCCESS RATE: {best_success_rate:.2%}")

                if avg_score > best_avg_reward:
                    best_avg_reward = avg_score
                    print(f"  🎯 NEW BEST AVERAGE REWARD: {best_avg_reward:.3f}")

            # Save model periodically
            if episode % save_interval == 0 and episode > 0:
                model_path = f"{model_dir}/ppo_episode_{episode}.pth"
                agent.save(model_path)
                if episode % (save_interval * 2) == 0:
                    print(f"💾 Model saved: {model_path}")

            # Early success celebration
            if original_reward > 0:
                print(f"✅ SUCCESS at episode {episode}! Steps: {steps}")

            # Early stopping
            if episode > 200 and episode % 100 == 0:
                recent_success_rate = np.mean(success_rate_window)
                if recent_success_rate > 0.85:
                    print(f"\n🎉 EXCELLENT PERFORMANCE! Success rate: {recent_success_rate:.2%}")
                    break
                elif consecutive_successes >= 20:
                    print(f"\n🎉 CONSISTENT SUCCESS! {consecutive_successes} in a row")
                    break

    except KeyboardInterrupt:
        print(f"\n⚠️ Training interrupted at episode {episode}")

    # Save final model
    final_model_path = f"{model_dir}/ppo_final.pth"
    agent.save(final_model_path)

    env.close()
    writer.close()

    # Training summary
    total_time = time.time() - training_start_time
    final_success_rate = np.mean(success_rate_window) if success_rate_window else 0

    print(f"\n" + "=" * 60)
    print("📊 TRAINING COMPLETED!")
    print("=" * 60)
    print(f"  Episodes completed: {episode}")
    print(f"  Total training time: {total_time / 3600:.2f} hours")
    print(f"  Final success rate: {final_success_rate:.2%}")
    print(f"  Best success rate: {best_success_rate:.2%}")
    print(f"  Best average reward: {best_avg_reward:.3f}")
    print(f"  Final model: {final_model_path}")
    print("=" * 60)

    return agent


def plot_ppo_training_results(scores, policy_losses, value_losses, episode_lengths,
                              success_rate_window, experiment_name):
    """Plot comprehensive PPO training metrics."""
    episodes = range(len(scores))

    # Create comprehensive plot
    fig = plt.figure(figsize=(20, 12))

    # 1. Rewards plot
    ax1 = plt.subplot(2, 3, 1)
    plt.plot(episodes, scores, alpha=0.3, color='blue', label='Total Reward')
    plt.plot(episodes, [np.mean(scores[max(0, i - 100):i + 1]) for i in episodes],
             'b-', linewidth=2, label='100-ep avg')
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Success rate
    ax2 = plt.subplot(2, 3, 2)
    success_rates = []
    for i in range(len(scores)):
        window_start = max(0, i - 99)
        window_scores = [1 if scores[j] > 0 else 0 for j in range(window_start, i + 1)]
        success_rates.append(np.mean(window_scores))
    plt.plot(episodes, success_rates, 'g-', linewidth=2)
    plt.title('Success Rate (100-episode window)')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)

    # 3. Episode lengths
    ax3 = plt.subplot(2, 3, 3)
    plt.plot(episodes, episode_lengths, alpha=0.3, color='red')
    plt.plot(episodes, [np.mean(episode_lengths[max(0, i - 100):i + 1]) for i in episodes],
             'r-', linewidth=2, label='100-ep avg')
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. Training losses
    ax4 = plt.subplot(2, 3, 4)
    if policy_losses:
        loss_episodes = np.linspace(0, len(episodes) - 1, len(policy_losses))
        plt.plot(loss_episodes, policy_losses, alpha=0.6, color='purple', label='Policy Loss')
        plt.plot(loss_episodes, value_losses, alpha=0.6, color='orange', label='Value Loss')
        plt.title('Training Losses')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # 5. Success Rate Distribution
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(episodes, success_rates, 'g-', linewidth=2, label='Success Rate')
    ax5.set_ylabel('Success Rate', color='g')
    ax5.set_xlabel('Episode')
    ax5.set_title('Success Rate Progression')
    ax5.legend(loc='upper left')
    ax5.grid(True, alpha=0.3)

    # 6. Reward distribution
    ax6 = plt.subplot(2, 3, 6)
    successful_episodes = [score for score in scores if score > 0]
    failed_episodes = [score for score in scores if score <= 0]

    if successful_episodes:
        plt.hist(successful_episodes, bins=20, alpha=0.7, color='green',
                 label=f'Successful ({len(successful_episodes)})')
    if failed_episodes:
        plt.hist(failed_episodes, bins=20, alpha=0.7, color='red',
                 label=f'Failed ({len(failed_episodes)})')

    plt.title('Reward Distribution')
    plt.xlabel('Total Reward')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{experiment_name}_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n📈 Training plots saved as: {experiment_name}_results.png")


def main():
    parser = argparse.ArgumentParser(description='Train PPO agent on ObstructedMaze-Full')
    parser.add_argument('--episodes', type=int, default=2000, help='Number of training episodes')
    parser.add_argument('--max_steps', type=int, default=1000, help='Maximum steps per episode')
    parser.add_argument('--update_frequency', type=int, default=2048, help='Update every N steps')
    parser.add_argument('--save_interval', type=int, default=250, help='Save model every N episodes')
    parser.add_argument('--model_dir', type=str, default='models_ppo', help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='logs_ppo', help='Directory for tensorboard logs')
    parser.add_argument('--no_curriculum', action='store_true', help='Disable curriculum learning')
    parser.add_argument('--no_reward_shaping', action='store_true', help='Disable reward shaping')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--render', type=int, default=0, help='Render first N episodes')  # ADD THIS LINE

    args = parser.parse_args()

    agent = train_ppo_agent(
        episodes=args.episodes,
        max_steps=args.max_steps,
        update_frequency=args.update_frequency,
        save_interval=args.save_interval,
        model_dir=args.model_dir,
        log_dir=args.log_dir,
        use_curriculum=not args.no_curriculum,
        use_reward_shaping=not args.no_reward_shaping,
        load_checkpoint=args.load_checkpoint,
        render_episodes = args.render  # ADD THIS LINE
    )

    print("\n🎊 Training session completed successfully!")


if __name__ == "__main__":
    main()
