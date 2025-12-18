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

from dqn_agent import DQNAgent
from enhanced_env_wrapper import create_enhanced_env


def preprocess_observation(obs):
    """Preprocess the observation from the environment."""
    return {
        'image': obs['image'],
        'direction': obs['direction'],
        'agent_pos': obs.get('agent_pos', [0, 0])
    }


def train_agent(episodes=2000, max_steps=1000, save_interval=250,
                        model_dir='models', log_dir='logs',
                        use_curriculum=True, use_reward_shaping=True,
                        load_checkpoint=None):
    """Train the DQN agent with all improvements."""
    
    # Create directories
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f'dqn_maze_{timestamp}'
    writer = SummaryWriter(f'{log_dir}/{experiment_name}')
    
    print("="*60)
    print("🚀 DQN TRAINING FOR OBSTRUCTED MAZE")
    print("="*60)
    
    # Create enhanced environment
    env = create_enhanced_env(
        use_curriculum=use_curriculum,
        use_reward_shaping=use_reward_shaping
    )
    
    # Get observation and action spaces
    obs_shape = env.observation_space['image'].shape
    action_size = env.action_space.n
    
    print(f"\n📋 TRAINING CONFIGURATION:")
    print(f"  Environment: MiniGrid-ObstructedMaze-Full-v1")
    print(f"  Observation shape: {obs_shape}")
    print(f"  Action space: {action_size}")
    print(f"  Max episodes: {episodes}")
    print(f"  Max steps per episode: {max_steps}")
    print(f"  Reward shaping: {use_reward_shaping}")
    print(f"  Curriculum learning: {use_curriculum}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Create agent with optimized hyperparameters for EXPLORATION
    agent = DQNAgent(
        obs_shape=obs_shape,
        action_size=action_size,
        lr=1e-4,                    # Lower learning rate for stability
        gamma=0.99,                 # Standard discount factor
        epsilon=1.0,                # Start with full exploration
        epsilon_decay=0.9995,       # SLOWER decay - maintain exploration much longer!
        epsilon_min=0.05,           # Higher minimum - always explore a bit
        memory_size=200000,         # Large replay buffer
        batch_size=64,              # Larger batch size
        target_update=500,          # Frequent target updates
        use_prioritized_replay=True
    )
    
    # Load checkpoint if specified
    start_episode = 0
    if load_checkpoint and os.path.exists(load_checkpoint):
        print(f"\n🔄 Loading checkpoint: {load_checkpoint}")
        agent.load(load_checkpoint)
        # Extract episode number from filename if possible
        try:
            start_episode = int(load_checkpoint.split('episode_')[-1].split('.')[0])
            print(f"   Resuming from episode: {start_episode}")
        except:
            print("   Starting fresh episode count")
    
    # Training metrics
    scores = []
    losses = []
    episode_lengths = []
    success_rate_window = []
    exploration_rates = []
    bonus_rewards = []
    
    # Performance tracking
    best_success_rate = 0
    best_avg_reward = float('-inf')
    consecutive_successes = 0
    training_start_time = time.time()
    
    print(f"\n TRAINING STARTED:")
    print(f"  Agent epsilon: {agent.epsilon:.4f}")
    print(f"  Memory capacity: {agent.memory.capacity if hasattr(agent.memory, 'capacity') else 'Standard'}")
    print(f"  Target update frequency: {agent.target_update}")
    
    # Training loop
    try:
        for episode in tqdm(range(start_episode, episodes), desc="Training", 
                           initial=start_episode, total=episodes):
            
            # Reset environment and agent tracking
            obs, info = env.reset()
            state = preprocess_observation(obs)
            agent.reset_episode()
            
            # Episode variables
            total_reward = 0
            original_reward = 0
            total_bonus = 0
            steps = 0
            episode_loss = []
            
            for step in range(max_steps):
                # Choose action
                action = agent.act(state, training=True)
                
                # Take action
                next_obs, reward, done, truncated, info = env.step(action)
                next_state = preprocess_observation(next_obs)
                
                # Track rewards
                bonus_reward = info.get('bonus_reward', 0)
                original_reward += info.get('original_reward', 0)
                total_bonus += bonus_reward
                
                # Store experience with enhanced info
                agent.remember(state, action, reward, next_state, done or truncated, info)
                
                # Train agent
                if len(agent.memory) > agent.batch_size:
                    loss = agent.replay()
                    if loss > 0:
                        episode_loss.append(loss)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if done or truncated:
                    break
            
            # Record metrics
            scores.append(total_reward)
            episode_lengths.append(steps)
            success_rate_window.append(1 if original_reward > 0 else 0)
            exploration_rates.append(agent.epsilon)
            bonus_rewards.append(total_bonus)
            
            # Keep success rate window at 100 episodes
            if len(success_rate_window) > 100:
                success_rate_window.pop(0)
            
            avg_loss = np.mean(episode_loss) if episode_loss else 0
            losses.append(avg_loss)
            
            # Update consecutive successes
            if original_reward > 0:
                consecutive_successes += 1
            else:
                consecutive_successes = 0
            
            # Log metrics to TensorBoard
            writer.add_scalar('Training/Total_Reward', total_reward, episode)
            writer.add_scalar('Training/Original_Reward', original_reward, episode)
            writer.add_scalar('Training/Bonus_Reward', total_bonus, episode)
            writer.add_scalar('Training/Episode_Length', steps, episode)
            writer.add_scalar('Training/Loss', avg_loss, episode)
            writer.add_scalar('Training/Epsilon', agent.epsilon, episode)
            writer.add_scalar('Training/Success_Rate', np.mean(success_rate_window), episode)
            writer.add_scalar('Training/Consecutive_Successes', consecutive_successes, episode)
            
            if hasattr(env, 'current_difficulty'):
                writer.add_scalar('Curriculum/Difficulty', env.current_difficulty, episode)
            
            # Detailed progress reporting
            if episode % 100 == 0 and episode > 0:
                avg_score = np.mean(scores[-100:])
                avg_length = np.mean(episode_lengths[-100:])
                avg_bonus = np.mean(bonus_rewards[-100:])
                success_rate = np.mean(success_rate_window)
                recent_successes = sum(success_rate_window[-50:])
                
                elapsed_time = time.time() - training_start_time
                eps_per_hour = episode / (elapsed_time / 3600) if elapsed_time > 0 else 0
                
                print(f"\n📊 EPISODE {episode} REPORT:")
                print(f"  Total Reward (avg last 100): {avg_score:.3f}")
                print(f"  Bonus Reward (avg last 100): {avg_bonus:.3f}")
                print(f"  Episode Length (avg last 100): {avg_length:.1f}")
                print(f"  Success Rate (last 100): {success_rate:.2%}")
                print(f"  Recent Successes (last 50): {recent_successes}/50")
                print(f"  Consecutive Successes: {consecutive_successes}")
                print(f"  Epsilon: {agent.epsilon:.4f}")
                print(f"  Memory Size: {len(agent.memory)}")
                print(f"  Training Speed: {eps_per_hour:.1f} eps/hour")
                
                # Track best performance
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    print(f"  NEW BEST SUCCESS RATE: {best_success_rate:.2%}")
                
                if avg_score > best_avg_reward:
                    best_avg_reward = avg_score
                    print(f"  NEW BEST AVERAGE REWARD: {best_avg_reward:.3f}")
                
                # Curriculum learning info
                if hasattr(env, 'current_difficulty'):
                    print(f"  Curriculum Difficulty: {env.current_difficulty:.3f}")
            
            # Save model periodically
            if episode % save_interval == 0 and episode > 0:
                model_path = f"{model_dir}/dqn_episode_{episode}.pth"
                agent.save(model_path)
                if episode % (save_interval * 2) == 0:  # Less frequent verbose saves
                    print(f"Model saved: {model_path}")
            
            # Early success celebration
            if original_reward > 0:
                print(f"SUCCESS at episode {episode}! Reward: {original_reward:.2f}, "
                      f"Steps: {steps}, Consecutive: {consecutive_successes}")
            
            # Early stopping conditions
            if episode > 1000:
                recent_success_rate = np.mean(success_rate_window[-100:])
                if recent_success_rate > 0.85:
                    print(f"\nEXCELLENT PERFORMANCE ACHIEVED!")
                    print(f"   Success rate: {recent_success_rate:.2%}")
                    print(f"   Stopping training early at episode {episode}")
                    break
                elif consecutive_successes >= 20:
                    print(f"\nCONSISTENT SUCCESS ACHIEVED!")
                    print(f"   {consecutive_successes} consecutive successes")
                    print(f"   Stopping training early at episode {episode}")
                    break
    
    except KeyboardInterrupt:
        print(f"\nTraining interrupted at episode {episode}")
    
    # Save final model
    final_model_path = f"{model_dir}/dqn_final.pth"
    agent.save(final_model_path)
    
    # Close environment and writer
    env.close()
    writer.close()
    
    # Training summary
    total_time = time.time() - training_start_time
    final_success_rate = np.mean(success_rate_window) if success_rate_window else 0
    
    print(f"\n" + "="*60)
    print("📊 TRAINING COMPLETED!")
    print("="*60)
    print(f"  Episodes completed: {episode}")
    print(f"  Total training time: {total_time/3600:.2f} hours")
    print(f"  Final success rate: {final_success_rate:.2%}")
    print(f"  Best success rate: {best_success_rate:.2%}")
    print(f"  Best average reward: {best_avg_reward:.3f}")
    print(f"  Final model: {final_model_path}")
    print("="*60)
    
    # Plot comprehensive training results
    plot_enhanced_training_results(
        scores, losses, episode_lengths, success_rate_window, 
        exploration_rates, bonus_rewards, experiment_name
    )
    
    return agent


def plot_enhanced_training_results(scores, losses, episode_lengths, success_rate_window, 
                                 exploration_rates, bonus_rewards, experiment_name):
    """Plot comprehensive training metrics."""
    episodes = range(len(scores))
    
    # Create comprehensive plot
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Rewards plot
    ax1 = plt.subplot(2, 3, 1)
    plt.plot(episodes, scores, alpha=0.3, color='blue', label='Total Reward')
    plt.plot(episodes, [np.mean(scores[max(0, i-100):i+1]) for i in episodes], 
             'b-', linewidth=2, label='100-ep avg')
    plt.plot(episodes, bonus_rewards, alpha=0.3, color='orange', label='Bonus Reward')
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Success rate
    ax2 = plt.subplot(2, 3, 2)
    success_rates = []
    for i in range(len(scores)):
        window_start = max(0, i-99)
        window_scores = [1 if scores[j] > 0 else 0 for j in range(window_start, i+1)]
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
    plt.plot(episodes, [np.mean(episode_lengths[max(0, i-100):i+1]) for i in episodes], 
             'r-', linewidth=2, label='100-ep avg')
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Training loss
    ax4 = plt.subplot(2, 3, 4)
    plt.plot(episodes, losses, alpha=0.6, color='purple')
    plt.plot(episodes, [np.mean(losses[max(0, i-100):i+1]) for i in episodes], 
             'purple', linewidth=2, label='100-ep avg')
    plt.title('Training Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Exploration vs Success
    ax5 = plt.subplot(2, 3, 5)
    ax5_twin = ax5.twinx()
    
    line1 = ax5.plot(episodes, success_rates, 'g-', linewidth=2, label='Success Rate')
    line2 = ax5_twin.plot(episodes, exploration_rates, 'orange', linewidth=2, label='Epsilon')
    
    ax5.set_ylabel('Success Rate', color='g')
    ax5_twin.set_ylabel('Exploration Rate', color='orange')
    ax5.set_xlabel('Episode')
    ax5.set_title('Success vs Exploration')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax5.legend(lines, labels, loc='center right')
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
    
    print(f"📈 Training plots saved as: {experiment_name}_results.png")


def main():
    parser = argparse.ArgumentParser(description='Train Enhanced DQN agent on ObstructedMaze-Full')
    parser.add_argument('--episodes', type=int, default=2000,
                        help='Number of training episodes')
    parser.add_argument('--max_steps', type=int, default=1000, 
                        help='Maximum steps per episode')
    parser.add_argument('--save_interval', type=int, default=250, 
                        help='Save model every N episodes')
    parser.add_argument('--model_dir', type=str, default='models', 
                        help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='logs', 
                        help='Directory for tensorboard logs')
    parser.add_argument('--no_curriculum', action='store_true', 
                        help='Disable curriculum learning')
    parser.add_argument('--no_reward_shaping', action='store_true', 
                        help='Disable reward shaping')
    parser.add_argument('--load_checkpoint', type=str, default=None, 
                        help='Path to checkpoint to resume training from')
    
    args = parser.parse_args()
    
    # Train the agent
    agent = train_agent(
        episodes=args.episodes,
        max_steps=args.max_steps,
        save_interval=args.save_interval,
        model_dir=args.model_dir,
        log_dir=args.log_dir,
        use_curriculum=not args.no_curriculum,
        use_reward_shaping=not args.no_reward_shaping,
        load_checkpoint=args.load_checkpoint
    )
    
    print("\n🎊 Training session completed successfully!")


if __name__ == "__main__":
    main()

