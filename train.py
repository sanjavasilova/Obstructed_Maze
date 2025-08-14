import gymnasium as gym
import minigrid
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import argparse

from dqn_agent import DQNAgent


def create_env():
    """Create and configure the MiniGrid ObstructedMaze-Full environment."""
    env = gym.make("MiniGrid-ObstructedMaze-Full-v1")
    return env


def preprocess_observation(obs):
    """Preprocess the observation from the environment."""
    return {
        'image': obs['image'],
        'direction': obs['direction']
    }


def train_agent(episodes=2000, max_steps=500, save_interval=200, 
                model_dir='models', log_dir='logs'):
    """Train the DQN agent on ObstructedMaze-Full environment."""
    
    # Create directories
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(f'{log_dir}/obstructed_maze_{timestamp}')
    
    # Create environment
    env = create_env()
    
    # Get observation and action spaces
    obs_shape = env.observation_space['image'].shape
    action_size = env.action_space.n
    
    print(f"Environment: MiniGrid-ObstructedMaze-Full-v1")
    print(f"Observation shape: {obs_shape}")
    print(f"Action space: {action_size}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Create agent
    agent = DQNAgent(obs_shape, action_size)
    
    # Training metrics
    scores = []
    losses = []
    episode_lengths = []
    success_rate_window = []
    
    # Training loop
    for episode in tqdm(range(episodes), desc="Training"):
        obs, _ = env.reset()
        state = preprocess_observation(obs)
        total_reward = 0
        steps = 0
        episode_loss = []
        
        for step in range(max_steps):
            # Choose action
            action = agent.act(state, training=True)
            
            # Take action
            next_obs, reward, done, truncated, info = env.step(action)
            next_state = preprocess_observation(next_obs)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done or truncated)
            
            # Train agent
            if len(agent.memory) > agent.batch_size:
                loss = agent.replay()
                episode_loss.append(loss)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done or truncated:
                break
        
        # Record metrics
        scores.append(total_reward)
        episode_lengths.append(steps)
        success_rate_window.append(1 if total_reward > 0 else 0)
        
        # Keep success rate window at 100 episodes
        if len(success_rate_window) > 100:
            success_rate_window.pop(0)
        
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        losses.append(avg_loss)
        
        # Log metrics
        writer.add_scalar('Training/Episode_Reward', total_reward, episode)
        writer.add_scalar('Training/Episode_Length', steps, episode)
        writer.add_scalar('Training/Average_Loss', avg_loss, episode)
        writer.add_scalar('Training/Epsilon', agent.epsilon, episode)
        writer.add_scalar('Training/Success_Rate', np.mean(success_rate_window), episode)
        
        # Print progress
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            success_rate = np.mean(success_rate_window)
            print(f"\nEpisode {episode}")
            print(f"Average Score (last 100): {avg_score:.2f}")
            print(f"Average Length (last 100): {avg_length:.2f}")
            print(f"Success Rate (last 100): {success_rate:.2%}")
            print(f"Epsilon: {agent.epsilon:.3f}")
        
        # Save model periodically
        if episode % save_interval == 0 and episode > 0:
            model_path = f"{model_dir}/dqn_obstructed_maze_episode_{episode}.pth"
            agent.save(model_path)
            print(f"Model saved: {model_path}")
    
    # Save final model
    final_model_path = f"{model_dir}/dqn_obstructed_maze_final.pth"
    agent.save(final_model_path)
    
    # Close environment and writer
    env.close()
    writer.close()
    
    # Plot training results
    plot_training_results(scores, losses, episode_lengths, success_rate_window)
    
    print(f"\nTraining completed!")
    print(f"Final model saved: {final_model_path}")
    
    return agent


def plot_training_results(scores, losses, episode_lengths, success_rate_window):
    """Plot training metrics."""
    episodes = range(len(scores))
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot scores
    ax1.plot(episodes, scores, alpha=0.6)
    ax1.plot(episodes, [np.mean(scores[max(0, i-100):i+1]) for i in episodes], 
             'r-', linewidth=2, label='100-episode average')
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.legend()
    ax1.grid(True)
    
    # Plot losses
    ax2.plot(episodes, losses, alpha=0.6)
    ax2.plot(episodes, [np.mean(losses[max(0, i-100):i+1]) for i in episodes], 
             'r-', linewidth=2, label='100-episode average')
    ax2.set_title('Training Loss')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    # Plot episode lengths
    ax3.plot(episodes, episode_lengths, alpha=0.6)
    ax3.plot(episodes, [np.mean(episode_lengths[max(0, i-100):i+1]) for i in episodes], 
             'r-', linewidth=2, label='100-episode average')
    ax3.set_title('Episode Lengths')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Steps')
    ax3.legend()
    ax3.grid(True)
    
    # Plot success rate
    success_rates = []
    for i in range(len(scores)):
        window_start = max(0, i-99)
        window_scores = scores[window_start:i+1]
        success_rate = np.mean([1 if score > 0 else 0 for score in window_scores])
        success_rates.append(success_rate)
    
    ax4.plot(episodes, success_rates, 'g-', linewidth=2)
    ax4.set_title('Success Rate (100-episode window)')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Success Rate')
    ax4.set_ylim(0, 1)
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Train DQN agent on MiniGrid ObstructedMaze-Full')
    parser.add_argument('--episodes', type=int, default=2000, help='Number of training episodes')
    parser.add_argument('--max_steps', type=int, default=500, help='Maximum steps per episode')
    parser.add_argument('--save_interval', type=int, default=200, help='Save model every N episodes')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory for tensorboard logs')
    
    args = parser.parse_args()
    
    # Train the agent
    agent = train_agent(
        episodes=args.episodes,
        max_steps=args.max_steps,
        save_interval=args.save_interval,
        model_dir=args.model_dir,
        log_dir=args.log_dir
    )


if __name__ == "__main__":
    main()