import gymnasium as gym
import minigrid
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
import os
from PIL import Image
import imageio

from dqn_agent import DQNAgent


def create_env(render_mode=None):
    """Create and configure the MiniGrid ObstructedMaze-Full environment."""
    env = gym.make("MiniGrid-ObstructedMaze-Full-v1", render_mode=render_mode)
    return env


def preprocess_observation(obs):
    """Preprocess the observation from the environment."""
    return {
        'image': obs['image'],
        'direction': obs['direction']
    }


def evaluate_agent(model_path, num_episodes=100, render=False, save_gif=False):
    """Evaluate a trained DQN agent."""
    
    # Create environment
    render_mode = "rgb_array" if (render or save_gif) else None
    env = create_env(render_mode=render_mode)
    
    # Get observation and action spaces
    obs_shape = env.observation_space['image'].shape
    action_size = env.action_space.n
    
    # Create and load agent
    agent = DQNAgent(obs_shape, action_size)
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"Model loaded from: {model_path}")
    else:
        print(f"Model file not found: {model_path}")
        return
    
    # Evaluation metrics
    scores = []
    episode_lengths = []
    successes = 0
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = preprocess_observation(obs)
        total_reward = 0
        steps = 0
        frames = []
        
        if save_gif and episode == 0:  # Save GIF for first episode
            frames.append(env.render())
        
        while True:
            # Choose action (no exploration)
            action = agent.act(state, training=False)
            
            # Take action
            next_obs, reward, done, truncated, info = env.step(action)
            next_state = preprocess_observation(next_obs)
            
            if save_gif and episode == 0:
                frames.append(env.render())
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done or truncated:
                break
        
        scores.append(total_reward)
        episode_lengths.append(steps)
        if total_reward > 0:
            successes += 1
        
        if episode % 10 == 0:
            print(f"Episode {episode}: Reward = {total_reward}, Steps = {steps}")
        
        # Save GIF for first episode
        if save_gif and episode == 0 and frames:
            save_episode_gif(frames, 'episode_demo.gif')
    
    # Print evaluation results
    print(f"\n=== Evaluation Results ({num_episodes} episodes) ===")
    print(f"Average Reward: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Success Rate: {successes/num_episodes:.2%} ({successes}/{num_episodes})")
    print(f"Max Reward: {np.max(scores):.2f}")
    print(f"Min Reward: {np.min(scores):.2f}")
    
    # Plot evaluation results
    plot_evaluation_results(scores, episode_lengths)
    
    env.close()
    
    return {
        'scores': scores,
        'episode_lengths': episode_lengths,
        'success_rate': successes / num_episodes,
        'avg_reward': np.mean(scores),
        'avg_length': np.mean(episode_lengths)
    }


def save_episode_gif(frames, filename):
    """Save episode frames as GIF."""
    print(f"Saving episode demonstration as {filename}...")
    imageio.mimsave(filename, frames, fps=5)
    print(f"GIF saved: {filename}")


def plot_evaluation_results(scores, episode_lengths):
    """Plot evaluation metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot rewards
    ax1.hist(scores, bins=20, alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(scores), color='red', linestyle='--', 
                label=f'Mean: {np.mean(scores):.2f}')
    ax1.set_title('Distribution of Episode Rewards')
    ax1.set_xlabel('Reward')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot episode lengths
    ax2.hist(episode_lengths, bins=20, alpha=0.7, edgecolor='black', color='orange')
    ax2.axvline(np.mean(episode_lengths), color='red', linestyle='--', 
                label=f'Mean: {np.mean(episode_lengths):.1f}')
    ax2.set_title('Distribution of Episode Lengths')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def demo_episode(model_path, render=True, save_gif=True):
    """Run a single demonstration episode."""
    
    # Create environment with rendering
    env = create_env(render_mode="rgb_array" if save_gif else "human")
    
    # Get observation and action spaces
    obs_shape = env.observation_space['image'].shape
    action_size = env.action_space.n
    
    # Create and load agent
    agent = DQNAgent(obs_shape, action_size)
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"Model loaded from: {model_path}")
    else:
        print(f"Model file not found: {model_path}")
        return
    
    obs, _ = env.reset()
    state = preprocess_observation(obs)
    total_reward = 0
    steps = 0
    frames = []
    
    print("Starting demonstration episode...")
    print("Mission:", obs.get('mission', 'N/A'))
    
    if save_gif:
        frames.append(env.render())
    
    while True:
        # Choose action (no exploration)
        action = agent.act(state, training=False)
        
        # Action names for better understanding
        action_names = ['left', 'right', 'forward', 'pickup', 'drop', 'toggle', 'done']
        print(f"Step {steps + 1}: Action = {action} ({action_names[action] if action < len(action_names) else 'unknown'})")
        
        # Take action
        next_obs, reward, done, truncated, info = env.step(action)
        next_state = preprocess_observation(next_obs)
        
        if save_gif:
            frames.append(env.render())
        
        state = next_state
        total_reward += reward
        steps += 1
        
        print(f"  Reward: {reward}, Total Reward: {total_reward}")
        
        if done or truncated:
            print(f"Episode finished! Final reward: {total_reward}, Steps: {steps}")
            if total_reward > 0:
                print("🎉 SUCCESS! Agent found the blue ball!")
            else:
                print("❌ Episode ended without success.")
            break
    
    if save_gif and frames:
        save_episode_gif(frames, 'demo_episode.gif')
    
    env.close()


def compare_models(model_paths, num_episodes=50):
    """Compare multiple trained models."""
    
    results = {}
    
    for model_name, model_path in model_paths.items():
        print(f"\nEvaluating {model_name}...")
        result = evaluate_agent(model_path, num_episodes, render=False, save_gif=False)
        results[model_name] = result
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    models = list(results.keys())
    avg_rewards = [results[model]['avg_reward'] for model in models]
    success_rates = [results[model]['success_rate'] for model in models]
    
    # Plot average rewards
    bars1 = ax1.bar(models, avg_rewards, alpha=0.7)
    ax1.set_title('Average Reward Comparison')
    ax1.set_ylabel('Average Reward')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars1, avg_rewards):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.2f}', ha='center', va='bottom')
    
    # Plot success rates
    bars2 = ax2.bar(models, [rate * 100 for rate in success_rates], 
                    alpha=0.7, color='orange')
    ax2.set_title('Success Rate Comparison')
    ax2.set_ylabel('Success Rate (%)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars2, success_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.1%}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate DQN agent on MiniGrid ObstructedMaze-Full')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the trained model')
    parser.add_argument('--episodes', type=int, default=100, 
                        help='Number of evaluation episodes')
    parser.add_argument('--demo', action='store_true', 
                        help='Run a single demonstration episode')
    parser.add_argument('--save_gif', action='store_true', 
                        help='Save demonstration as GIF')
    parser.add_argument('--render', action='store_true', 
                        help='Render the environment')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_episode(args.model_path, render=args.render, save_gif=args.save_gif)
    else:
        evaluate_agent(args.model_path, args.episodes, render=args.render, 
                      save_gif=args.save_gif)


if __name__ == "__main__":
    main()