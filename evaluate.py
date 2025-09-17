import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
import os
from PIL import Image
import imageio
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


def evaluate_agent(model_path, num_episodes=100, render=False, save_gif=False,
                          use_curriculum=False, use_reward_shaping=False):
    """Evaluate a DQN agent."""
    
    print("="*60)
    print("🔍 DQN AGENT EVALUATION")
    print("="*60)
    
    # Create environment (disable wrappers for fair evaluation)
    render_mode = "rgb_array" if (render or save_gif) else None
    if use_curriculum or use_reward_shaping:
        env = create_enhanced_env(use_curriculum=use_curriculum, 
                                use_reward_shaping=use_reward_shaping)
        if render_mode:
            # For rendering, we need to create a separate env
            base_env = gym.make("MiniGrid-ObstructedMaze-Full-v1", render_mode=render_mode)
            env.env.env = base_env  # Replace the base environment
    else:
        env = gym.make("MiniGrid-ObstructedMaze-Full-v1", render_mode=render_mode)
    
    # Get observation and action spaces
    obs_shape = env.observation_space['image'].shape
    action_size = env.action_space.n
    
    print(f"\n📋 EVALUATION CONFIGURATION:")
    print(f"  Model: {model_path}")
    print(f"  Episodes: {num_episodes}")
    print(f"  Environment: MiniGrid-ObstructedMaze-Full-v1")
    print(f"  Reward shaping: {use_reward_shaping}")
    print(f"  Curriculum learning: {use_curriculum}")
    print(f"  Rendering: {render or save_gif}")
    
    # Create and load agent
    agent = DQNAgent(obs_shape, action_size)
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"✓ Model loaded successfully")
    else:
        print(f"❌ Model file not found: {model_path}")
        return None
    
    # Evaluation metrics
    scores = []
    episode_lengths = []
    original_rewards = []
    bonus_rewards = []
    successes = 0
    total_time = 0
    
    print(f"\n🚀 Starting evaluation...")
    
    for episode in range(num_episodes):
        start_time = time.time()
        obs, info = env.reset()
        state = preprocess_observation(obs)
        agent.reset_episode()
        
        total_reward = 0
        original_reward = 0
        bonus_reward = 0
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
            
            # Track different reward components if available
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
        
        # Progress reporting
        if episode % 10 == 0 or original_reward > 0:
            status = "SUCCESS" if original_reward > 0 else "FAILED"
            print(f"Episode {episode:3d}: {status:7s} | "
                  f"Reward: {total_reward:6.2f} | "
                  f"Steps: {steps:4d} | "
                  f"Time: {episode_time:.2f}s")
            
            if original_reward > 0:
                print(f"  🎉 Success! Original reward: {original_reward:.2f}, "
                      f"Bonus: {bonus_reward:.2f}")
        
        # Save GIF for first episode
        if save_gif and episode == 0 and frames:
            save_episode_gif(frames, 'enhanced_demo.gif')
    
    # Calculate final statistics
    avg_time_per_episode = total_time / num_episodes
    success_rate = successes / num_episodes
    avg_score = np.mean(scores)
    avg_length = np.mean(episode_lengths)
    avg_original = np.mean(original_rewards)
    avg_bonus = np.mean(bonus_rewards)
    
    print(f"\n" + "="*60)
    print("📊 EVALUATION RESULTS")
    print("="*60)
    print(f"Total Episodes: {num_episodes}")
    print(f"Successful Episodes: {successes}")
    print(f"Success Rate: {success_rate:.2%}")
    print(f"Average Total Reward: {avg_score:.3f} ± {np.std(scores):.3f}")
    print(f"Average Original Reward: {avg_original:.3f} ± {np.std(original_rewards):.3f}")
    if use_reward_shaping:
        print(f"Average Bonus Reward: {avg_bonus:.3f} ± {np.std(bonus_rewards):.3f}")
    print(f"Average Episode Length: {avg_length:.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Max Reward: {np.max(scores):.3f}")
    print(f"Min Reward: {np.min(scores):.3f}")
    print(f"Average Time per Episode: {avg_time_per_episode:.2f}s")
    print("="*60)
    
    # Performance analysis
    if success_rate > 0.5:
        print("🏆 EXCELLENT PERFORMANCE! Agent is solving the maze consistently.")
    elif success_rate > 0.2:
        print("✅ GOOD PERFORMANCE! Agent is learning to solve the maze.")
    elif success_rate > 0.05:
        print("🔄 MODERATE PERFORMANCE! Agent shows some learning but needs improvement.")
    else:
        print("⚠️  LOW PERFORMANCE! Agent needs more training or different approach.")
    
    # Plot evaluation results
    plot_enhanced_evaluation_results(scores, episode_lengths, original_rewards, 
                                   bonus_rewards, successes, num_episodes)
    
    env.close()
    
    return {
        'scores': scores,
        'episode_lengths': episode_lengths,
        'original_rewards': original_rewards,
        'bonus_rewards': bonus_rewards,
        'success_rate': success_rate,
        'avg_reward': avg_score,
        'avg_length': avg_length,
        'avg_time': avg_time_per_episode
    }


def save_episode_gif(frames, filename):
    """Save episode frames as GIF."""
    print(f"💾 Saving demonstration as {filename}...")
    imageio.mimsave(filename, frames, fps=8)
    print(f"✅ GIF saved: {filename}")


def plot_enhanced_evaluation_results(scores, episode_lengths, original_rewards, 
                                   bonus_rewards, successes, num_episodes):
    """Plot comprehensive evaluation metrics."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Reward distribution
    ax1.hist(scores, bins=30, alpha=0.7, edgecolor='black', color='skyblue')
    ax1.axvline(np.mean(scores), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(scores):.2f}')
    ax1.axvline(np.median(scores), color='green', linestyle='--', linewidth=2,
                label=f'Median: {np.median(scores):.2f}')
    ax1.set_title('Distribution of Total Rewards')
    ax1.set_xlabel('Total Reward')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Episode length distribution
    ax2.hist(episode_lengths, bins=30, alpha=0.7, edgecolor='black', color='orange')
    ax2.axvline(np.mean(episode_lengths), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(episode_lengths):.1f}')
    ax2.set_title('Distribution of Episode Lengths')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Success rate over time
    episode_range = range(len(scores))
    success_indicators = [1 if reward > 0 else 0 for reward in original_rewards]
    
    # Calculate rolling success rate (window of 20)
    rolling_success = []
    window_size = min(20, len(success_indicators))
    for i in range(len(success_indicators)):
        start_idx = max(0, i - window_size + 1)
        window_successes = success_indicators[start_idx:i+1]
        rolling_success.append(np.mean(window_successes))
    
    ax3.plot(episode_range, rolling_success, 'g-', linewidth=2, label=f'Rolling Success Rate')
    ax3.scatter(episode_range, success_indicators, alpha=0.5, s=10, 
               c=['red' if x == 0 else 'green' for x in success_indicators])
    ax3.set_title(f'Success Rate Over Episodes (Final: {successes}/{num_episodes} = {successes/num_episodes:.1%})')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Success Rate')
    ax3.set_ylim(-0.1, 1.1)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Reward components comparison (if bonus rewards exist)
    if any(bonus_rewards):
        episodes_with_bonus = [(orig, bonus, total) for orig, bonus, total in 
                             zip(original_rewards, bonus_rewards, scores)]
        
        success_episodes = [(orig, bonus, total) for orig, bonus, total in episodes_with_bonus 
                          if orig > 0]
        failed_episodes = [(orig, bonus, total) for orig, bonus, total in episodes_with_bonus 
                         if orig <= 0]
        
        if success_episodes:
            success_orig, success_bonus, success_total = zip(*success_episodes)
            ax4.scatter(success_orig, success_bonus, alpha=0.7, c='green', s=30, label='Successful')
        
        if failed_episodes:
            failed_orig, failed_bonus, failed_total = zip(*failed_episodes)
            ax4.scatter(failed_orig, failed_bonus, alpha=0.5, c='red', s=20, label='Failed')
        
        ax4.set_title('Original vs Bonus Rewards')
        ax4.set_xlabel('Original Reward')
        ax4.set_ylabel('Bonus Reward')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        # Just show reward over time
        ax4.plot(episode_range, scores, alpha=0.7, color='blue')
        ax4.set_title('Rewards Over Episodes')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Total Reward')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📈 Evaluation plots saved as: evaluation_results.png")


def demo_episode(model_path, render=True, save_gif=True, detailed=True):
    """Run a detailed demonstration episode with step-by-step analysis."""
    
    print("="*60)
    print("🎮 DQN AGENT DEMONSTRATION")
    print("="*60)
    
    # Create environment with rendering
    env_render_mode = "rgb_array" if save_gif else ("human" if render else None)
    env = create_enhanced_env(use_curriculum=False, use_reward_shaping=True)
    
    if env_render_mode:
        # We need to modify the base environment for rendering
        base_env = gym.make("MiniGrid-ObstructedMaze-Full-v1", render_mode=env_render_mode)
        # This is a bit hacky, but we need to access the base environment
        if hasattr(env, 'env'):
            if hasattr(env.env, 'env'):
                env.env.env = base_env
            else:
                env.env = base_env
        else:
            env = base_env
    
    # Get observation and action spaces
    obs_shape = env.observation_space['image'].shape
    action_size = env.action_space.n
    
    # Create and load agent
    agent = DQNAgent(obs_shape, action_size)
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"✅ Model loaded: {model_path}")
    else:
        print(f"❌ Model not found: {model_path}")
        return
    
    obs, info = env.reset()
    state = preprocess_observation(obs)
    agent.reset_episode()
    
    total_reward = 0
    original_reward = 0
    bonus_reward = 0
    steps = 0
    frames = []
    
    print(f"\n🚀 Starting demonstration...")
    print(f"Mission: {obs.get('mission', 'Find the blue ball')}")
    print(f"Initial position: {obs.get('agent_pos', 'Unknown')}")
    print(f"Initial direction: {obs.get('direction', 'Unknown')}")
    
    if save_gif:
        frames.append(env.render())
    
    action_names = ['Turn Left', 'Turn Right', 'Move Forward', 'Pick Up', 'Drop', 'Toggle/Open', 'Done']
    
    while True:
        # Choose action
        action = agent.act(state, training=False)
        action_name = action_names[action] if action < len(action_names) else f"Action_{action}"
        
        if detailed:
            print(f"\nStep {steps + 1}:")
            print(f"  Position: {state.get('agent_pos', 'Unknown')}")
            print(f"  Direction: {state.get('direction', 'Unknown')}")
            print(f"  Action: {action} ({action_name})")
        
        # Take action
        next_obs, reward, done, truncated, info = env.step(action)
        next_state = preprocess_observation(next_obs)
        
        if save_gif:
            frames.append(env.render())
        
        # Track rewards
        step_original = info.get('original_reward', reward)
        step_bonus = info.get('bonus_reward', 0)
        
        original_reward += step_original
        bonus_reward += step_bonus
        total_reward += reward
        steps += 1
        
        if detailed:
            print(f"  Reward: {reward:.3f} (Original: {step_original:.3f}, Bonus: {step_bonus:.3f})")
            print(f"  Total Reward: {total_reward:.3f}")
            if 'exploration_progress' in info:
                print(f"  Exploration: {info['exploration_progress']:.2%}")
        
        state = next_state
        
        if done or truncated:
            break
        
        # Stop if too many steps (safety)
        if steps >= 2000:
            print("  ⚠️  Stopping due to step limit")
            break
    
    # Final results
    success = original_reward > 0
    print(f"\n" + "="*60)
    print("🏁 DEMONSTRATION COMPLETED")
    print("="*60)
    print(f"Result: {'SUCCESS! 🎉' if success else 'FAILED ❌'}")
    print(f"Steps taken: {steps}")
    print(f"Original reward: {original_reward:.3f}")
    print(f"Bonus reward: {bonus_reward:.3f}")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Average reward per step: {total_reward/max(1, steps):.4f}")
    print("="*60)
    
    if save_gif and frames:
        gif_name = f"enhanced_demo_{'success' if success else 'failed'}.gif"
        save_episode_gif(frames, gif_name)
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate Enhanced DQN agent')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of evaluation episodes')
    parser.add_argument('--demo', action='store_true',
                        help='Run a single detailed demonstration')
    parser.add_argument('--save_gif', action='store_true',
                        help='Save demonstration as GIF')
    parser.add_argument('--render', action='store_true',
                        help='Render the environment')
    parser.add_argument('--use_curriculum', action='store_true',
                        help='Use curriculum learning wrapper during evaluation')
    parser.add_argument('--use_reward_shaping', action='store_true',
                        help='Use reward shaping wrapper during evaluation')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_episode(args.model_path, 
                            render=args.render, 
                            save_gif=args.save_gif)
    else:
        evaluate_agent(args.model_path, 
                              args.episodes, 
                              render=args.render,
                              save_gif=args.save_gif,
                              use_curriculum=args.use_curriculum,
                              use_reward_shaping=args.use_reward_shaping)


if __name__ == "__main__":
    main()

