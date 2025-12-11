import gymnasium as gym
import numpy as np
from collections import deque


class ObstructedMazeWrapper(gym.Wrapper):
    """
    Enhanced wrapper for ObstructedMaze-Full with reward shaping and progress tracking.
    This wrapper adds intermediate rewards to help the agent learn more effectively.
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # Track agent progress
        self.agent_positions = deque(maxlen=50)  # Last 50 positions
        self.action_history = deque(maxlen=50)
        self.objects_interacted = set()
        self.doors_opened = set()
        self.keys_picked = set()
        self.boxes_opened = set()
        
        # Reward shaping parameters
        self.exploration_reward = 0.002
        self.key_pickup_reward = 0.3
        self.door_open_reward = 0.2
        self.box_open_reward = 0.1
        self.progress_reward = 0.01
        self.stuck_penalty = -0.05
        self.step_penalty = 0.02

        # Track game state
        self.initial_obs = None
        self.step_count = 0
        self.max_steps = 1000  # Reasonable limit
        self.turn_in_place_streak = 0
        self.last_agent_pos = None
        
        print("Enhanced ObstructedMaze Wrapper initialized with reward shaping:")
        print(f"  Exploration reward: {self.exploration_reward}")
        print(f"  Key pickup reward: {self.key_pickup_reward}")
        print(f"  Door open reward: {self.door_open_reward}")
        print(f"  Box open reward: {self.box_open_reward}")
        print(f"  Max steps: {self.max_steps}")
    
    def reset(self, **kwargs):
        """Reset the environment and tracking variables."""
        obs, info = self.env.reset(**kwargs)
        
        # Reset tracking
        self.agent_positions.clear()
        self.action_history.clear()
        self.objects_interacted.clear()
        self.doors_opened.clear()
        self.keys_picked.clear()
        self.boxes_opened.clear()
        self.step_count = 0
        self.initial_obs = obs.copy()
        
        # Add agent position to tracking
        # Get agent position from environment state
        agent_pos = tuple(self.env.agent_pos) if hasattr(self.env, 'agent_pos') else (0, 0)
        self.agent_positions.append(agent_pos)
        self.last_agent_pos = agent_pos
        self.turn_in_place_streak = 0
        
        # Enhanced observation with additional info
        enhanced_obs = self._enhance_observation(obs)
        
        return enhanced_obs, info
    
    def step(self, action):
        """Enhanced step with reward shaping."""
        obs, reward, done, truncated, info = self.env.step(action)
        
        self.step_count += 1
        # Track action for behavior diagnostics
        self.action_history.append(action)
        
        # Calculate shaped reward
        shaped_reward = reward  # Original reward (1 for success, 0 otherwise)
        bonus_reward = self._calculate_bonus_reward(obs, action, reward, done)
        total_reward = shaped_reward + bonus_reward
        
        # Add step penalty to encourage efficiency
        total_reward -= self.step_penalty  # Uses variable
        
        # Enhanced observation
        enhanced_obs = self._enhance_observation(obs)
        
        # Early truncation if agent gets stuck or takes too long
        if self.step_count >= self.max_steps:
            truncated = True
        
        # Add bonus info
        # Compute recent spin statistics
        recent_actions = list(self.action_history)[-20:]
        turn_count = sum(1 for a in recent_actions if a in (0, 1)) if recent_actions else 0
        turn_ratio_20 = turn_count / max(1, len(recent_actions))
        recent_positions = list(self.agent_positions)[-20:]
        unique_recent_positions = len(set(recent_positions))
        spin_warning = turn_ratio_20 > 0.7 and unique_recent_positions <= 2
        info.update({
            'bonus_reward': bonus_reward,
            'original_reward': reward,
            'step_count': self.step_count,
            'exploration_progress': len(set(self.agent_positions)) / max(1, len(self.agent_positions)),
            'turn_in_place_streak': self.turn_in_place_streak,
            'turn_ratio_20': turn_ratio_20,
            'spin_warning': spin_warning
        })
        
        return enhanced_obs, total_reward, done, truncated, info
    
    def _enhance_observation(self, obs):
        """Add additional information to observation."""
        enhanced_obs = obs.copy()
        
        # Add agent position to tracking
        agent_pos = tuple(self.env.agent_pos) if hasattr(self.env, 'agent_pos') else (0, 0)
        self.agent_positions.append(agent_pos)
        
        # Add exploration statistics
        enhanced_obs['agent_pos'] = list(agent_pos)
        enhanced_obs['step_count'] = self.step_count
        enhanced_obs['unique_positions'] = len(set(self.agent_positions))
        
        return enhanced_obs
    
    def _calculate_bonus_reward(self, obs, action, reward, done):
        """Calculate bonus reward based on agent progress and exploration."""
        bonus = 0.0
        
        agent_pos = tuple(self.env.agent_pos) if hasattr(self.env, 'agent_pos') else (0, 0)
        agent_dir = obs['direction']
        prev_pos = self.agent_positions[-1] if len(self.agent_positions) > 0 else agent_pos
        moved = agent_pos != prev_pos
        is_turn = action in (0, 1)
        is_forward = action == 2
        
        # Track spinning in place
        if is_turn and not moved:
            self.turn_in_place_streak += 1
        else:
            # reset streak when moving forward or changing position
            if moved or is_forward:
                self.turn_in_place_streak = 0
        
        # 1. Exploration bonus for visiting new positions
        if moved and agent_pos not in set(list(self.agent_positions)[:-1]):  # New position
            bonus += self.exploration_reward
        
        # 2. Check for stuck behavior (oscillating between positions)
        if len(self.agent_positions) >= 4:
            recent_positions = list(self.agent_positions)[-4:]
            if len(set(recent_positions)) <= 2:  # Oscillating between 2 positions
                bonus += self.stuck_penalty
        
        # 2b. Penalize sustained turning-in-place
        if self.turn_in_place_streak >= 2:
            bonus -= 0.05 * min(10, self.turn_in_place_streak - 1)
        
        # 3. Analyze the grid to detect interactions
        grid = obs['image']
        
        # 4. Reward for picking up objects (keys)
        if action == 3:  # pickup action
            # Check if there's a key in front of the agent
            front_pos = self._get_front_position(agent_pos, agent_dir)
            if self._is_valid_position(front_pos, grid.shape):
                front_cell = grid[front_pos[1], front_pos[0]]
                if self._is_key(front_cell) and front_pos not in self.keys_picked:
                    bonus += self.key_pickup_reward
                    self.keys_picked.add(front_pos)
        
        # 5. Reward for opening doors
        if action == 5:  # toggle action
            front_pos = self._get_front_position(agent_pos, agent_dir)
            if self._is_valid_position(front_pos, grid.shape):
                front_cell = grid[front_pos[1], front_pos[0]]
                if self._is_door(front_cell) and front_pos not in self.doors_opened:
                    bonus += self.door_open_reward
                    self.doors_opened.add(front_pos)
                elif self._is_box(front_cell) and front_pos not in self.boxes_opened:
                    bonus += self.box_open_reward
                    self.boxes_opened.add(front_pos)
        
        # 6. Progress reward for moving towards unexplored areas (only if actually moved)
        if len(self.agent_positions) > 0:
            prev_pos_for_progress = self.agent_positions[-1]
            if agent_pos != prev_pos_for_progress and self._is_moving_towards_exploration(agent_pos, prev_pos_for_progress):
                bonus += self.progress_reward
        
        # 6b. Encourage forward movement that changes position
        if is_forward and moved:
            # Extra bonus if this is a newly visited cell
            # if agent_pos not in set(self.agent_positions):
            #     bonus += 0.03
            # else:
            #     bonus += 0.01
            pass
        
        # 6c. Penalize excessive turning without position change over a short window
        if len(self.action_history) >= 8:
            recent_actions = list(self.action_history)[-8:]
            recent_turn_ratio = sum(1 for a in recent_actions if a in (0, 1)) / 8.0
            recent_positions = list(self.agent_positions)[-8:] if len(self.agent_positions) >= 8 else list(self.agent_positions)
            if recent_turn_ratio >= 0.75 and len(set(recent_positions)) <= 2:
                bonus -= 0.05
        
        # 7. Bonus for successful completion
        if reward > 0:  # Original success reward
            bonus += 50.0  # Additional bonus for finding the ball
            print(f"🎉 SUCCESS! Agent completed the maze in {self.step_count} steps!")
        
        return bonus
    
    def _get_front_position(self, pos, direction):
        """Get the position in front of the agent based on direction."""
        x, y = pos
        if direction == 0:  # Right
            return (x + 1, y)
        elif direction == 1:  # Down
            return (x, y + 1)
        elif direction == 2:  # Left
            return (x - 1, y)
        elif direction == 3:  # Up
            return (x, y - 1)
        return pos
    
    def _is_valid_position(self, pos, grid_shape):
        """Check if position is within grid bounds."""
        x, y = pos
        return 0 <= x < grid_shape[1] and 0 <= y < grid_shape[0]
    
    def _is_key(self, cell):
        """Check if cell contains a key (yellow object)."""
        # In MiniGrid, keys are typically encoded as [5, yellow_color, 0]
        return len(cell) >= 2 and cell[0] == 5  # Key object type
    
    def _is_door(self, cell):
        """Check if cell contains a door."""
        # In MiniGrid, doors are typically encoded as [4, color, state]
        return len(cell) >= 1 and cell[0] == 4  # Door object type
    
    def _is_box(self, cell):
        """Check if cell contains a box."""
        # In MiniGrid, boxes are typically encoded as [7, color, state]
        return len(cell) >= 1 and cell[0] == 7  # Box object type
    
    def _is_moving_towards_exploration(self, current_pos, prev_pos):
        """Check if agent is moving towards less explored areas."""
        # Simple heuristic: reward movement away from frequently visited areas
        position_counts = {}
        for pos in self.agent_positions:
            position_counts[pos] = position_counts.get(pos, 0) + 1
        
        current_visits = position_counts.get(current_pos, 0)
        prev_visits = position_counts.get(prev_pos, 0)
        
        return current_visits <= prev_visits


class CurriculumLearningWrapper(gym.Wrapper):
    """
    Wrapper that implements curriculum learning by gradually increasing task difficulty.
    """
    
    def __init__(self, env, start_difficulty=0.5, max_difficulty=1.0, difficulty_increment=0.01):
        super().__init__(env)
        self.current_difficulty = start_difficulty
        self.max_difficulty = max_difficulty
        self.difficulty_increment = difficulty_increment
        self.episodes_completed = 0
        self.success_rate = deque(maxlen=100)
        
        print(f"Curriculum Learning initialized:")
        print(f"  Start difficulty: {start_difficulty}")
        print(f"  Max difficulty: {max_difficulty}")
        print(f"  Increment: {difficulty_increment}")
    
    def reset(self, **kwargs):
        """Reset with current difficulty level."""
        obs, info = self.env.reset(**kwargs)
        
        # Adjust max steps based on difficulty
        if hasattr(self.env, 'max_steps'):
            base_steps = 1000
            self.env.max_steps = int(base_steps * (0.5 + 0.5 * self.current_difficulty))
        
        info['curriculum_difficulty'] = self.current_difficulty
        return obs, info
    
    def step(self, action):
        """Step with curriculum learning adjustments."""
        obs, reward, done, truncated, info = self.env.step(action)
        
        if done or truncated:
            self.episodes_completed += 1
            original_reward = info.get('original_reward', reward)
            self.success_rate.append(1 if original_reward > 0 else 0)
            
            # Increase difficulty if success rate is good
            if len(self.success_rate) >= 50:
                recent_success_rate = np.mean(list(self.success_rate)[-50:])
                if recent_success_rate > 0.7 and  self.current_difficulty < self.max_difficulty:
                    self.current_difficulty = min(
                        self.max_difficulty, 
                        self.current_difficulty + self.difficulty_increment
                    )
                    print(f"Curriculum difficulty increased to: {self.current_difficulty:.3f}")
                    # Decrease difficulty if struggling (<5% success)
                elif recent_success_rate < 0.05 and self.current_difficulty > 0.3:
                    self.current_difficulty = max(0.3, self.current_difficulty - self.difficulty_increment * 2)
                    print(f"Curriculum difficulty decreased to: {self.current_difficulty:.3f}")
        
        info['curriculum_difficulty'] = self.current_difficulty
        info['success_rate'] = np.mean(self.success_rate) if self.success_rate else 0.0
        
        return obs, reward, done, truncated, info


def create_enhanced_env(env_id="MiniGrid-ObstructedMaze-Full-v1", 
                       use_curriculum=True, 
                       use_reward_shaping=True):
    """Create an enhanced environment with all wrappers."""
    
    # Create base environment
    env = gym.make(env_id)
    
    # Apply reward shaping wrapper
    if use_reward_shaping:
        env = ObstructedMazeWrapper(env)
        print("✓ Applied reward shaping wrapper")
    
    # Apply curriculum learning wrapper
    if use_curriculum:
        env = CurriculumLearningWrapper(env)
        print("✓ Applied curriculum learning wrapper")
    
    return env


if __name__ == "__main__":
    # Test the enhanced environment
    print("Testing Enhanced Environment Wrapper...")
    
    env = create_enhanced_env()
    obs, info = env.reset()
    
    print(f"Initial observation keys: {obs.keys()}")
    print(f"Initial info: {info}")
    
    # Take a few random steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        print(f"Step {i+1}: action={action}, reward={reward:.3f}, "
              f"bonus={info.get('bonus_reward', 0):.3f}")
        
        if done or truncated:
            break
    
    env.close()
    print("Environment test completed!")
