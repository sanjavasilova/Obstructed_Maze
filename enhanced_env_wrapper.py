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
        self.keys_picked_count = 0  # Track total key pickups
        self.last_key_pickup_step = -100  # Prevent rapid pickup spam
        self.last_key_drop_step = -100
        self.balls_moved = set()  # Track which balls have been moved
        self.last_carrying = None  # Track what agent was carrying
        self.agent_positions = deque(maxlen=50)  # Last 50 positions
        self.action_history = deque(maxlen=50)
        self.objects_interacted = set()
        self.doors_opened = set()
        self.keys_picked = set()
        self.boxes_opened = set()
        
        # Reward shaping parameters - IMPROVED for better learning DQN
        # self.exploration_reward = 0.01       # Increased from 0.002
        # self.key_pickup_reward = 2.0         # Massively increased from 0.3 - keys are critical!
        # self.door_open_reward = 0.2          # 0.2
        # self.box_open_reward = 0.5           # Increased from 0.1
        # self.progress_reward = 0.02          # Doubled from 0.01
        # self.stuck_penalty = -0.01           # Reduced from -0.05 (was too harsh), then changed from -0.03 to -0.01
        # self.step_penalty = 0.0001           # Reduced from 0.005 (was discouraging movement), then reduced from 0.001
        # self.carrying_key_bonus = 0.005      # Small bonus each step while carrying key

        # Reward shaping parameters - FOR OBSTRUCTED MAZE PPO
        self.exploration_reward = 0.01
        self.door_open_reward = 0.2
        self.key_pickup_reward = 20.0  # Keys unlock doors
        self.box_open_reward = 3.0  # Boxes contain keys
        self.ball_pickup_reward = 2.0  # NEW: Picking up green balls
        self.ball_moved_reward = 3.0  # NEW: Dropping ball away from door namaleno od 5.0
        self.progress_toward_goal_reward = 0.05  # NEW: Moving toward blue ball
        self.progress_reward = 0.01  # ADD THIS - for general exploration progress
        self.stuck_penalty = -0.01
        self.step_penalty = 0.00005
        self.carrying_key_bonus = 0.02

        # Track carrying state
        self.carrying_key = False
        
        # Goal position (found on reset)
        self.goal_pos = None
        self.grid_size = 14  # Approximate grid size for normalization

        # Track game state
        self.initial_obs = None
        self.step_count = 0
        self.max_steps = 500  # Reasonable limit od 1000 na 500 pa na 100, go vrativ pak na 500 :)
        self.turn_in_place_streak = 0
        self.last_agent_pos = None
        self.spin_warning_streak = 0
        self.last_turn_action = None  # Track last turn direction (0 left, 1 right)
        self.turn_repeat_streak = 0   # How many identical turn actions in a row
        
        print("Enhanced ObstructedMaze Wrapper initialized with reward shaping:")
        print(f"  Exploration reward: {self.exploration_reward}")
        print(f"  Key pickup reward: {self.key_pickup_reward}")
        #print(f"  Door open reward: {self.door_open_reward}")
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
        self.keys_picked_count = 0
        self.last_key_pickup_step = -100
        self.last_key_drop_step = -100
        self.balls_moved.clear()
        self.last_carrying = None
        self.step_count = 0
        self.initial_obs = obs.copy()
        
        # Add agent position to tracking
        # Get agent position from environment state
        # agent_pos = tuple(self.env.agent_pos) if hasattr(self.env, 'agent_pos') else (0, 0)
        agent_pos = tuple(self.env.unwrapped.agent_pos) if hasattr(self.env.unwrapped, 'agent_pos') else (0, 0)
        self.agent_positions.append(agent_pos)
        self.last_agent_pos = agent_pos
        self.turn_in_place_streak = 0
        self.spin_warning_streak = 0
        self.last_turn_action = None
        self.turn_repeat_streak = 0
        self.carrying_key = False
        
        # Find goal position in the grid
        self._find_goal_position()
        
        # Enhanced observation with additional info
        enhanced_obs = self._enhance_observation(obs)
        
        return enhanced_obs, info
    
    def _find_goal_position(self):
        """Find the goal position in the full grid."""
        self.goal_pos = None
        if hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'grid'):
            grid = self.env.unwrapped.grid
            self.grid_size = max(grid.width, grid.height)
            for x in range(grid.width):
                for y in range(grid.height):
                    cell = grid.get(x, y)
                    if cell is not None and (cell.type == 'goal' or cell.type == 'ball'):
                        self.goal_pos = (x, y)
                        return
    
    def step(self, action):
        """Enhanced step with reward shaping."""
        obs, reward, done, truncated, info = self.env.step(action)
        
        self.step_count += 1
        # Track action for behavior diagnostics
        self.action_history.append(action)

        # Track repeated same-direction turns even if the agent is moving
        if action in (0, 1):  # turn left or right
            if self.last_turn_action == action:
                self.turn_repeat_streak += 1
            else:
                self.turn_repeat_streak = 1
            self.last_turn_action = action
        else:
            self.turn_repeat_streak = 0
            self.last_turn_action = None
        
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
        # Spin warning triggers when the agent repeats the same turn action several times,
        # even if it is moving through nearby cells (not only when stationary).
        spin_warning = self.turn_repeat_streak >= 3
        info.update({
            'bonus_reward': bonus_reward,
            'original_reward': reward,
            'step_count': self.step_count,
            'exploration_progress': len(set(self.agent_positions)) / max(1, len(self.agent_positions)),
            'turn_in_place_streak': self.turn_in_place_streak,
            'turn_repeat_streak': self.turn_repeat_streak,
            'spin_warning': spin_warning
        })

        # If spinning persists, truncate episode with a penalty to prevent endless loops
        if spin_warning:
            self.spin_warning_streak += 1
        else:
            self.spin_warning_streak = 0
        if self.spin_warning_streak >= 3:
            total_reward -= 1.0
            truncated = True
            info['spin_truncated'] = True
        
        return enhanced_obs, total_reward, done, truncated, info
    
    def _enhance_observation(self, obs):
        """Add compass information to observation - agent is no longer blind!"""
        enhanced_obs = obs.copy()
        
        # Add agent position to tracking
        # agent_pos = tuple(self.env.agent_pos) if hasattr(self.env, 'agent_pos') else (0, 0)
        agent_pos = tuple(self.env.unwrapped.agent_pos) if hasattr(self.env.unwrapped, 'agent_pos') else (0, 0)
        self.agent_positions.append(agent_pos)
        
        # COMPASS INFO: Where am I? (normalized position)
        agent_x, agent_y = agent_pos
        enhanced_obs['agent_pos'] = list(agent_pos)
        enhanced_obs['agent_pos_normalized'] = [
            agent_x / self.grid_size,  # Normalized x
            agent_y / self.grid_size   # Normalized y
        ]
        
        # COMPASS INFO: Where's the goal? (relative direction, normalized)
        if self.goal_pos is not None:
            goal_x, goal_y = self.goal_pos
            enhanced_obs['goal_direction'] = [
                (goal_x - agent_x) / self.grid_size,  # Relative x to goal
                (goal_y - agent_y) / self.grid_size   # Relative y to goal
            ]
        else:
            enhanced_obs['goal_direction'] = [0.0, 0.0]
        
        # Additional info
        enhanced_obs['step_count'] = self.step_count
        enhanced_obs['unique_positions'] = len(set(self.agent_positions))

        # Add carrying information for action masking
        carrying = None
        if hasattr(self.env, 'unwrapped'):
            carrying = self.env.unwrapped.carrying
        enhanced_obs['carrying'] = carrying
        
        return enhanced_obs
    
    def _calculate_bonus_reward(self, obs, action, reward, done):
        """Calculate bonus reward based on agent progress and exploration."""
        bonus = 0.0
        
        # agent_pos = tuple(self.env.agent_pos) if hasattr(self.env, 'agent_pos') else (0, 0)
        agent_pos = tuple(self.env.unwrapped.agent_pos) if hasattr(self.env.unwrapped, 'agent_pos') else (0, 0)
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
        
        # Generic movement bonus: reward any position change
        if moved:
            bonus += 0.005
        
        # 1. Exploration bonus for visiting new positions
        if moved and agent_pos not in set(list(self.agent_positions)[:-1]):  # New position
            bonus += self.exploration_reward
            bonus += 0.02  # Extra bonus for new cell discovery
        
        # 2. Check for stuck behavior (oscillating between positions)
        if len(self.agent_positions) >= 4:
            recent_positions = list(self.agent_positions)[-4:]
            if len(set(recent_positions)) <= 2:  # Oscillating between 2 positions
                bonus += self.stuck_penalty
        
        # 2b. Penalize sustained turning-in-place (reduced penalties)
        if self.turn_in_place_streak >= 3:
            bonus -= 0.2
        if self.turn_in_place_streak >= 8:
            bonus -= 0.3
            self.turn_in_place_streak = 0
        
        # 3. Analyze the grid to detect interactions
        grid = obs['image']
        
        # 4. Reward for picking up objects (keys) - CRITICAL for maze solving!
        if action == 3:  # pickup action
            # front_pos = self._get_front_position(agent_pos, agent_dir)
            # if self._is_valid_position(front_pos, grid.shape):
            #     front_cell = grid[front_pos[1], front_pos[0]]
            #     if self._is_key(front_cell) and front_pos not in self.keys_picked:
            #         bonus += self.key_pickup_reward
            #         self.keys_picked.add(front_pos)
            #         self.carrying_key = True
            #         print(f"🔑 KEY PICKED UP at step {self.step_count}!")

            front_pos = self._get_front_position(agent_pos, agent_dir)
            if self._is_valid_position(front_pos, grid.shape):
                front_cell = grid[front_pos[1], front_pos[0]]

                if self._is_key(front_cell):
                    # Cooldown period: only reward if enough steps passed since last pickup
                    steps_since_last = self.step_count - self.last_key_pickup_step

                    if steps_since_last > 50:  # At least 50 steps between pickups
                        self.keys_picked_count += 1

                        # Diminishing returns: first pickup = full reward, later = less
                        if self.keys_picked_count == 1:
                            bonus += self.key_pickup_reward  # Full reward
                            print(f"🔑 KEY PICKED UP at step {self.step_count}!")
                        elif self.keys_picked_count == 2:
                            bonus += self.key_pickup_reward * 0.5  # Half reward
                            print(f"🔑 KEY PICKED UP AGAIN at step {self.step_count}!")
                        else:
                            bonus += 0.1  # Tiny reward after that

                        self.last_key_pickup_step = self.step_count
                        self.carrying_key = True
                    else:
                        # Too soon - no reward (prevents spam)
                        self.carrying_key = True

                # 4b. NEW: Reward for picking up green balls (to move them)
                elif len(front_cell) >= 1 and front_cell[0] == 8:  # Ball object
                    # Check if it's green (not blue goal) - blue has color code 1
                    if len(front_cell) >= 2 and front_cell[1] != 1:  # Not blue
                        bonus += self.ball_pickup_reward
                        print(f"🟢 GREEN BALL PICKED UP at step {self.step_count}!")

        # 4c. Reward for strategically dropping objects
        if action == 4:  # drop action
            # Get current agent position first
            agentpos = tuple(self.env.unwrapped.agent_pos) if hasattr(self.env.unwrapped, 'agent_pos') else (0, 0)
            if hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'carrying'):
                carrying = self.env.unwrapped.carrying
                if carrying is not None:
                    if hasattr(carrying, 'type') and carrying.type == 'ball':
                        if agentpos not in self.balls_moved:
                            bonus += self.ball_moved_reward  # Keep this - moving balls is good
                            self.balls_moved.add(agentpos)
                            print(f"📍 BALL MOVED OUT OF WAY at step {self.step_count}!")
                        else:
                            bonus -= 0.5  # Penalty for dropping ball again
                            print(f"📍 BALL RE-DROPPED (PENALTY -0.5) at step {self.step_count}!")

                    elif hasattr(carrying, 'type') and carrying.type == 'key':
                        bonus -= 5.0  # STRONG penalty - keys should NOT be dropped!
                        print(f"🔑 KEY DROPPED (PENALTY -5.0) at step {self.step_count}")
                    else:
                        bonus -= 2.0  # Strong penalty for dropping other objects
                        print(f"📍 OBJECT DROPPED (PENALTY -2.0) at step {self.step_count}!")

        # Check if agent is carrying a key (check environment state)
        if hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'carrying'):
            if self.env.unwrapped.carrying is not None:
                self.carrying_key = True
                # Give small continuous bonus for carrying key
                bonus += self.carrying_key_bonus
            else:
                self.carrying_key = False
        
        # 5. Reward for opening doors - CRITICAL for maze solving!
        if action == 5:  # toggle action
            front_pos = self._get_front_position(agent_pos, agent_dir)
            if self._is_valid_position(front_pos, grid.shape):
                front_cell = grid[front_pos[1], front_pos[0]]
                # if self._is_door(front_cell) and front_pos not in self.doors_opened:
                #     bonus += self.door_open_reward
                #     self.doors_opened.add(front_pos)
                #     print(f"🚪 DOOR OPENED at step {self.step_count}!")
                if self._is_door(front_cell) and front_pos not in self.doors_opened:
                    # Check if door was locked and agent has key
                    door_obj = self.env.unwrapped.grid.get(*front_pos)
                    carrying = self.env.unwrapped.carrying

                    was_locked = door_obj and getattr(door_obj, 'is_locked', False)
                    has_key = carrying and getattr(carrying, 'type', None) == 'key'

                    if was_locked and has_key:
                        bonus += 50.0  # HUGE - unlocked with key!
                        print(f"🚪🔑 LOCKED DOOR UNLOCKED WITH KEY at step {self.step_count}!")
                    else:
                        # Regular door (unlocked) - no reward, just message
                        print(f"🚪 DOOR OPENED at step {self.step_count}!")

                    self.doors_opened.add(front_pos)

                elif self._is_box(front_cell) and front_pos not in self.boxes_opened:
                    bonus += self.box_open_reward
                    self.boxes_opened.add(front_pos)
                    print(f"📦 BOX OPENED at step {self.step_count}!")
        
        # 6. Progress reward for moving towards unexplored areas
        if len(self.agent_positions) > 0:
            prev_pos_for_progress = self.agent_positions[-1]
            if agent_pos != prev_pos_for_progress and self._is_moving_towards_exploration(agent_pos, prev_pos_for_progress):
                bonus += self.progress_reward
        
        # 6b. Encourage forward movement that changes position
        if is_forward and moved:
            if agent_pos not in set(self.agent_positions):
                bonus += 0.03
            else:
                bonus += 0.01
        elif is_forward and not moved:
            # Reduced penalty for hitting obstacles (was too harsh)
            bonus -= 0.01
            
        # 6c. Penalize excessive turning without position change
        if len(self.action_history) >= 8:
            recent_actions = list(self.action_history)[-8:]
            recent_turn_ratio = sum(1 for a in recent_actions if a in (0, 1)) / 8.0
            recent_positions = list(self.agent_positions)[-8:] if len(self.agent_positions) >= 8 else list(self.agent_positions)
            if recent_turn_ratio >= 0.75 and len(set(recent_positions)) <= 2:
                bonus -= 0.03

        # 6d. Penalize taking 'done' without task completion
        if action == 6 and not done:
            bonus -= 0.05
        
        # 7. Reward for seeing important objects (encourages investigation)
        visible_objects = self._scan_visible_objects(grid)
        if visible_objects['key_visible'] and not self.carrying_key:
            bonus += 0.005  # Small bonus for seeing a key
        if visible_objects['door_visible'] and self.carrying_key:
            bonus += 0.005  # Small bonus for seeing a door while carrying key
        if visible_objects['goal_visible']:
            bonus += 0.01  # Bonus for seeing the goal
        # 7b. NEW: Extra reward for getting closer to the blue ball (goal)
        if self.goal_pos is not None:
            current_dist = abs(agent_pos[0] - self.goal_pos[0]) + abs(agent_pos[1] - self.goal_pos[1])
            prev_dist = abs(prev_pos[0] - self.goal_pos[0]) + abs(prev_pos[1] - self.goal_pos[1])

            if current_dist < prev_dist:  # Moving toward goal
                bonus += self.progress_toward_goal_reward

        # 8. Bonus for successful completion
        if reward > 0:  # Original success reward
            # Scale bonus based on how fast the agent completed
            efficiency_bonus = max(10.0, 200.0 - self.step_count * 0.1)
            bonus += efficiency_bonus
            print(f"🎉 SUCCESS! Agent completed the maze in {self.step_count} steps! Bonus: {efficiency_bonus:.1f}")
        
        return bonus
    
    def _scan_visible_objects(self, grid):
        """Scan the visible 7x7 grid for important objects."""
        result = {
            'key_visible': False,
            'door_visible': False,
            'goal_visible': False,
            'box_visible': False
        }
        
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                cell = grid[y, x]
                if cell[0] == 5:  # Key
                    result['key_visible'] = True
                elif cell[0] == 4:  # Door
                    result['door_visible'] = True
                elif cell[0] == 8:  # Goal/Ball
                    result['goal_visible'] = True
                elif cell[0] == 7:  # Box
                    result['box_visible'] = True
        
        return result
    
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
            base_steps = 100
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
