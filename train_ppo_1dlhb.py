"""
PPO Training for MiniGrid-ObstructedMaze-1Dlhb-v0

This file implements a task-progression reward shaping system that guides
the agent through the correct sequence of actions:

1. Find and pick up the green ball
2. Drop the green ball aside
3. Find and open the box
4. Pick up the key from the box
5. Unlock the door with the key
6. Drop the key
7. Pick up the blue ball (SUCCESS)

The reward shaping uses a state machine to track progress and provides
incremental rewards for correct actions while penalizing useless behavior.
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from enum import IntEnum
from tqdm import tqdm
import os
from datetime import datetime
import time


# =============================================================================
# TASK PROGRESSION STATE MACHINE
# =============================================================================

class TaskStage(IntEnum):
    """Stages of task completion in correct order."""
    START = 0
    FOUND_GREEN_BALL = 1
    PICKED_GREEN_BALL = 2
    DROPPED_GREEN_BALL = 3
    FOUND_BOX = 4
    OPENED_BOX = 5
    PICKED_KEY = 6
    UNLOCKED_DOOR = 7
    DROPPED_KEY = 8
    PICKED_BLUE_BALL = 9  # SUCCESS


class TaskProgressionWrapper(gym.Wrapper):
    """
    Reward shaping wrapper that tracks task progression through a state machine.

    Provides rewards for:
    - Advancing to the next task stage (positive, scaled by importance)
    - Exploration of new positions (small positive)
    - Moving toward relevant objectives (small positive)

    Provides penalties for:
    - Regressing to earlier stages (negative)
    - Repeated useless actions (negative)
    - Wandering without progress (small negative)
    - Taking too many steps (tiny negative per step)
    """

    # Stage completion rewards - EXTREMELY weighted toward later stages
    STAGE_REWARDS = {
        TaskStage.FOUND_GREEN_BALL: 0.1,
        TaskStage.PICKED_GREEN_BALL: 0.3,      # Tiny - just a stepping stone
        TaskStage.DROPPED_GREEN_BALL: 0.3,     # Tiny - just a stepping stone
        TaskStage.FOUND_BOX: 1.0,              # Increased - finding box is important
        TaskStage.OPENED_BOX: 15.0,            # MASSIVE - this is the bottleneck!
        TaskStage.PICKED_KEY: 25.0,            # HUGE - critical step
        TaskStage.UNLOCKED_DOOR: 50.0,         # MASSIVE - major achievement
        TaskStage.DROPPED_KEY: 5.0,
        TaskStage.PICKED_BLUE_BALL: 200.0,     # ENORMOUS - final goal
    }

    def __init__(self, env):
        super().__init__(env)
        self.reset_tracking()

        # Configuration
        self.exploration_reward = 0.01
        self.progress_reward = 0.02
        self.stuck_penalty = -0.02
        self.step_penalty = -0.0001
        self.regression_penalty = -2.0
        self.useless_action_penalty = -0.1
        self.max_steps = 250  # Balanced - enough to explore but not too long

        # Object type codes in MiniGrid
        self.OBJ_BALL = 6
        self.OBJ_KEY = 5
        self.OBJ_DOOR = 4
        self.OBJ_BOX = 7

        # Color codes
        self.COLOR_GREEN = 1
        self.COLOR_BLUE = 0
        self.COLOR_YELLOW = 4

        print("TaskProgressionWrapper initialized for ObstructedMaze-1Dlhb")
        print(f"  Max steps: {self.max_steps}")
        print(f"  Stage rewards: {dict(self.STAGE_REWARDS)}")

    def reset_tracking(self):
        """Reset all tracking variables."""
        self.current_stage = TaskStage.START
        self.step_count = 0
        self.visited_positions = set()
        self.last_position = None
        self.last_carrying = None
        self.action_history = deque(maxlen=20)
        self.green_ball_pos = None
        self.box_pos = None
        self.key_pos = None
        self.door_pos = None
        self.blue_ball_pos = None
        self.green_ball_dropped = False
        self.box_opened = False
        self.door_unlocked = False

        # Track repeated interactions to prevent exploitation
        self.green_ball_pickup_count = 0
        self.green_ball_drop_count = 0
        self.key_pickup_count = 0
        self.steps_since_stage_change = 0

    def reset(self, **kwargs):
        """Reset environment and tracking."""
        obs, info = self.env.reset(**kwargs)
        self.reset_tracking()

        # Scan the full grid for object positions
        self._scan_grid_for_objects()

        # Enhance observation
        enhanced_obs = self._enhance_observation(obs)

        info['task_stage'] = int(self.current_stage)
        info['stage_name'] = self.current_stage.name

        return enhanced_obs, info

    def _scan_grid_for_objects(self):
        """Scan the environment grid to find all important objects."""
        if not hasattr(self.env, 'unwrapped') or not hasattr(self.env.unwrapped, 'grid'):
            return

        grid = self.env.unwrapped.grid
        for x in range(grid.width):
            for y in range(grid.height):
                cell = grid.get(x, y)
                if cell is None:
                    continue

                obj_type = cell.type
                obj_color = getattr(cell, 'color', None)

                if obj_type == 'ball':
                    if obj_color == 'green':
                        self.green_ball_pos = (x, y)
                    elif obj_color == 'blue':
                        self.blue_ball_pos = (x, y)
                elif obj_type == 'box':
                    self.box_pos = (x, y)
                elif obj_type == 'door':
                    self.door_pos = (x, y)
                elif obj_type == 'key':
                    self.key_pos = (x, y)  # Track key position too

    def _enhance_observation(self, obs):
        """Add compass and task info to observation."""
        enhanced = obs.copy()

        # Get agent position
        agent_pos = tuple(self.env.unwrapped.agent_pos)
        grid_size = max(self.env.unwrapped.grid.width, self.env.unwrapped.grid.height)

        enhanced['agent_pos'] = list(agent_pos)
        enhanced['agent_pos_normalized'] = [
            agent_pos[0] / grid_size,
            agent_pos[1] / grid_size
        ]

        # Goal direction (blue ball or current objective)
        target = self._get_current_target()
        if target:
            enhanced['goal_direction'] = [
                (target[0] - agent_pos[0]) / grid_size,
                (target[1] - agent_pos[1]) / grid_size
            ]
        else:
            enhanced['goal_direction'] = [0.0, 0.0]

        # Add carrying info for action masking
        enhanced['carrying'] = self.env.unwrapped.carrying

        return enhanced

    def _get_current_target(self):
        """Get the position of the current objective based on task stage."""
        if self.current_stage < TaskStage.DROPPED_GREEN_BALL:
            return self.green_ball_pos
        elif self.current_stage < TaskStage.OPENED_BOX:
            return self.box_pos
        elif self.current_stage < TaskStage.UNLOCKED_DOOR:
            return self.door_pos
        else:
            return self.blue_ball_pos

    def step(self, action):
        """Execute action and calculate shaped reward."""
        # Capture pre-action state
        pre_carrying = self.env.unwrapped.carrying
        pre_stage = self.current_stage

        # Execute action
        obs, env_reward, done, truncated, info = self.env.step(action)

        self.step_count += 1
        self.action_history.append(action)

        # Get post-action state
        post_carrying = self.env.unwrapped.carrying
        agent_pos = tuple(self.env.unwrapped.agent_pos)
        agent_dir = self.env.unwrapped.agent_dir

        # Calculate shaped reward
        shaped_reward = self._calculate_shaped_reward(
            action, env_reward, done,
            pre_carrying, post_carrying,
            agent_pos, agent_dir, obs
        )

        # Update visited positions
        if agent_pos not in self.visited_positions:
            self.visited_positions.add(agent_pos)
            shaped_reward += self.exploration_reward

        # Step penalty
        shaped_reward += self.step_penalty

        # Track steps since last stage change
        self.steps_since_stage_change += 1

        # Stagnation penalty - if stuck at same stage for too long
        if self.steps_since_stage_change > 50:
            # Much stronger penalty after green ball is dropped - force exploration!
            if self.green_ball_dropped and not self.box_opened:
                stagnation_penalty = -0.02 * (self.steps_since_stage_change - 50) / 50
                shaped_reward += max(stagnation_penalty, -0.3)  # Higher cap for box finding phase
            else:
                stagnation_penalty = -0.01 * (self.steps_since_stage_change - 50) / 50
                shaped_reward += max(stagnation_penalty, -0.1)

        # Check for truncation
        if self.step_count >= self.max_steps:
            truncated = True

        # Enhance observation
        enhanced_obs = self._enhance_observation(obs)

        # Update info
        info['task_stage'] = int(self.current_stage)
        info['stage_name'] = self.current_stage.name
        info['shaped_reward'] = shaped_reward
        info['original_reward'] = env_reward
        info['step_count'] = self.step_count
        info['stage_advanced'] = self.current_stage > pre_stage

        # Track last position
        self.last_position = agent_pos
        self.last_carrying = post_carrying

        # Total reward
        total_reward = env_reward + shaped_reward

        return enhanced_obs, total_reward, done, truncated, info

    def _calculate_shaped_reward(self, action, env_reward, done,
                                  pre_carrying, post_carrying,
                                  agent_pos, agent_dir, obs):
        """Calculate the shaped reward based on task progression."""
        reward = 0.0

        # Get front position (for interaction detection)
        front_pos = self._get_front_pos(agent_pos, agent_dir)

        # === PICKUP ACTION (action 3) ===
        if action == 3:
            picked_up = (pre_carrying is None and post_carrying is not None)

            if picked_up:
                obj_type = getattr(post_carrying, 'type', None)
                obj_color = getattr(post_carrying, 'color', None)

                # Picked up green ball
                if obj_type == 'ball' and obj_color == 'green':
                    self.green_ball_pickup_count += 1

                    if self.green_ball_pickup_count == 1 and not self.green_ball_dropped:
                        # First pickup - this is correct!
                        self.current_stage = TaskStage.PICKED_GREEN_BALL
                        self.steps_since_stage_change = 0
                        reward += self.STAGE_REWARDS[TaskStage.PICKED_GREEN_BALL]
                        print(f"  [Stage {self.current_stage}] Picked up GREEN BALL! +{self.STAGE_REWARDS[TaskStage.PICKED_GREEN_BALL]}")
                    else:
                        # Picking green ball AGAIN - MASSIVE penalty
                        penalty = -10.0 * self.green_ball_pickup_count  # Escalating penalty
                        reward += penalty
                        print(f"  [EXPLOIT] Re-picked green ball ({self.green_ball_pickup_count}x)! {penalty}")

                # Picked up key
                elif obj_type == 'key':
                    self.key_pickup_count += 1

                    # In this environment, key may be accessible without opening box
                    # Just require green ball to be handled first
                    if self.green_ball_dropped and self.key_pickup_count == 1:
                        # First key pickup after green ball - correct!
                        self.current_stage = TaskStage.PICKED_KEY
                        self.steps_since_stage_change = 0
                        reward += self.STAGE_REWARDS[TaskStage.PICKED_KEY]
                        print(f"  [Stage {self.current_stage}] Picked up KEY! +{self.STAGE_REWARDS[TaskStage.PICKED_KEY]}")
                    elif not self.green_ball_dropped:
                        # Haven't handled green ball yet - BIG penalty
                        reward += -10.0
                        print(f"  [WRONG ORDER] Picked key before handling green ball! -10.0")
                    elif self.key_pickup_count > 1:
                        # Re-picking key - penalty
                        reward += -5.0
                        print(f"  [EXPLOIT] Re-picked key ({self.key_pickup_count}x)! -5.0")

                # Picked up blue ball - SUCCESS!
                elif obj_type == 'ball' and obj_color == 'blue':
                    if self.door_unlocked:
                        # Door is unlocked, can pick up blue ball - SUCCESS!
                        self.current_stage = TaskStage.PICKED_BLUE_BALL
                        self.steps_since_stage_change = 0
                        reward += self.STAGE_REWARDS[TaskStage.PICKED_BLUE_BALL]
                        print(f"  [Stage {self.current_stage}] SUCCESS! Picked up BLUE BALL! +{self.STAGE_REWARDS[TaskStage.PICKED_BLUE_BALL]}")
                    else:
                        # Trying to pick blue ball before unlocking door - penalty
                        reward += -5.0
                        print(f"  [WRONG ORDER] Picked blue ball before unlocking door! -5.0")
            else:
                # Failed pickup attempt
                reward += self.useless_action_penalty * 0.5

        # === DROP ACTION (action 4) ===
        elif action == 4:
            dropped = (pre_carrying is not None and post_carrying is None)

            if dropped:
                obj_type = getattr(pre_carrying, 'type', None)
                obj_color = getattr(pre_carrying, 'color', None)

                # Dropped green ball
                if obj_type == 'ball' and obj_color == 'green':
                    self.green_ball_drop_count += 1

                    if self.green_ball_drop_count == 1 and self.current_stage == TaskStage.PICKED_GREEN_BALL:
                        # First drop after first pickup - this is correct!
                        self.current_stage = TaskStage.DROPPED_GREEN_BALL
                        self.green_ball_dropped = True
                        self.steps_since_stage_change = 0
                        reward += self.STAGE_REWARDS[TaskStage.DROPPED_GREEN_BALL]
                        print(f"  [Stage {self.current_stage}] Dropped GREEN BALL! +{self.STAGE_REWARDS[TaskStage.DROPPED_GREEN_BALL]}")
                    else:
                        # Dropping green ball AGAIN - HUGE penalty
                        penalty = -5.0 * self.green_ball_drop_count
                        reward += penalty
                        print(f"  [EXPLOIT] Re-dropped green ball ({self.green_ball_drop_count}x)! {penalty}")

                # Dropped key
                elif obj_type == 'key':
                    if self.current_stage == TaskStage.UNLOCKED_DOOR:
                        self.current_stage = TaskStage.DROPPED_KEY
                        self.steps_since_stage_change = 0
                        reward += self.STAGE_REWARDS[TaskStage.DROPPED_KEY]
                        print(f"  [Stage {self.current_stage}] Dropped KEY! +{self.STAGE_REWARDS[TaskStage.DROPPED_KEY]}")
                    elif self.current_stage == TaskStage.PICKED_KEY:
                        # Dropped key before using it - big penalty
                        reward += -5.0
                        print(f"  [REGRESSION] Dropped key before unlocking door! -5.0")
                    else:
                        reward += -2.0
            else:
                # Failed drop (nothing to drop)
                reward += self.useless_action_penalty * 0.5

        # === TOGGLE ACTION (action 5) - Open box or door ===
        elif action == 5:
            # Small reward just for TRYING toggle - encourages exploration of this action
            reward += 0.05

            # Check what's in front
            front_cell = self._get_cell_at(front_pos)

            if front_cell is not None:
                obj_type = getattr(front_cell, 'type', None)

                # Toggle box (open it)
                if obj_type == 'box' and not self.box_opened:
                    if self.green_ball_dropped:
                        self.current_stage = TaskStage.OPENED_BOX
                        self.box_opened = True
                        self.steps_since_stage_change = 0
                        reward += self.STAGE_REWARDS[TaskStage.OPENED_BOX]
                        print(f"  [Stage {self.current_stage}] Opened BOX! +{self.STAGE_REWARDS[TaskStage.OPENED_BOX]}")
                        # Re-scan grid to find the key that was inside the box
                        self._scan_grid_for_objects()
                        if self.key_pos:
                            print(f"  [INFO] Key found at position {self.key_pos}")
                    else:
                        # Trying to open box before dropping green ball - bigger penalty
                        reward += -5.0
                        print(f"  [WRONG ORDER] Tried to open box before handling green ball! -5.0")

                # Toggle door (unlock it)
                elif obj_type == 'door':
                    # After toggle, key is consumed if door was locked
                    had_key = (pre_carrying is not None and
                              getattr(pre_carrying, 'type', None) == 'key')

                    if had_key and not self.door_unlocked:
                        # Unlocking door with key - success!
                        self.current_stage = TaskStage.UNLOCKED_DOOR
                        self.door_unlocked = True
                        self.steps_since_stage_change = 0
                        reward += self.STAGE_REWARDS[TaskStage.UNLOCKED_DOOR]
                        print(f"  [Stage {self.current_stage}] Unlocked DOOR! +{self.STAGE_REWARDS[TaskStage.UNLOCKED_DOOR]}")
                    elif not had_key and not self.door_unlocked:
                        # Trying to open door without key
                        reward += -0.5
                        print(f"  [WRONG ORDER] Tried to unlock door without key! -0.5")

        # === MOVEMENT ACTIONS (0, 1, 2) ===
        elif action in (0, 1, 2):  # Turn left, turn right, forward
            # Reward for moving toward current objective
            target = self._get_current_target()
            if target and self.last_position:
                old_dist = abs(self.last_position[0] - target[0]) + abs(self.last_position[1] - target[1])
                new_dist = abs(agent_pos[0] - target[0]) + abs(agent_pos[1] - target[1])
                if new_dist < old_dist:
                    # Bigger reward after green ball is handled - incentivize moving to box
                    if self.green_ball_dropped:
                        reward += self.progress_reward * 3  # 3x reward after green ball
                    else:
                        reward += self.progress_reward
                elif new_dist > old_dist and self.green_ball_dropped:
                    # Small penalty for moving away from objective after green ball
                    reward += -0.01

            # Penalty for spinning in place
            if len(self.action_history) >= 4:
                recent = list(self.action_history)[-4:]
                if all(a in (0, 1) for a in recent):  # All turns
                    reward += self.stuck_penalty

        # === "SEEING" OBJECTS (bonus for approaching objectives) ===
        grid = obs['image']
        visible_reward = self._check_visible_objects(grid)
        reward += visible_reward

        # === FACING BOX BONUS - encourage toggling ===
        if self.green_ball_dropped and not self.box_opened:
            front_cell = self._get_cell_at(front_pos)
            if front_cell is not None and getattr(front_cell, 'type', None) == 'box':
                # Agent is facing the box! Give bonus to encourage toggle
                reward += 0.5
                if action != 5:  # If didn't toggle, hint that it should
                    print(f"  [HINT] Facing box but didn't toggle! Try action 5")

        # === FACING DOOR WITH KEY BONUS ===
        if self.current_stage == TaskStage.PICKED_KEY and not self.door_unlocked:
            front_cell = self._get_cell_at(front_pos)
            if front_cell is not None and getattr(front_cell, 'type', None) == 'door':
                reward += 0.5
                if action != 5:
                    print(f"  [HINT] Facing door with key but didn't toggle! Try action 5")

        # === DISTANCE-BASED SHAPING - continuous guidance ===
        reward += self._calculate_distance_reward(agent_pos)

        return reward

    def _calculate_distance_reward(self, agent_pos):
        """Give reward/penalty based on distance to current objective."""
        reward = 0.0
        target = None
        reward_multiplier = 1.0

        # Determine current target based on stage
        if not self.green_ball_dropped:
            # Need to get green ball first
            target = self.green_ball_pos
        elif not self.box_opened:
            # Need to open box first - this is critical!
            target = self.box_pos
            reward_multiplier = 2.0  # Extra incentive to find box
        elif not self.door_unlocked:
            # Need key, then door
            carrying = self.env.unwrapped.carrying
            if carrying is None:
                # Need to pick up key - it should be where the box was (or nearby)
                target = self.key_pos if self.key_pos else self.box_pos
                reward_multiplier = 1.5
            else:
                # Have key, go to door
                target = self.door_pos
                reward_multiplier = 2.0  # Very important to reach door
        else:
            # Door unlocked, go to blue ball
            target = self.blue_ball_pos
            reward_multiplier = 2.0

        if target is None or self.last_position is None:
            return 0.0

        # Calculate distances
        old_dist = abs(self.last_position[0] - target[0]) + abs(self.last_position[1] - target[1])
        new_dist = abs(agent_pos[0] - target[0]) + abs(agent_pos[1] - target[1])

        # Reward for getting closer, penalty for getting further
        if new_dist < old_dist:
            reward = 0.15 * reward_multiplier  # Getting closer
        elif new_dist > old_dist:
            reward = -0.08 * reward_multiplier  # Getting further

        # Bonus for being very close to target
        if new_dist <= 1:
            reward += 0.3 * reward_multiplier

        return reward

    def _check_visible_objects(self, grid):
        """Give small rewards for seeing relevant objects in the 7x7 view."""
        reward = 0.0

        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                cell = grid[y, x]
                obj_type = cell[0]
                obj_color = cell[1]

                # Green ball visible and we need it (ONLY if not yet handled)
                if obj_type == 6 and obj_color == 1:  # Ball, green
                    if not self.green_ball_dropped and self.current_stage < TaskStage.FOUND_GREEN_BALL:
                        self.current_stage = TaskStage.FOUND_GREEN_BALL
                        self.steps_since_stage_change = 0
                        reward += self.STAGE_REWARDS[TaskStage.FOUND_GREEN_BALL]

                # Box visible and we need it
                if obj_type == 7:  # Box
                    if self.green_ball_dropped and self.current_stage == TaskStage.DROPPED_GREEN_BALL:
                        self.current_stage = TaskStage.FOUND_BOX
                        self.steps_since_stage_change = 0
                        reward += self.STAGE_REWARDS[TaskStage.FOUND_BOX]
                        print(f"  [Stage {self.current_stage}] Found BOX! +{self.STAGE_REWARDS[TaskStage.FOUND_BOX]}")

                # Blue ball visible (always good to see the goal)
                if obj_type == 6 and obj_color == 0:  # Ball, blue
                    if self.current_stage >= TaskStage.DROPPED_KEY:
                        reward += 0.02  # Bonus for seeing goal when ready

        return reward

    def _get_front_pos(self, pos, direction):
        """Get position in front of agent."""
        x, y = pos
        if direction == 0:    # Right
            return (x + 1, y)
        elif direction == 1:  # Down
            return (x, y + 1)
        elif direction == 2:  # Left
            return (x - 1, y)
        elif direction == 3:  # Up
            return (x, y - 1)
        return pos

    def _get_cell_at(self, pos):
        """Get the cell object at a position."""
        if not hasattr(self.env, 'unwrapped') or not hasattr(self.env.unwrapped, 'grid'):
            return None
        grid = self.env.unwrapped.grid
        x, y = pos
        if 0 <= x < grid.width and 0 <= y < grid.height:
            return grid.get(x, y)
        return None


# =============================================================================
# PPO NETWORK (same architecture as existing)
# =============================================================================

class PPONetwork(nn.Module):
    """Actor-Critic network for PPO."""

    def __init__(self, obs_shape, action_size, hidden_size=256):
        super().__init__()

        img_height, img_width, img_channels = obs_shape

        # CNN for image
        self.conv1 = nn.Conv2d(img_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        conv_out_size = img_height * img_width * 64
        additional_features = 5  # direction(1) + agent_pos_norm(2) + goal_dir(2)

        # Shared layers
        self.fc_shared = nn.Linear(conv_out_size + additional_features, hidden_size)
        self.fc_shared2 = nn.Linear(hidden_size, hidden_size)

        # Actor head
        self.actor_fc = nn.Linear(hidden_size, hidden_size // 2)
        self.actor_out = nn.Linear(hidden_size // 2, action_size)

        # Critic head
        self.critic_fc = nn.Linear(hidden_size, hidden_size // 2)
        self.critic_out = nn.Linear(hidden_size // 2, 1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, image, direction, agent_pos_norm, goal_direction):
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)

        additional = torch.cat([
            direction.unsqueeze(1) if direction.dim() == 1 else direction,
            agent_pos_norm,
            goal_direction
        ], dim=1)

        x = torch.cat([x, additional], dim=1)
        x = F.relu(self.fc_shared(x))
        x = F.relu(self.fc_shared2(x))

        # Actor
        actor_x = F.relu(self.actor_fc(x))
        action_logits = self.actor_out(actor_x)

        # Critic
        critic_x = F.relu(self.critic_fc(x))
        state_value = self.critic_out(critic_x)

        return action_logits, state_value


class PPOAgent:
    """PPO Agent with action masking."""

    def __init__(self, obs_shape, action_size, lr=5e-4, gamma=0.99,
                 gae_lambda=0.95, clip_epsilon=0.2, c1=0.5, c2=0.2,
                 epochs=6, batch_size=32, hidden_size=256):

        self.obs_shape = obs_shape
        self.action_size = action_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.c1 = c1
        self.c2 = c2  # Entropy coefficient - higher for exploration
        self.epochs = epochs
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = PPONetwork(obs_shape, action_size, hidden_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        self.reset_storage()

        print(f"PPO Agent (1Dlhb) initialized on {self.device}")
        print(f"  LR: {lr}, Gamma: {gamma}, Entropy coef: {c2}")

    def reset_storage(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def preprocess_state(self, state):
        image = torch.FloatTensor(state['image']).permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        direction = torch.FloatTensor([state['direction']]).to(self.device)
        agent_pos_norm = torch.FloatTensor(state['agent_pos_normalized']).unsqueeze(0).to(self.device)
        goal_direction = torch.FloatTensor(state['goal_direction']).unsqueeze(0).to(self.device)
        return image, direction, agent_pos_norm, goal_direction

    def act(self, state, training=True):
        image, direction, agent_pos_norm, goal_direction = self.preprocess_state(state)

        with torch.no_grad():
            action_logits, state_value = self.network(image, direction, agent_pos_norm, goal_direction)

        # Action masking: can't drop if not carrying
        if state.get('carrying') is None:
            action_logits[0, 4] = -1e10  # Mask drop action

        probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(probs)

        if training:
            action = dist.sample()
            log_prob = dist.log_prob(action)

            self.states.append(state)
            self.actions.append(action.item())
            self.log_probs.append(log_prob.item())
            self.values.append(state_value.item())

            return action.item()
        else:
            return torch.argmax(probs, dim=-1).item()

    def store_reward_done(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_gae(self, next_value):
        advantages = []
        returns = []
        gae = 0
        values = self.values + [next_value]

        for t in reversed(range(len(self.rewards))):
            next_non_terminal = 1.0 - self.dones[t]
            next_val = values[t + 1]
            delta = self.rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        return advantages, returns

    def update(self, next_state=None):
        if len(self.states) == 0:
            return 0.0, 0.0, 0.0

        # Get next value
        if next_state is not None:
            image, direction, agent_pos_norm, goal_direction = self.preprocess_state(next_state)
            with torch.no_grad():
                _, next_value = self.network(image, direction, agent_pos_norm, goal_direction)
                next_value = next_value.item()
        else:
            next_value = 0.0

        # Compute GAE
        advantages, returns = self.compute_gae(next_value)

        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_updates = 0

        for _ in range(self.epochs):
            indices = np.arange(len(self.states))
            np.random.shuffle(indices)

            for start in range(0, len(self.states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                # Prepare batch
                batch_images = []
                batch_directions = []
                batch_agent_pos = []
                batch_goal_dirs = []
                batch_actions = []

                for idx in batch_indices:
                    state = self.states[idx]
                    image, direction, agent_pos_norm, goal_direction = self.preprocess_state(state)
                    batch_images.append(image)
                    batch_directions.append(direction)
                    batch_agent_pos.append(agent_pos_norm)
                    batch_goal_dirs.append(goal_direction)
                    batch_actions.append(self.actions[idx])

                batch_images = torch.cat(batch_images, dim=0)
                batch_directions = torch.cat(batch_directions, dim=0)
                batch_agent_pos = torch.cat(batch_agent_pos, dim=0)
                batch_goal_dirs = torch.cat(batch_goal_dirs, dim=0)
                batch_actions = torch.LongTensor(batch_actions).to(self.device)

                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]

                # Forward pass
                action_logits, state_values = self.network(
                    batch_images, batch_directions, batch_agent_pos, batch_goal_dirs
                )

                probs = F.softmax(action_logits, dim=-1)
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # PPO loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                ratio = torch.clamp(ratio, 0.0, 10.0)

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(state_values.squeeze(-1), batch_returns)

                loss = policy_loss + self.c1 * value_loss - self.c2 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                num_updates += 1

        self.reset_storage()

        return (total_loss / max(1, num_updates),
                total_policy_loss / max(1, num_updates),
                total_value_loss / max(1, num_updates))

    def save(self, path):
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


# =============================================================================
# ENVIRONMENT CREATION
# =============================================================================

def create_env():
    """Create the ObstructedMaze-1Dlhb environment with task progression wrapper."""
    base_env = gym.make("MiniGrid-ObstructedMaze-1Dlhb-v0")
    env = TaskProgressionWrapper(base_env)
    return env


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train(episodes=5000, max_steps=250, update_frequency=128,
          save_interval=500, model_dir='models_1dlhb', log_dir='logs_1dlhb'):
    """Train PPO agent on ObstructedMaze-1Dlhb."""

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(f'{log_dir}/ppo_1dlhb_{timestamp}')

    print("=" * 60)
    print("PPO TRAINING FOR ObstructedMaze-1Dlhb")
    print("=" * 60)

    # Create environment
    env = create_env()

    obs_shape = env.observation_space['image'].shape
    action_size = env.action_space.n

    print(f"\nConfiguration:")
    print(f"  Environment: MiniGrid-ObstructedMaze-1Dlhb-v0")
    print(f"  Observation shape: {obs_shape}")
    print(f"  Action space: {action_size}")
    print(f"  Episodes: {episodes}")
    print(f"  Max steps: {max_steps}")
    print(f"  Update frequency: {update_frequency}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # Create agent with higher entropy for exploration
    agent = PPOAgent(
        obs_shape=obs_shape,
        action_size=action_size,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        c1=0.5,
        c2=0.2,   # Good entropy for exploration
        epochs=6,  # More training per update
        batch_size=32  # Smaller batches for faster updates
    )

    # Metrics
    scores = []
    success_window = deque(maxlen=100)
    stage_reached = []
    best_success_rate = 0
    total_steps = 0
    start_time = time.time()

    # Track key milestones
    total_box_opens = 0
    total_key_pickups = 0
    total_door_unlocks = 0
    total_successes = 0

    print("\nTraining started...\n")

    try:
        for episode in tqdm(range(episodes), desc="Training"):
            obs, info = env.reset()
            state = obs

            episode_reward = 0
            max_stage = 0

            for step in range(max_steps):
                action = agent.act(state, training=True)
                next_obs, reward, done, truncated, info = env.step(action)

                agent.store_reward_done(reward, done or truncated)

                episode_reward += reward
                max_stage = max(max_stage, info.get('task_stage', 0))
                total_steps += 1

                state = next_obs

                # Update policy
                if total_steps % update_frequency == 0:
                    loss, pl, vl = agent.update(state if not (done or truncated) else None)
                    writer.add_scalar('Loss/Total', loss, total_steps)
                    writer.add_scalar('Loss/Policy', pl, total_steps)
                    writer.add_scalar('Loss/Value', vl, total_steps)

                if done or truncated:
                    break

            # End of episode update
            if len(agent.states) > 0:
                agent.update()

            # Track metrics
            scores.append(episode_reward)
            stage_reached.append(max_stage)
            success = 1 if max_stage == TaskStage.PICKED_BLUE_BALL else 0
            success_window.append(success)

            # Track milestones
            if max_stage >= TaskStage.OPENED_BOX:
                total_box_opens += 1
            if max_stage >= TaskStage.PICKED_KEY:
                total_key_pickups += 1
            if max_stage >= TaskStage.UNLOCKED_DOOR:
                total_door_unlocks += 1
            if max_stage >= TaskStage.PICKED_BLUE_BALL:
                total_successes += 1

            # Log to tensorboard
            writer.add_scalar('Episode/Reward', episode_reward, episode)
            writer.add_scalar('Episode/MaxStage', max_stage, episode)
            writer.add_scalar('Episode/SuccessRate', np.mean(success_window), episode)
            writer.add_scalar('Episode/Steps', step + 1, episode)
            writer.add_scalar('Milestones/BoxOpens', total_box_opens, episode)
            writer.add_scalar('Milestones/KeyPickups', total_key_pickups, episode)
            writer.add_scalar('Milestones/DoorUnlocks', total_door_unlocks, episode)
            writer.add_scalar('Milestones/Successes', total_successes, episode)

            # Progress report
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(scores[-100:])
                avg_stage = np.mean(stage_reached[-100:])
                success_rate = np.mean(success_window)
                elapsed = time.time() - start_time

                print(f"\nEpisode {episode + 1}:")
                print(f"  Avg Reward (100): {avg_reward:.2f}")
                print(f"  Avg Stage (100): {avg_stage:.2f}")
                print(f"  Success Rate: {success_rate:.2%}")
                print(f"  Milestones - Box: {total_box_opens}, Key: {total_key_pickups}, Door: {total_door_unlocks}, Success: {total_successes}")
                print(f"  Total Steps: {total_steps}")
                print(f"  Time: {elapsed/60:.1f} min")

                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    print(f"  NEW BEST SUCCESS RATE!")

            # Save model
            if (episode + 1) % save_interval == 0:
                path = f"{model_dir}/ppo_1dlhb_ep{episode+1}.pth"
                agent.save(path)
                print(f"  Model saved: {path}")

            # Success message
            if success:
                print(f"SUCCESS at episode {episode + 1}! Steps: {step + 1}")

            # Early stopping
            if len(success_window) >= 100 and np.mean(success_window) > 0.9:
                print(f"\nExcellent performance achieved! Success rate: {np.mean(success_window):.2%}")
                break

    except KeyboardInterrupt:
        print("\nTraining interrupted.")

    # Save final model
    final_path = f"{model_dir}/ppo_1dlhb_final.pth"
    agent.save(final_path)

    env.close()
    writer.close()

    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Episodes: {len(scores)}")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Final success rate: {np.mean(success_window):.2%}")
    print(f"  Best success rate: {best_success_rate:.2%}")
    print(f"  Final model: {final_path}")

    return agent, scores, stage_reached


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(model_path, num_episodes=10, render=True, save_best_gif=False):
    """Evaluate a trained agent and optionally save best episode as GIF."""

    print(f"\nEvaluating model: {model_path}")

    # Create environment (no rendering for initial evaluation)
    env = create_env()

    obs_shape = env.observation_space['image'].shape
    action_size = env.action_space.n

    agent = PPOAgent(obs_shape=obs_shape, action_size=action_size)
    agent.load(model_path)

    successes = 0
    total_steps = []

    # Track all episodes for finding best one
    episode_results = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        state = obs
        ep_reward = 0
        ep_actions = []  # Store actions for replay
        seed = np.random.randint(0, 1000000)  # Store seed for reproducibility

        # Reset with specific seed for reproducibility
        obs, info = env.reset(seed=seed)
        state = obs

        for step in range(300):
            action = agent.act(state, training=True)
            ep_actions.append(action)

            if len(agent.states) > 100:
                agent.reset_storage()

            next_obs, reward, done, truncated, info = env.step(action)

            ep_reward += reward
            state = next_obs

            if done or truncated:
                break

        agent.reset_storage()

        max_stage = info.get('task_stage', 0)
        success = max_stage == TaskStage.PICKED_BLUE_BALL

        if success:
            successes += 1
            total_steps.append(step + 1)

        episode_results.append({
            'episode': ep + 1,
            'reward': ep_reward,
            'stage': max_stage,
            'steps': step + 1,
            'success': success,
            'seed': seed,
            'actions': ep_actions
        })

        print(f"Episode {ep + 1}: Stage={max_stage} ({TaskStage(max_stage).name}), "
              f"Steps={step + 1}, Reward={ep_reward:.2f}, Success={success}")

    env.close()

    print(f"\nResults: {successes}/{num_episodes} successes ({successes/num_episodes:.1%})")
    if total_steps:
        print(f"Average steps to success: {np.mean(total_steps):.1f}")

    # Find best episode and save as GIF
    if save_best_gif and episode_results:
        best_ep = max(episode_results, key=lambda x: (x['success'], x['reward'], -x['steps']))
        print(f"\nBest episode: #{best_ep['episode']} with reward {best_ep['reward']:.2f}, "
              f"stage {best_ep['stage']} ({TaskStage(best_ep['stage']).name})")

        save_episode_as_gif(model_path, best_ep['seed'], best_ep['actions'])


def save_episode_as_gif(model_path, seed, actions, output_path=None):
    """Replay an episode and save as GIF."""
    try:
        import imageio
    except ImportError:
        print("Installing imageio for GIF creation...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'imageio'])
        import imageio

    print(f"\nRecording best episode as GIF...")

    # Create environment with rgb_array rendering
    base_env = gym.make("MiniGrid-ObstructedMaze-1Dlhb-v0", render_mode="rgb_array")
    env = TaskProgressionWrapper(base_env)

    # Reset with the same seed
    obs, info = env.reset(seed=seed)

    frames = []

    # Capture initial frame
    frame = env.render()
    if frame is not None:
        frames.append(frame)

    # Replay the actions
    for action in actions:
        obs, reward, done, truncated, info = env.step(action)

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        if done or truncated:
            break

    env.close()

    # Save as GIF
    if output_path is None:
        output_path = "best_episode.gif"

    if frames:
        # Save with imageio
        imageio.mimsave(output_path, frames, fps=10, loop=0)
        print(f"GIF saved to: {output_path}")
        print(f"Frames captured: {len(frames)}")
    else:
        print("No frames captured - GIF not saved")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='PPO for ObstructedMaze-1Dlhb')
    parser.add_argument('--mode', choices=['train', 'eval'], default='train')
    parser.add_argument('--episodes', type=int, default=3000)
    parser.add_argument('--model', type=str, default=None, help='Model path for evaluation')
    parser.add_argument('--render', action='store_true', help='Render during evaluation')
    parser.add_argument('--save-gif', action='store_true', help='Save best episode as GIF')
    parser.add_argument('--num-eval', type=int, default=10, help='Number of evaluation episodes')

    args = parser.parse_args()

    if args.mode == 'train':
        train(episodes=args.episodes)
    elif args.mode == 'eval':
        if args.model is None:
            args.model = 'models_1dlhb/ppo_1dlhb_final.pth'
        evaluate(args.model, num_episodes=args.num_eval, render=args.render, save_best_gif=args.save_gif)
