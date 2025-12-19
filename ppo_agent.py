import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import deque


class PPONetwork(nn.Module):
    """Combined Actor-Critic network for PPO."""

    def __init__(self, obs_shape, action_size, hidden_size=256):
        super(PPONetwork, self).__init__()

        # Calculate input dimensions
        self.obs_shape = obs_shape
        img_height, img_width, img_channels = obs_shape

        # CNN for image processing (same as your DQN)
        self.conv1 = nn.Conv2d(img_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)


        # Calculate conv output size
        conv_out_size = img_height * img_width * 64

        # Additional features: direction (1) + agent_pos_normalized (2) + goal_direction (2)
        additional_features = 5

        # Shared feature layers
        self.fc_shared = nn.Linear(conv_out_size + additional_features, hidden_size)
        self.fc_shared2 = nn.Linear(hidden_size, hidden_size)

        # Actor head (policy)
        self.actor_fc = nn.Linear(hidden_size, hidden_size // 2)
        self.actor_out = nn.Linear(hidden_size // 2, action_size)

        # Critic head (value function)
        self.critic_fc = nn.Linear(hidden_size, hidden_size // 2)
        self.critic_out = nn.Linear(hidden_size // 2, 1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, image, direction, agent_pos_norm, goal_direction):
        """Forward pass returning both policy logits and state value."""
        # Process image
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))

        x = x.view(x.size(0), -1)

        # Concatenate with additional features
        additional = torch.cat([
            direction.unsqueeze(1) if direction.dim() == 1 else direction,
            agent_pos_norm,
            goal_direction
        ], dim=1)

        x = torch.cat([x, additional], dim=1)

        # Shared layers
        x = F.relu(self.fc_shared(x))
        x = F.relu(self.fc_shared2(x))

        # Actor (policy)
        actor_x = F.relu(self.actor_fc(x))
        action_logits = self.actor_out(actor_x)

        # Critic (value)
        critic_x = F.relu(self.critic_fc(x))
        state_value = self.critic_out(critic_x)

        return action_logits, state_value


class PPOAgent:
    """PPO agent for MiniGrid environments."""

    def __init__(
            self,
            obs_shape,
            action_size,
            lr=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            c1=0.5,  # value loss coefficient
            c2=0.01,  # entropy coefficient
            epochs=10,  # ← CHANGE THIS from 4 to 10
            batch_size=64,
            hidden_size=256
    ):
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.c1 = c1
        self.c2 = c2
        self.epochs = epochs
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create network
        self.network = PPONetwork(obs_shape, action_size, hidden_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # Storage for trajectories
        self.reset_storage()

        print(f"PPO Agent initialized on device: {self.device}")
        print(f"  Learning rate: {lr}")
        print(f"  Gamma: {gamma}")
        print(f"  GAE Lambda: {gae_lambda}")
        print(f"  Clip epsilon: {clip_epsilon}")
        print(f"  Epochs per update: {epochs}")
        print(f"  Batch size: {batch_size}")

    def reset_storage(self):
        """Reset trajectory storage."""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def preprocess_state(self, state):
        """Convert state dict to tensors."""
        image = torch.FloatTensor(state['image']).permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        direction = torch.FloatTensor([state['direction']]).to(self.device)
        agent_pos_norm = torch.FloatTensor(state['agent_pos_normalized']).unsqueeze(0).to(self.device)
        goal_direction = torch.FloatTensor(state['goal_direction']).unsqueeze(0).to(self.device)

        return image, direction, agent_pos_norm, goal_direction

    def act(self, state, training=True):
        """Select action using current policy."""
        image, direction, agent_pos_norm, goal_direction = self.preprocess_state(state)

        with torch.no_grad():
            action_logits, state_value = self.network(image, direction, agent_pos_norm, goal_direction)

        # ADD THIS NEW CODE - Mask drop action if not carrying anything
        if 'carrying' in state and state['carrying'] is None:
            action_logits[0, 4] = -1e10  # Make drop action nearly impossible when not carrying

        # Create probability distribution
        probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(probs)

        if training:
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # Store for training
            self.states.append(state)
            self.actions.append(action.item())
            self.log_probs.append(log_prob.item())
            self.values.append(state_value.item())

            return action.item()
        else:
            # Greedy action for evaluation
            action = torch.argmax(probs, dim=-1)
            return action.item()

    def store_reward_done(self, reward, done):
        """Store reward and done flag."""
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_gae(self, next_value):
        """Compute Generalized Advantage Estimation."""
        advantages = []
        returns = []
        gae = 0

        values = self.values + [next_value]

        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = values[t + 1]
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = values[t + 1]

            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae

            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        return advantages, returns

    def update(self, next_state=None):
        """Update policy using collected trajectories."""
        if len(self.states) == 0:
            return 0.0, 0.0, 0.0

        # Compute next value for GAE
        if next_state is not None:
            image, direction, agent_pos_norm, goal_direction = self.preprocess_state(next_state)
            with torch.no_grad():
                _, next_value = self.network(image, direction, agent_pos_norm, goal_direction)
                next_value = next_value.item()
        else:
            next_value = 0.0

        # Compute advantages and returns
        advantages, returns = self.compute_gae(next_value)

        # Convert to tensors
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)

        # Normalize advantages - handle edge cases
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = advantages - advantages.mean()
        # Prepare batch data
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_updates = 0

        # Multiple epochs over the data
        for _ in range(self.epochs):
            # Create mini-batches
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

                # Compute new log probs
                probs = F.softmax(action_logits, dim=-1)

                # Check for NaN
                if torch.isnan(probs).any() or torch.isinf(probs).any():
                    print("⚠️ NaN/Inf detected in probabilities! Skipping update.")
                    self.reset_storage()
                    return 0.0, 0.0, 0.0

                dist = Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # Policy loss (PPO clipped objective)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                # Clamp ratio to prevent exploding values
                ratio = torch.clamp(ratio, 0.0, 10.0)

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(state_values.squeeze(-1), batch_returns)

                # Total loss
                loss = policy_loss + self.c1 * value_loss - self.c2 * entropy

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                num_updates += 1

        # Reset storage
        self.reset_storage()

        avg_loss = total_loss / num_updates if num_updates > 0 else 0.0
        avg_policy_loss = total_policy_loss / num_updates if num_updates > 0 else 0.0
        avg_value_loss = total_value_loss / num_updates if num_updates > 0 else 0.0

        return avg_loss, avg_policy_loss, avg_value_loss

    def save(self, path):
        """Save model."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
