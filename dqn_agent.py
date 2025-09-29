import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import torch.nn.functional as F


class DQNNetwork(nn.Module):
    """Deep Q-Network with improved architecture and attention mechanism."""
    
    def __init__(self, obs_shape, action_size, hidden_size=512):
        super(DQNNetwork, self).__init__()
        
        # Enhanced CNN layers with residual connections
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        # Batch normalization for stable training
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)
        
        # Attention mechanism for important features
        self.attention = nn.Conv2d(128, 1, kernel_size=1)
        
        # Calculate conv output size: 7x7x128 = 6272
        conv_out_size = 7 * 7 * 128
        
        # Add direction input (4 possible directions)
        total_input_size = conv_out_size + 4
        
        # Dueling DQN architecture
        self.feature_layer = nn.Linear(total_input_size, hidden_size)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, image, direction):
        batch_size = image.size(0)
        
        # Process image: (batch, 7, 7, 3) -> (batch, 3, 7, 7)
        x = image.permute(0, 3, 1, 2).float() / 255.0
        
        # Enhanced CNN with residual connections and batch norm
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x4 = F.relu(self.bn4(self.conv4(x3)))
        
        # Apply attention mechanism
        attention_weights = torch.sigmoid(self.attention(x4))
        x4_attended = x4 * attention_weights
        
        # Flatten conv output
        x = x4_attended.reshape(batch_size, -1)
        
        # One-hot encode direction
        direction_onehot = F.one_hot(direction, num_classes=4).float()
        
        # Concatenate image features and direction
        x = torch.cat([x, direction_onehot], dim=1)
        
        # Feature extraction
        features = F.relu(self.feature_layer(x))
        features = self.dropout(features)
        
        # Dueling DQN: V(s) and A(s,a)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer for more efficient learning."""
    
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = 1e-6
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        
    def add(self, experience):
        max_priority = np.max(self.priorities) if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None, None, None
        
        # Calculate probabilities
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return experiences, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """Deep Q-Learning Agent with multiple improvements."""
    
    def __init__(self, obs_shape, action_size, lr=1e-4, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.02,
                 memory_size=100000, batch_size=32, target_update=1000,
                 use_prioritized_replay=True, use_noisy_nets=False):
        
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update = target_update
        self.use_prioritized_replay = use_prioritized_replay
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural networks
        self.q_network = DQNNetwork(obs_shape, action_size).to(self.device)
        self.target_network = DQNNetwork(obs_shape, action_size).to(self.device)
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=lr, weight_decay=1e-5)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.8, patience=1000
        )
        
        # Replay buffer
        if use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(memory_size)
        else:
            self.memory = deque(maxlen=memory_size)
        
        # Training counters and metrics
        self.step_count = 0
        self.episode_count = 0
        
        # Intrinsic motivation tracking
        self.visited_positions = set()
        self.last_position = None
        self.step_rewards = []
        
        # Initialize target network
        self.update_target_network()
        
        print(f"DQN Agent initialized:")
        print(f"  Device: {self.device}")
        print(f"  Architecture: Dueling DQN with Attention")
        print(f"  Prioritized Replay: {use_prioritized_replay}")
        print(f"  Learning rate: {lr}")
        print(f"  Memory size: {memory_size}")
        
    def get_intrinsic_reward(self, state, action, next_state, done):
        """Calculate intrinsic rewards for better exploration."""
        intrinsic_reward = 0.0
        
        # Exploration bonus for visiting new positions
        agent_pos = tuple(next_state.get('agent_pos', [0, 0]))
        if agent_pos not in self.visited_positions:
            self.visited_positions.add(agent_pos)
            intrinsic_reward += 0.1  # Small reward for exploration
        
        # Movement reward (encourage moving vs staying still)
        if self.last_position is not None:
            if agent_pos != self.last_position:
                intrinsic_reward += 0.01
        
        self.last_position = agent_pos
        
        # Penalize inactivity (if taking 'done' action without completing task)
        if action == 6 and not done:  # 'done' action
            intrinsic_reward -= 0.1
        
        return intrinsic_reward
    
    def remember(self, state, action, reward, next_state, done, info=None):
        """Store experience with intrinsic reward shaping."""
        # Add intrinsic reward
        intrinsic_reward = self.get_intrinsic_reward(state, action, next_state, done)
        shaped_reward = reward + intrinsic_reward
        
        experience = (state, action, shaped_reward, next_state, done)
        
        if self.use_prioritized_replay:
            self.memory.add(experience)
        else:
            self.memory.append(experience)
    
    def act(self, state, training=True):
        """Enhanced action selection with epsilon-greedy and UCB exploration."""
        # Epsilon-greedy exploration
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            image_np = np.asarray(state['image'])
            image = torch.from_numpy(image_np).unsqueeze(0).to(self.device)
            direction = torch.tensor([state['direction']], dtype=torch.long, device=self.device)
            
            q_values = self.q_network(image, direction)
            
            # Add small amount of noise for better exploration during training
            if training and self.epsilon > 0.1:
                noise = torch.randn_like(q_values) * 0.1 * self.epsilon
                q_values += noise
            
            return int(torch.argmax(q_values, dim=1).item())
    
    def replay(self):
        """Enhanced training with prioritized experience replay."""
        if self.use_prioritized_replay:
            if len(self.memory) < self.batch_size:
                return 0.0
            
            experiences, indices, weights = self.memory.sample(self.batch_size)
            if experiences is None:
                return 0.0
            
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            if len(self.memory) < self.batch_size:
                return 0.0
            
            experiences = random.sample(self.memory, self.batch_size)
            weights = torch.ones(self.batch_size).to(self.device)
        
        # Prepare batch data
        states = {'image': [], 'direction': []}
        actions = []
        rewards = []
        next_states = {'image': [], 'direction': []}
        dones = []
        
        for state, action, reward, next_state, done in experiences:
            states['image'].append(state['image'])
            states['direction'].append(state['direction'])
            actions.append(action)
            rewards.append(reward)
            next_states['image'].append(next_state['image'])
            next_states['direction'].append(next_state['direction'])
            dones.append(done)
        
        # Convert to tensors
        state_images_np = np.stack(states['image'], axis=0)
        next_state_images_np = np.stack(next_states['image'], axis=0)
        state_images = torch.from_numpy(state_images_np).to(self.device)
        next_state_images = torch.from_numpy(next_state_images_np).to(self.device)
        state_directions = torch.tensor(states['direction'], dtype=torch.long, device=self.device)
        next_state_directions = torch.tensor(next_states['direction'], dtype=torch.long, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        
        # Current Q values
        current_q_values = self.q_network(state_images, state_directions).gather(1, actions.unsqueeze(1))
        
        # Double DQN: use main network to select actions, target network to evaluate
        with torch.no_grad():
            next_actions = self.q_network(next_state_images, next_state_directions).argmax(1)
            next_q_values = self.target_network(next_state_images, next_state_directions).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute TD errors for prioritized replay
        td_errors = torch.abs(current_q_values.squeeze() - target_q_values).detach()
        
        # Weighted loss
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values, reduction='none')
        weighted_loss = (weights * loss).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update priorities
        if self.use_prioritized_replay:
            priorities = td_errors.cpu().numpy() + 1e-6
            self.memory.update_priorities(indices, priorities)
        
        # Learning rate scheduling
        self.scheduler.step(weighted_loss.item())
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.update_target_network()
        
        return weighted_loss.item()
    
    def update_target_network(self):
        """Soft update of target network."""
        tau = 0.005  # Soft update parameter
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def reset_episode(self):
        """Reset episode-specific tracking."""
        self.episode_count += 1
        self.last_position = None
        # Reset exploration tracking periodically
        if self.episode_count % 100 == 0:
            self.visited_positions.clear()
    
    def save(self, filepath):
        """Save the model with additional metadata."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'visited_positions': self.visited_positions
        }, filepath)
    
    def load(self, filepath):
        """Load the model with additional metadata."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.step_count = checkpoint.get('step_count', 0)
        self.episode_count = checkpoint.get('episode_count', 0)
        self.visited_positions = checkpoint.get('visited_positions', set())
