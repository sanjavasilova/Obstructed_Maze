import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import torch.nn.functional as F


class DQNNetwork(nn.Module):
    """Deep Q-Network for processing MiniGrid observations."""
    
    def __init__(self, obs_shape, action_size, hidden_size=512):
        super(DQNNetwork, self).__init__()
        
        # Calculate the size after CNN layers
        # Input: (batch, 7, 7, 3) -> (batch, 3, 7, 7)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Calculate conv output size: 7x7x64 = 3136
        conv_out_size = 7 * 7 * 64
        
        # Add direction input (4 possible directions)
        total_input_size = conv_out_size + 4
        
        self.fc1 = nn.Linear(total_input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, image, direction):
        # Process image: (batch, 7, 7, 3) -> (batch, 3, 7, 7)
        x = image.permute(0, 3, 1, 2).float() / 255.0
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten conv output
        x = x.reshape(x.size(0), -1)
        
        # One-hot encode direction
        direction_onehot = F.one_hot(direction, num_classes=4).float()
        
        # Concatenate image features and direction
        x = torch.cat([x, direction_onehot], dim=1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class DQNAgent:
    """Deep Q-Learning Agent for MiniGrid environments."""
    
    def __init__(self, obs_shape, action_size, lr=1e-4, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 memory_size=50000, batch_size=32, target_update=1000):
        
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQNNetwork(obs_shape, action_size).to(self.device)
        self.target_network = DQNNetwork(obs_shape, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.memory = deque(maxlen=memory_size)
        
        # Training counters
        self.step_count = 0
        
        # Initialize target network
        self.update_target_network()
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy."""
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            image_np = np.asarray(state['image'])
            image = torch.from_numpy(image_np).unsqueeze(0).to(self.device)
            direction = torch.tensor([state['direction']], dtype=torch.long, device=self.device)
            q_values = self.q_network(image, direction)
            return int(torch.argmax(q_values, dim=1).item())
    
    def replay(self):
        """Train the model on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        batch = random.sample(self.memory, self.batch_size)
        
        # Prepare batch data
        states = {'image': [], 'direction': []}
        actions = []
        rewards = []
        next_states = {'image': [], 'direction': []}
        dones = []
        
        for state, action, reward, next_state, done in batch:
            states['image'].append(state['image'])
            states['direction'].append(state['direction'])
            actions.append(action)
            rewards.append(reward)
            next_states['image'].append(next_state['image'])
            next_states['direction'].append(next_state['direction'])
            dones.append(done)
        
        # Convert to tensors (use numpy stacking for speed and contiguous layout)
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
        
        # Next Q values from target network
        next_q_values = self.target_network(next_state_images, next_state_directions).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.update_target_network()
        
        return loss.item()
    
    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, filepath):
        """Save the model."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, filepath)
    
    def load(self, filepath):
        """Load the model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']