# MiniGrid ObstructedMaze-Full Reinforcement Learning Project

This project implements a Double Deep Q-Network (DDQN) agent to solve the MiniGrid ObstructedMaze-Full environment. In this challenging environment, a blue ball is hidden in one of the 4 corners of a 3x3 maze, with locked doors, obstructed paths, and keys hidden in boxes.

## 🎯 Environment Description

The **ObstructedMaze-Full** environment from MiniGrid presents several challenges:
- **Objective**: Find the blue ball hidden in one of the 4 corners
- **Obstacles**: Locked doors, balls blocking paths, keys hidden in boxes
- **Observation Space**: 7x7x3 image + agent direction
- **Action Space**: 7 discrete actions (left, right, forward, pickup, drop, toggle, done)
- **Reward**: 1 for success, 0 otherwise

## 🏗️ Project Structure

```
├── requirements.txt          # Python dependencies
├── dqn_agent.py             # DQN agent implementation
├── train.py                 # Training script
├── evaluate.py              # Evaluation and visualization
├── README.md                # This file
├── models/                  # Saved model checkpoints
├── logs/                    # TensorBoard logs
└── *.png, *.gif            # Generated plots and demonstrations
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Training

```bash
# Basic training (2000 episodes)
python train.py

# Train without curriculum
python train.py --no_curriculum

# Custom training parameters
python train.py --episodes 3000 --max_steps 500 --save_interval 200

# Without curriculum learning
python train_ppo.py --no_curriculum --episodes 2000 --max_steps 1000

# Without curriculum AND without reward shaping
python train_ppo.py --no_curriculum --no_reward_shaping

# Without curriculum but with specific update frequency
python train_ppo.py --no_curriculum --update_frequency 2048
```

Training will:
- Save model checkpoints every 200 episodes
- Generate TensorBoard logs for monitoring
- Create training plots showing progress
- Save the final model as `models/dqn_obstructed_maze_final.pth`

### 3. Evaluation

```bash
# Evaluate trained model
python evaluate.py --model_path models/dqn_final.pth --episodes 100

# Run demonstration with GIF
python evaluate.py --model_path models/dqn_final.pth --demo --save_gif

# Live demonstration with rendering
python evaluate.py --model_path models/dqn_final.pth --demo --demo_episodes 5 --render

# Live demonstration with rendering without curriculum
python evaluate.py --model_path models/dqn_final.pth --no_curriculum --demo --demo_episodes 5 --render

# Live demonstration with rendering without curriculum and top episodes
python evaluate.py --model_path models/dqn_final.pth --no_curriculum --demo --demo_episodes 2000 --demo_top_k 2 --save_gif

# Run 10 episodes, save best as GIF (no live rendering - fast)
python train_ppo_1dlhb.py --mode eval --model models_1dlhb/ppo_1dlhb_final.pth --save-gif

# Run 20 episodes, save best as GIF
python train_ppo_1dlhb.py --mode eval --model models_1dlhb/ppo_1dlhb_final.pth --save-gif --num-eval 20

# Run with live rendering AND save GIF
python train_ppo_1dlhb.py --mode eval --model models_1dlhb/ppo_1dlhb_final.pth --render --save-gif
```

### 4. Monitoring Training

```bash
# Launch TensorBoard (in separate terminal)

# View PPO logs
tensorboard --logdir logs_ppo

# View DQN logs
tensorboard --logdir logs

# View both simultaneously
tensorboard --logdir=logs,logs_ppo

```

Then open http://localhost:6006 to view training metrics in real-time.

## 🧠 Algorithm Details

### Double Deep Q-Network Architecture
- **Convolutional layers**: Process 7x7x3 visual observations
- **Fully connected layers**: Combine visual features with direction information
- **Input**: Image (7×7×3) + Direction (one-hot encoded)
- **Output**: Q-values for 7 possible actions

### Key Features
- **Experience Replay**: Stores and samples past experiences for stable learning
- **Target Network**: Reduces correlation in Q-value updates
- **Epsilon-Greedy Exploration**: Balances exploration vs exploitation
- **Gradient Clipping**: Prevents exploding gradients
- **Adaptive Learning**: Epsilon decay for reduced exploration over time

### Hyperparameters
- Learning Rate: 1e-4
- Discount Factor (γ): 0.99
- Epsilon Decay: 0.995
- Batch Size: 32
- Memory Size: 50,000
- Target Network Update: Every 1000 steps

## 📊 Expected Results

After training, you should expect:
- **Success Rate**: 60-80% in the final episodes
- **Average Episode Length**: 100-200 steps
- **Training Time**: ~30-60 minutes (depending on hardware)

### Training Progress
The agent typically shows:
1. **Initial Phase** (0-500 episodes): Random exploration, low success rate
2. **Learning Phase** (500-1500 episodes): Rapid improvement in success rate
3. **Convergence** (1500+ episodes): Stable performance with occasional improvements

## 🔧 Customization

### Modify Training Parameters
Edit `train.py` or use command line arguments:
```python
# Adjust hyperparameters in dqn_agent.py
agent = DQNAgent(
    obs_shape, action_size,
    lr=1e-4,                 # Learning rate
    gamma=0.99,              # Discount factor
    epsilon=1.0,             # Initial exploration rate
    epsilon_decay=0.995,     # Exploration decay rate
    memory_size=50000,       # Replay buffer size
    batch_size=32,           # Training batch size
    target_update=1000       # Target network update frequency
)
```

### Network Architecture
Modify the `DQNNetwork` class in `dqn_agent.py`:
```python
# Adjust hidden layers
self.fc1 = nn.Linear(total_input_size, hidden_size)
self.fc2 = nn.Linear(hidden_size, hidden_size)
```

## 📈 Monitoring and Analysis

### Training Metrics
- **Episode Reward**: Immediate feedback on agent performance
- **Episode Length**: Efficiency of solution finding
- **Loss**: Neural network training stability
- **Epsilon**: Current exploration rate
- **Success Rate**: Rolling 100-episode success percentage

### Evaluation Tools
- **Statistical Analysis**: Mean, std, min, max performance
- **Visualization**: Histograms of rewards and episode lengths
- **GIF Generation**: Visual demonstration of agent behavior
- **Model Comparison**: Compare different training checkpoints

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size in dqn_agent.py
   batch_size=16  # Instead of 32
   ```

2. **Slow Training**
   - Ensure PyTorch is using GPU if available
   - Reduce episode count for testing
   - Use fewer episodes for initial validation

3. **Poor Performance**
   - Increase training episodes (try 3000-5000)
   - Adjust learning rate (try 5e-5 or 2e-4)
   - Modify exploration parameters

4. **Environment Issues**
   ```bash
   # Reinstall MiniGrid
   pip uninstall minigrid
   pip install minigrid==2.3.1
   ```

## 📚 Further Reading

- [MiniGrid Documentation](https://minigrid.farama.org/)
- [Deep Q-Learning Paper](https://arxiv.org/abs/1312.5602)
- [PyTorch DQN Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)

**Happy Learning! 🎓🤖**

*This project demonstrates reinforcement learning principles using a challenging grid-world environment. The ObstructedMaze-Full environment tests the agent's ability to navigate complex spaces, manipulate objects, and achieve long-term goals.*