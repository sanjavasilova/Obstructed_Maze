# MiniGrid ObstructedMaze Reinforcement Learning Project

This project implements and compares Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) agents to solve challenging MiniGrid ObstructedMaze environments. The project explores various techniques including reward shaping, curriculum learning, task progression state machines, and enhanced observations to improve agent performance.

## 🎯 Environments

The project works on two different MiniGrid environments:

### 1. MiniGrid-ObstructedMaze-Full-v1 (Main Environment)
One of the most challenging environments in MiniGrid:
- **Objective**: Find the blue ball hidden in one of the 4 corners of a 3x3 maze
- **Obstacles**: Locked doors, green balls blocking paths, keys hidden in boxes
- **Observation Space**: 7x7x3 image + agent direction
- **Action Space**: 7 discrete actions (turn left, turn right, move forward, pickup, drop, toggle/open, done)
- **Reward**: 1 for success, 0 otherwise
- **Complexity**: Requires navigation, object manipulation, key finding, door unlocking, and path clearing

### 2. MiniGrid-ObstructedMaze-1Dlhb-v0 (Additional Environment)
A simplified version with fixed structure requiring a specific action sequence:
- **Objective**: Follow a specific sequence to reach the blue ball
- **Required Sequence**:
  1. Find and pick up the green ball
  2. Drop the green ball aside
  3. Find and open the box
  4. Pick up the key from the box
  5. Unlock the door with the key
  6. Drop the key
  7. Pick up the blue ball (SUCCESS)
- **Task Progression**: Uses a state machine to guide the agent through the correct sequence

## 🏗️ Project Structure

```
├── requirements.txt              # Python dependencies
├── dqn_agent.py                  # DQN agent with Double DQN and Prioritized Replay
├── ppo_agent.py                  # PPO agent with Actor-Critic architecture
├── enhanced_env_wrapper.py       # Reward shaping and Curriculum learning (Full-v1)
├── train.py                      # DQN training script (Full-v1)
├── train_ppo.py                 # PPO training script (Full-v1)
├── train_ppo_1dlhb.py           # PPO training for 1Dlhb-v0 with Task Progression
├── evaluate.py                   # DQN model evaluation
├── evaluate_ppo.py               # PPO model evaluation (Full-v1)
├── models/                       # Saved DQN models (Full-v1)
├── models_ppo/                   # Saved PPO models (Full-v1)
├── models_1dlhb/                 # Saved PPO models (1Dlhb-v0)
├── logs/                         # TensorBoard logs for DQN
├── logs_ppo/                     # TensorBoard logs for PPO (Full-v1)
└── logs_1dlhb/                   # TensorBoard logs for PPO (1Dlhb-v0)
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Training

#### DQN on Full-v1:
```bash
# Basic training (2000 episodes)
python train.py

# Train without curriculum
python train.py --no_curriculum

# Custom training parameters
python train.py --episodes 3000 --max_steps 1000 --save_interval 200
```

#### PPO on Full-v1:
```bash
# Basic training
python train_ppo.py

# Without curriculum learning
python train_ppo.py --no_curriculum --episodes 2000 --max_steps 1000

# Without curriculum AND without reward shaping
python train_ppo.py --no_curriculum --no_reward_shaping

# Custom update frequency
python train_ppo.py --no_curriculum --update_frequency 512
```

#### PPO on 1Dlhb-v0:
```bash
# Basic training (3000 episodes)
python train_ppo_1dlhb.py --mode train --episodes 3000

# Custom episodes
python train_ppo_1dlhb.py --mode train --episodes 5000
```

Training will:
- Save model checkpoints at regular intervals
- Generate TensorBoard logs for monitoring
- Create training plots showing progress
- Save final models in respective directories

### 3. Evaluation

#### DQN Evaluation:
```bash
# Evaluate trained model
python evaluate.py --model_path models/dqn_final.pth --episodes 100

# Run demonstration with GIF
python evaluate.py --model_path models/dqn_final.pth --demo --save_gif

# Live demonstration with rendering
python evaluate.py --model_path models/dqn_final.pth --demo --demo_episodes 5 --render
```

#### PPO Evaluation (Full-v1):
```bash
# Evaluate PPO model
python evaluate_ppo.py --model_path models_ppo/ppo_final.pth --episodes 100

# With rendering and GIF
python evaluate_ppo.py --model_path models_ppo/ppo_final.pth --episodes 100 --render --save_gif
```

#### PPO Evaluation (1Dlhb-v0):
```bash
# Evaluate and save best episode as GIF
python train_ppo_1dlhb.py --mode eval --model models_1dlhb/ppo_1dlhb_final.pth --save-gif

# With more episodes
python train_ppo_1dlhb.py --mode eval --model models_1dlhb/ppo_1dlhb_final.pth --save-gif --num-eval 20

# With live rendering
python train_ppo_1dlhb.py --mode eval --model models_1dlhb/ppo_1dlhb_final.pth --render --save-gif
```

### 4. Monitoring Training

```bash
# Launch TensorBoard (in separate terminal)

# View DQN logs
tensorboard --logdir logs

# View PPO logs (Full-v1)
tensorboard --logdir logs_ppo

# View PPO logs (1Dlhb-v0)
tensorboard --logdir logs_1dlhb

# View all simultaneously
tensorboard --logdir=logs,logs_ppo,logs_1dlhb
```

Then open http://localhost:6006 to view training metrics in real-time.

## 🧠 Algorithm Details

### Deep Q-Network (DQN) with Improvements
- **Double DQN**: Reduces overestimation of Q-values
- **Prioritized Experience Replay**: Focuses learning on important experiences
- **Target Network**: Stabilizes learning by using a separate target network
- **CNN Architecture**: 2 convolutional layers for processing visual observations
- **Compass Information**: Agent knows its position and relative direction to goal
- **Input**: Image (7×7×3) + Direction (one-hot) + Compass (agent position + goal direction)
- **Output**: Q-values for 7 possible actions

**Key Features:**
- Experience Replay with prioritization
- Epsilon-Greedy Exploration with adaptive decay
- Gradient Clipping to prevent exploding gradients
- Learning Rate Scheduling

**Hyperparameters:**
- Learning Rate: 1e-4 to 1e-5
- Discount Factor (γ): 0.99
- Epsilon Decay: 0.997 to 0.9995
- Batch Size: 32 to 64
- Memory Size: 100,000 to 200,000
- Target Network Update: Every 500-1000 steps (soft update)

### Proximal Policy Optimization (PPO)
- **Actor-Critic Architecture**: Combined network for policy and value function
- **Generalized Advantage Estimation (GAE)**: Improved advantage estimation
- **Clipped Objective**: Stabilizes learning by clipping policy updates
- **Action Masking**: Masks invalid actions (e.g., drop when not carrying anything)
- **CNN Architecture**: Same as DQN with compass information
- **Input**: Image (7×7×3) + Direction + Agent Position + Goal Direction
- **Output**: Action probabilities (policy) and state value

**Key Features:**
- Multiple epochs per update (6-10 epochs)
- Entropy bonus for exploration
- Value function loss for stable learning
- Gradient clipping

**Hyperparameters:**
- Learning Rate: 3e-4 to 5e-4
- Discount Factor (γ): 0.99
- GAE Lambda: 0.95
- Clip Epsilon: 0.2
- Epochs per Update: 6-10
- Batch Size: 32-64
- Update Frequency: 128-2048 steps

## 🔧 Key Techniques for Performance Improvement

### 1. Reward Shaping

#### For ObstructedMaze-Full-v1:
Sophisticated reward system that rewards important steps:
- **Key pickup**: +20.0 (critical for unlocking doors)
- **Unlocking door with key**: +50.0 (major achievement)
- **Opening box**: +3.0 (keys are in boxes)
- **Picking up green ball**: +2.0 (to remove from path)
- **Moving ball**: +3.0 (when dropped in new location)
- **Progress toward goal**: +0.05 per step toward blue ball
- **Exploration**: +0.01 for visiting new positions
- **Penalty for repetition**: -2.0 to -5.0 for repeated pickup/drop of same objects

#### For ObstructedMaze-1Dlhb-v0:
**Task Progression State Machine** - Advanced system guiding agent through correct sequence:
- **State Machine**: 10 different progress phases tracking task completion
- **Progressive Rewards**: Rewards scale with phase importance:
  - Opening box: +15.0
  - Picking up key: +25.0
  - Unlocking door: +50.0
  - Picking up blue ball: +200.0 (final goal)
- **Sequence Penalty**: -5.0 to -10.0 for actions in wrong order
- **Distance-based Shaping**: Continuous rewards/penalties based on distance to current objective
- **Stagnation Penalty**: Penalty if agent stays at same phase too long

### 2. Curriculum Learning
- Gradually increases task difficulty
- Starts with lower difficulty (0.5) and increases to maximum (1.0)
- Automatic adaptation based on success rate
- Helps agent learn progressively

### 3. Enhanced Observations (Compass Information)
- **Agent Position**: Normalized position in the grid
- **Goal Direction**: Relative direction to the goal (blue ball or current objective)
- **Carrying State**: Information about what agent is carrying
- **Step Count**: Number of steps in current episode

### 4. Action Masking (PPO)
- Masks invalid actions (e.g., drop action when not carrying anything)
- Prevents agent from learning useless behaviors
- Improves sample efficiency

### 5. Anti-Stuck Mechanisms
- Detects "spinning" behavior (turning in place)
- Penalizes repeated identical actions
- Early truncation if agent gets truly stuck
- Prevents degenerate behavior patterns

## 📊 Results and Achievements

### Experimental Results:

#### ObstructedMaze-Full-v1:
- **DQN Model**: Trained models at various episodes (250, 500, 750, 1000, 1250, 1500, 1750, final)
- **PPO Model**: Trained models at various episodes (250, 500, 750, 1000, 1250, 1500, 1750, final)
- **PPO Results**: Achieved rewards up to 88.17 in best episodes
- **Success Rate**: Varies based on training configuration and techniques used

#### ObstructedMaze-1Dlhb-v0:
- **PPO Model**: Successfully trained with Task Progression State Machine
- **Success**: Agent successfully reached the final goal (blue ball) in 1Dlhb environment
- **Demonstration**: Task Progression State Machine effectively guides agent through correct sequence:
  1. Green ball → Drop → Open box → Pick key → Unlock door → Drop key → Pick blue ball

### Training Metrics:
- TensorBoard logging for detailed monitoring (separate logs for DQN, PPO Full-v1, and PPO 1Dlhb)
- Visualizations of progress (graphs for rewards, success rate, episode lengths)
- GIF demonstrations of best episodes (including successful and failed for behavior analysis)
- Multiple experiments with different configurations (with/without curriculum learning, with/without reward shaping)

## 📈 Monitoring and Analysis

### Training Metrics
- **Episode Reward**: Immediate feedback on agent performance
- **Episode Length**: Efficiency of solution finding
- **Loss**: Neural network training stability (policy loss, value loss for PPO)
- **Epsilon**: Current exploration rate (DQN)
- **Success Rate**: Rolling 100-episode success percentage
- **Task Stage**: Current phase in task progression (1Dlhb)

### Evaluation Tools
- **Statistical Analysis**: Mean, std, min, max performance
- **Visualization**: Histograms of rewards and episode lengths
- **GIF Generation**: Visual demonstration of agent behavior
- **Model Comparison**: Compare different training checkpoints
- **Behavior Analysis**: Detect spinning, stuck behavior, and progress patterns

## 🔧 Customization

### Modify Training Parameters

#### DQN:
```python
# Adjust hyperparameters in dqn_agent.py
agent = DQNAgent(
    obs_shape, action_size,
    lr=1e-4,                 # Learning rate
    gamma=0.99,              # Discount factor
    epsilon=1.0,             # Initial exploration rate
    epsilon_decay=0.997,      # Exploration decay rate
    memory_size=100000,      # Replay buffer size
    batch_size=32,           # Training batch size
    target_update=1000       # Target network update frequency
)
```

#### PPO:
```python
# Adjust hyperparameters in ppo_agent.py or train_ppo.py
agent = PPOAgent(
    obs_shape, action_size,
    lr=3e-4,                # Learning rate
    gamma=0.99,              # Discount factor
    gae_lambda=0.95,         # GAE lambda
    clip_epsilon=0.2,        # PPO clip epsilon
    c1=0.5,                  # Value loss coefficient
    c2=0.01,                 # Entropy coefficient
    epochs=10,               # Epochs per update
    batch_size=64            # Batch size
)
```

### Network Architecture
Modify the network classes in respective agent files:
```python
# DQN: dqn_agent.py - DQNNetwork class
# PPO: ppo_agent.py - PPONetwork class
```

### Reward Shaping
Modify reward parameters in `enhanced_env_wrapper.py` (Full-v1) or `train_ppo_1dlhb.py` (1Dlhb-v0).

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size in agent files
   batch_size=16  # Instead of 32 or 64
   ```

2. **Slow Training**
   - Ensure PyTorch is using GPU if available
   - Reduce episode count for testing
   - Use fewer episodes for initial validation
   - Reduce update frequency for PPO

3. **Poor Performance**
   - Increase training episodes (try 3000-5000)
   - Adjust learning rate (try 5e-5 or 2e-4)
   - Modify exploration parameters
   - Enable/disable reward shaping or curriculum learning
   - Check reward shaping parameters

4. **Environment Issues**
   ```bash
   # Reinstall MiniGrid
   pip uninstall minigrid
   pip install minigrid==2.3.1
   ```

5. **Agent Getting Stuck**
   - Check anti-stuck mechanisms are enabled
   - Adjust reward shaping penalties
   - Increase exploration (epsilon for DQN, entropy for PPO)
   - Review action masking settings

## 📚 Further Reading

- [MiniGrid Documentation](https://minigrid.farama.org/)
- [Deep Q-Learning Paper](https://arxiv.org/abs/1312.5602)
- [Proximal Policy Optimization Paper](https://arxiv.org/abs/1707.06347)
- [PyTorch DQN Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)

## 🎓 Scientific Contributions

This project contributes:
1. **Algorithm Comparison**: Direct comparison of DQN and PPO on the same challenging environment
2. **Reward Shaping Strategy**: Detailed analysis of how different rewards affect learning
3. **Task Progression State Machine**: Innovative approach for 1Dlhb environment guiding agent through correct action sequence
4. **Compass Information**: Investigation of the effect of adding positional information
5. **Curriculum Learning**: Implementation and evaluation of curriculum for complex tasks
6. **Action Masking**: Implementation of masking invalid actions in PPO for improved efficiency
7. **Anti-Stuck Mechanisms**: Development of systems for detecting and preventing degenerate behavior
8. **Environment Comparison**: Research on two different environments (Full-v1 and 1Dlhb-v0) with different approaches

## 🏆 Key Achievements

- Successfully implemented and compared DQN and PPO algorithms
- Developed sophisticated reward shaping systems for both environments
- Implemented Task Progression State Machine for 1Dlhb-v0
- Achieved successful completion in 1Dlhb-v0 environment (agent reached blue ball)
- PPO agent on Full-v1 achieved rewards up to 88.17
- Comprehensive evaluation and visualization tools
- Multiple successful training runs with different configurations

**Happy Learning! 🎓🤖**

*This project demonstrates reinforcement learning principles using challenging grid-world environments. The ObstructedMaze environments test the agent's ability to navigate complex spaces, manipulate objects, follow sequences, and achieve long-term goals.*
