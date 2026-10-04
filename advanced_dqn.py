import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import sys
from collections import deque
import math

# Add the outdated directory to the path for SnakeGame
sys.path.append(os.path.join(os.path.dirname(__file__), 'outdated'))

from outdated.snake_game import SnakeGame, Direction
from outdated.utils import save_best_score

# --- Constants ---
WIDTH = 600
HEIGHT = 600
GAME_WIDTH_BLOCKS = 20
GAME_HEIGHT_BLOCKS = 20

# Advanced DQN hyperparameters
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
GAMMA = 0.95  # Higher discount rate for long-term planning
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 10  # More frequent updates
LEARNING_START = 1000  # Start learning after collecting some experiences

# Improved exploration
NOISY_NETS = True  # Use noisy networks for better exploration
PRIORITY_REPLAY = True  # Use prioritized experience replay

# Multi-step learning
N_STEP = 3  # N-step Q-learning

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NoisyLinear(nn.Module):
    """Noisy linear layer for better exploration"""
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
        
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
        
    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x
    
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, noisy_nets=False):
        super(DQN, self).__init__()
        self.noisy_nets = noisy_nets
        
        if noisy_nets:
            self.fc1 = NoisyLinear(input_size, hidden_size)
            self.fc2 = NoisyLinear(hidden_size, hidden_size)
            self.fc3 = NoisyLinear(hidden_size, output_size)
        else:
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def reset_noise(self):
        if self.noisy_nets:
            self.fc1.reset_noise()
            self.fc2.reset_noise()
            self.fc3.reset_noise()

class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer"""
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.position = 0
        
    def push(self, experience, priority=1.0):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority
            
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return None, None, None
            
        priorities = np.array(self.priorities[:len(self.buffer)])
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        samples = [self.buffer[idx] for idx in indices]
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        
        return samples, indices, weights
        
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = priority + 1e-5  # Small epsilon to keep priority positive
                
    def __len__(self):
        return len(self.buffer)

class AdvancedDQNAgent:
    def __init__(self, input_size, hidden_size, output_size):
        self.n_games = 0
        self.epsilon = EPSILON_START
        self.total_steps = 0
        
        # Use prioritized replay if enabled
        if PRIORITY_REPLAY:
            self.memory = PrioritizedReplayBuffer(MAX_MEMORY)
        else:
            self.memory = deque(maxlen=MAX_MEMORY)
            
        # N-step buffer for multi-step learning
        self.n_step_buffer = deque(maxlen=N_STEP)
        
        # Models
        self.model = DQN(input_size, hidden_size, output_size, noisy_nets=NOISY_NETS).to(device)
        self.target_model = DQN(input_size, hidden_size, output_size, noisy_nets=NOISY_NETS).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        
    def remember(self, state, action, reward, next_state, done):
        # For n-step learning
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # If we have enough steps for n-step learning
        if len(self.n_step_buffer) == N_STEP:
            n_step_state, n_step_action = self.n_step_buffer[0][:2]
            n_step_reward, n_step_next_state, n_step_done = self._get_n_step_info()
            
            # Calculate priority (TD error)
            if PRIORITY_REPLAY:
                with torch.no_grad():
                    current_q = self.model(torch.tensor(n_step_state, dtype=torch.float).unsqueeze(0).to(device))
                    current_q_value = current_q[0][torch.argmax(torch.tensor(n_step_action)).item()]
                    
                    if n_step_done:
                        target_q_value = n_step_reward
                    else:
                        next_q = self.target_model(torch.tensor(n_step_next_state, dtype=torch.float).unsqueeze(0).to(device))
                        target_q_value = n_step_reward + (GAMMA ** N_STEP) * torch.max(next_q)
                        
                    td_error = abs(current_q_value - target_q_value).item()
                    priority = td_error + 1e-5
                    
                self.memory.push((n_step_state, n_step_action, n_step_reward, n_step_next_state, n_step_done), priority)
            else:
                self.memory.append((n_step_state, n_step_action, n_step_reward, n_step_next_state, n_step_done))
                
    def _get_n_step_info(self):
        """Calculate n-step reward, next_state, and done"""
        reward, next_state, done = self.n_step_buffer[-1][-3:]
        
        # Calculate n-step reward
        n_step_reward = 0
        for i, (state, action, r, next_s, d) in enumerate(self.n_step_buffer):
            n_step_reward += (GAMMA ** i) * r
            if d:
                next_state = next_s
                done = True
                break
                
        return n_step_reward, next_state, done
        
    def train_long_memory(self):
        # Wait until we have enough experiences
        if (PRIORITY_REPLAY and len(self.memory) < BATCH_SIZE) or \
           (not PRIORITY_REPLAY and len(self.memory) < BATCH_SIZE):
            return
            
        # Sample batch
        if PRIORITY_REPLAY:
            samples, indices, weights = self.memory.sample(BATCH_SIZE)
            if samples is None:
                return
            states, actions, rewards, next_states, dones = zip(*samples)
            weights = torch.tensor(weights, dtype=torch.float).to(device)
        else:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*mini_sample)
            weights = None
            
        self.train_step(states, actions, rewards, next_states, dones, weights)
        
        # Update priorities if using prioritized replay
        if PRIORITY_REPLAY and indices is not None:
            with torch.no_grad():
                states_tensor = torch.tensor(np.array(states), dtype=torch.float).to(device)
                next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float).to(device)
                actions_tensor = torch.tensor(actions, dtype=torch.long).to(device)
                rewards_tensor = torch.tensor(rewards, dtype=torch.float).to(device)
                dones_tensor = torch.tensor(dones, dtype=torch.bool).to(device)
                
                current_q = self.model(states_tensor)
                current_q_values = current_q.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
                
                next_q = self.target_model(next_states_tensor)
                next_q_values = next_q.max(1)[0]
                target_q_values = rewards_tensor + (GAMMA ** N_STEP) * next_q_values * (~dones_tensor)
                
                td_errors = torch.abs(current_q_values - target_q_values).cpu().numpy()
                self.memory.update_priorities(indices, td_errors)
        
    def train_step(self, states, actions, rewards, next_states, dones, weights=None):
        states = torch.tensor(np.array(states), dtype=torch.float).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        dones = torch.tensor(dones, dtype=torch.bool).to(device)
        
        # Current Q values
        pred = self.model(states)
        target = pred.clone()
        
        with torch.no_grad():
            next_q = self.target_model(next_states)
            
        for idx in range(len(dones)):
            Q_new = rewards[idx]
            if not dones[idx]:
                Q_new = rewards[idx] + (GAMMA ** N_STEP) * torch.max(next_q[idx])
                
            target[idx][torch.argmax(actions[idx]).item()] = Q_new
            
        # Loss calculation
        self.optimizer.zero_grad()
        
        if weights is not None:
            # Weighted loss for prioritized replay
            loss = (weights * F.mse_loss(pred.gather(1, actions.unsqueeze(1)).squeeze(1), target.gather(1, actions.unsqueeze(1)).squeeze(1), reduction='none')).mean()
        else:
            loss = F.mse_loss(pred.gather(1, actions.unsqueeze(1)).squeeze(1), target.gather(1, actions.unsqueeze(1)).squeeze(1))
            
        loss.backward()
        self.optimizer.step()
        
    def get_action(self, state):
        # With noisy nets, we don't need epsilon-greedy
        if NOISY_NETS:
            state_tensor = torch.tensor(np.array(state), dtype=torch.float).unsqueeze(0).to(device)
            prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()
            final_move = [0, 0, 0]
            final_move[move] = 1
        else:
            # Epsilon-greedy
            final_move = [0, 0, 0]
            if random.random() < self.epsilon:
                move = random.randint(0, 2)
                final_move[move] = 1
            else:
                state_tensor = torch.tensor(np.array(state), dtype=torch.float).unsqueeze(0).to(device)
                prediction = self.model(state_tensor)
                move = torch.argmax(prediction).item()
                final_move[move] = 1
                
        return final_move

def get_state(game):
    """
    Enhanced state representation for better learning
    """
    # Get head position
    head_x, head_y = game.head.x, game.head.y
    
    # 1. Check for danger in three directions (straight, left, right)
    danger_straight = game._is_danger(game.direction)
    
    # Calculate left direction
    left_dir = game.direction
    if left_dir == Direction.UP:
        left_dir = Direction.LEFT
    elif left_dir == Direction.LEFT:
        left_dir = Direction.DOWN
    elif left_dir == Direction.DOWN:
        left_dir = Direction.RIGHT
    elif left_dir == Direction.RIGHT:
        left_dir = Direction.UP
    
    danger_left = game._is_danger(left_dir)
    
    # Calculate right direction
    right_dir = game.direction
    if right_dir == Direction.UP:
        right_dir = Direction.RIGHT
    elif right_dir == Direction.RIGHT:
        right_dir = Direction.DOWN
    elif right_dir == Direction.DOWN:
        right_dir = Direction.LEFT
    elif right_dir == Direction.LEFT:
        right_dir = Direction.UP
    
    danger_right = game._is_danger(right_dir)
    
    # 2. Current direction
    dir_up = 1 if game.direction == Direction.UP else 0
    dir_right = 1 if game.direction == Direction.RIGHT else 0
    dir_down = 1 if game.direction == Direction.DOWN else 0
    dir_left = 1 if game.direction == Direction.LEFT else 0
    
    # 3. Food direction (relative to snake head)
    food_left = 1 if game.food.x < head_x else 0
    food_right = 1 if game.food.x > head_x else 0
    food_up = 1 if game.food.y < head_y else 0
    food_down = 1 if game.food.y > head_y else 0
    
    # 4. Additional features for better decision making
    # Distance to food (normalized)
    dist_x = abs(game.food.x - head_x) / game.w
    dist_y = abs(game.food.y - head_y) / game.h
    
    # Snake length (normalized)
    snake_length = len(game.snake) / (game.w * game.h)
    
    return [
        danger_straight,
        danger_left,
        danger_right,
        dir_up,
        dir_right,
        dir_down,
        dir_left,
        food_up,
        food_down,
        food_left,
        food_right,
        dist_x,
        dist_y,
        snake_length
    ]

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    best_score = 0
    
    # Initialize agent with enhanced state representation
    agent = AdvancedDQNAgent(input_size=14, hidden_size=512, output_size=3)
    
    # Load best score if exists
    if os.path.exists("best_score.txt"):
        with open("best_score.txt", "r") as f:
            best_score = int(f.read())
    
    # Initialize game
    game = SnakeGame(w=GAME_WIDTH_BLOCKS, h=GAME_HEIGHT_BLOCKS)
    
    # Pygame setup for visualization (optional)
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Advanced DQN Snake AI Training")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 25)
    
    while True:
        # Get current state
        state_old = get_state(game)
        
        # Get move
        final_move = agent.get_action(state_old)
        
        # Perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = get_state(game)
        
        # Remember experience
        agent.remember(state_old, final_move, reward, state_new, done)
        
        # Increment step counter
        agent.total_steps += 1
        
        if done:
            # Train long memory (experience replay)
            game.reset()
            agent.n_games += 1
            
            # Update target network periodically
            if agent.n_games % TARGET_UPDATE_FREQ == 0:
                agent.target_model.load_state_dict(agent.model.state_dict())
                
            # Reset noise for noisy networks
            if NOISY_NETS:
                agent.model.reset_noise()
                agent.target_model.reset_noise()
            
            # Start training after collecting enough experiences
            if agent.total_steps > LEARNING_START:
                agent.train_long_memory()
            
            if score > record:
                record = score
                best_score = score
                save_best_score(best_score)
                # Save model
                torch.save(agent.model.state_dict(), 'advanced_dqn_model.pth')
                
            print('Game', agent.n_games, 'Score', score, 'Record:', record, 
                  'Epsilon:', f"{agent.epsilon:.3f}" if not NOISY_NETS else "N/A",
                  'Steps:', agent.total_steps)
            
            # Decay epsilon (if not using noisy nets)
            if not NOISY_NETS and agent.epsilon > EPSILON_END:
                agent.epsilon *= EPSILON_DECAY
                agent.epsilon = max(EPSILON_END, agent.epsilon)
                
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
                
        # Draw game
        screen.fill((0, 0, 0))
        game.draw(screen)
        
        # Draw info
        info_text = f"Score: {game.score} | Record: {record} | Game: {agent.n_games}"
        score_text = font.render(info_text, True, (255, 255, 255))
        screen.blit(score_text, [10, 10])
        
        if not NOISY_NETS:
            epsilon_text = font.render(f"Epsilon: {agent.epsilon:.3f}", True, (255, 255, 255))
            screen.blit(epsilon_text, [10, 30])
        
        pygame.display.flip()
        clock.tick(60)  # 60 FPS

if __name__ == '__main__':
    train()