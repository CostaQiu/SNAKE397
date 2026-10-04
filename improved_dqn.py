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

# Improved DQN hyperparameters
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
GAMMA = 0.9
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 10
LEARNING_START = 1000

# Reward shaping parameters
DISTANCE_REWARD = 0.1
SURVIVAL_REWARD = 0.01
FOOD_REWARD = 10.0
DEATH_PENALTY = -10.0
SELF_DEATH_PENALTY = -15.0  # Stronger penalty for hitting itself
TIME_PENALTY = -0.01  # Small penalty for each step to encourage efficiency

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ImprovedDQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ImprovedDQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class ImprovedDQNAgent:
    def __init__(self, input_size, hidden_size, output_size):
        self.n_games = 0
        self.epsilon = EPSILON_START
        self.total_steps = 0
        self.memory = deque(maxlen=MAX_MEMORY)
        
        # Models
        self.model = ImprovedDQN(input_size, hidden_size, output_size).to(device)
        self.target_model = ImprovedDQN(input_size, hidden_size, output_size).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def train_long_memory(self):
        if len(self.memory) < BATCH_SIZE:
            return
            
        # Sample batch
        mini_sample = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        
        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        dones = torch.tensor(dones, dtype=torch.bool).to(device)
        
        # Current Q values
        pred = self.model(states)
        target = pred.clone()
        
        # Calculate target Q values
        with torch.no_grad():
            next_q = self.target_model(next_states)
            
        for idx in range(len(dones)):
            Q_new = rewards[idx]
            if not dones[idx]:
                Q_new = rewards[idx] + GAMMA * torch.max(next_q[idx])
                
            target[idx][actions[idx]] = Q_new
            
        # Loss and backpropagation
        self.optimizer.zero_grad()
        loss = F.mse_loss(pred, target)
        loss.backward()
        self.optimizer.step()
        
    def get_action(self, state):
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        else:
            state_tensor = torch.tensor(np.array(state), dtype=torch.float).unsqueeze(0).to(device)
            prediction = self.model(state_tensor)
            return torch.argmax(prediction).item()

def get_improved_state(game):
    """
    Enhanced state representation with more features
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
    
    # 5. Manhattan distance to food (normalized)
    manhattan_dist = (abs(game.food.x - head_x) + abs(game.food.y - head_y)) / (game.w + game.h)
    
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
        snake_length,
        manhattan_dist
    ]

def train():
    # Initialize agent with enhanced state representation
    agent = ImprovedDQNAgent(input_size=15, hidden_size=512, output_size=3)
    
    # Load best score if exists
    best_score = 0
    if os.path.exists("best_score.txt"):
        with open("best_score.txt", "r") as f:
            best_score = int(f.read())
    
    # Initialize game
    game = SnakeGame(w=GAME_WIDTH_BLOCKS, h=GAME_HEIGHT_BLOCKS)
    
    # Pygame setup for visualization
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Improved DQN Snake AI Training")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 25)
    
    record = 0
    total_steps = 0
    
    # Training loop
    while True:
        # Get current state
        state_old = get_improved_state(game)
        
        # Get move
        action_idx = agent.get_action(state_old)
        final_move = [0, 0, 0]
        final_move[action_idx] = 1
        
        # Perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = get_improved_state(game)
        
        # Modify reward based on improved reward structure
        # Calculate distance to food before and after move
        head_x, head_y = game.head.x, game.head.y
        old_dist = abs(game.food.x - state_old[0]) + abs(game.food.y - state_old[1])
        new_dist = abs(game.food.x - head_x) + abs(game.food.y - head_y)
        
        # Reward for getting closer to food
        if new_dist < old_dist:
            reward += DISTANCE_REWARD
        elif new_dist > old_dist:
            reward -= DISTANCE_REWARD
            
        # Small reward for survival
        reward += SURVIVAL_REWARD
        
        # Time penalty to encourage efficiency
        reward += TIME_PENALTY
        
        # Remember experience
        agent.remember(state_old, action_idx, reward, state_new, done)
        
        # Increment counters
        agent.total_steps += 1
        total_steps += 1
        
        if done:
            # Reset game
            game.reset()
            agent.n_games += 1
            
            # Update target network periodically
            if agent.n_games % TARGET_UPDATE_FREQ == 0:
                agent.target_model.load_state_dict(agent.model.state_dict())
            
            # Start training after collecting enough experiences
            if agent.total_steps > LEARNING_START:
                agent.train_long_memory()
            
            # Update record and save model
            if score > record:
                record = score
                best_score = score
                save_best_score(best_score)
                torch.save(agent.model.state_dict(), 'improved_dqn_model.pth')
                
            print(f'Game {agent.n_games} | Score {score} | Record {record} | Epsilon {agent.epsilon:.3f} | Steps {total_steps}')
            
            # Decay epsilon
            if agent.epsilon > EPSILON_END:
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
        info_text = f"Score: {game.score} | Record: {record} | Game: {agent.n_games} | Epsilon: {agent.epsilon:.3f}"
        score_text = font.render(info_text, True, (255, 255, 255))
        screen.blit(score_text, [10, 10])
        
        pygame.display.flip()
        clock.tick(60)

if __name__ == '__main__':
    train()