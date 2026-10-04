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

# CNN DQN hyperparameters
MAX_MEMORY = 100_000
BATCH_SIZE = 512
LR = 0.0005
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 5
LEARNING_START = 1000

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNNQNetwork(nn.Module):
    """CNN-based Q-Network for processing grid-based state"""
    def __init__(self, grid_size, action_size):
        super(CNNQNetwork, self).__init__()
        self.grid_size = grid_size
        
        # CNN layers for processing the grid
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # 3 channels: snake head, snake body, food
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Calculate the size of the flattened features after conv layers
        conv_output_size = 64 * grid_size * grid_size
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_size)
        
    def forward(self, x):
        # x shape: (batch_size, 3, grid_size, grid_size)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten the conv output
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

class CNNDQNAgent:
    def __init__(self, grid_size, action_size):
        self.grid_size = grid_size
        self.action_size = action_size
        self.n_games = 0
        self.epsilon = EPSILON_START
        self.total_steps = 0
        self.memory = deque(maxlen=MAX_MEMORY)
        
        # CNN-based models
        self.q_network = CNNQNetwork(grid_size, action_size).to(device)
        self.target_network = CNNQNetwork(grid_size, action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LR)
        
    def get_state(self, game):
        """
        Convert game state to CNN-friendly format (3-channel grid)
        Channel 0: Snake head (1.0 at head position, 0 elsewhere)
        Channel 1: Snake body (1.0 at body positions, 0 elsewhere)
        Channel 2: Food (1.0 at food position, 0 elsewhere)
        """
        # Create 3-channel grid
        grid = np.zeros((3, self.grid_size, self.grid_size), dtype=np.float32)
        
        # Add snake head
        head_x, head_y = int(game.head.x), int(game.head.y)
        if 0 <= head_x < self.grid_size and 0 <= head_y < self.grid_size:
            grid[0, head_y, head_x] = 1.0
            
        # Add snake body (excluding head)
        for i, segment in enumerate(game.snake[1:]):
            x, y = int(segment.x), int(segment.y)
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                grid[1, y, x] = 1.0
                
        # Add food
        food_x, food_y = int(game.food.x), int(game.food.y)
        if 0 <= food_x < self.grid_size and 0 <= food_y < self.grid_size:
            grid[2, food_y, food_x] = 1.0
            
        return grid
    
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
        current_q_values = self.q_network(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (GAMMA * next_q_values * ~dones)
            
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def get_action(self, state):
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Random action
            return random.randint(0, self.action_size - 1)
        else:
            # Best action according to Q-network
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

def train():
    # Initialize agent
    agent = CNNDQNAgent(GAME_WIDTH_BLOCKS, 3)  # 3 actions: left, straight, right
    
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
    pygame.display.set_caption("CNN DQN Snake AI Training")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 25)
    
    record = 0
    
    # Training loop
    while True:
        # Get current state
        state = agent.get_state(game)
        
        # Get action
        action_idx = agent.get_action(state)
        final_move = [0, 0, 0]
        final_move[action_idx] = 1
        
        # Perform move
        reward, done, score = game.play_step(final_move)
        
        # Get next state
        next_state = agent.get_state(game)
        
        # Store experience
        agent.remember(state, action_idx, reward, next_state, done)
        
        # Increment counters
        agent.total_steps += 1
        
        if done:
            # Reset game
            game.reset()
            agent.n_games += 1
            
            # Update target network periodically
            if agent.n_games % TARGET_UPDATE_FREQ == 0:
                agent.target_network.load_state_dict(agent.q_network.state_dict())
            
            # Start training after collecting enough experiences
            if agent.total_steps > LEARNING_START:
                agent.train_long_memory()
            
            # Update record and save model
            if score > record:
                record = score
                best_score = score
                save_best_score(best_score)
                torch.save(agent.q_network.state_dict(), 'cnn_dqn_model.pth')
                
            print(f'Game {agent.n_games} | Score {score} | Record {record} | Epsilon {agent.epsilon:.3f}')
            
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