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

# DQN hyperparameters
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
GAMMA = 0.9  # discount rate
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 100  # update target network every 100 games

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, input_size, hidden_size, output_size):
        self.n_games = 0
        self.epsilon = EPSILON_START  # randomness
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = DQN(input_size, hidden_size, output_size).to(device)
        self.target_model = DQN(input_size, hidden_size, output_size).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached
        
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.train_step(states, actions, rewards, next_states, dones)
        
    def train_short_memory(self, state, action, reward, next_state, done):
        self.train_step([state], [action], [reward], [next_state], [done])
        
    def train_step(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(np.array(states), dtype=torch.float).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        dones = torch.tensor(dones, dtype=torch.bool).to(device)
        
        # Current Q values
        pred = self.model(states)
        target = pred.clone()
        
        for idx in range(len(dones)):
            Q_new = rewards[idx]
            if not dones[idx]:
                Q_new = rewards[idx] + GAMMA * torch.max(self.target_model(next_states[idx]))
            
            target[idx][torch.argmax(actions[idx]).item()] = Q_new
        
        # Loss and backpropagation
        self.optimizer.zero_grad()
        loss = F.mse_loss(target, pred)
        loss.backward()
        self.optimizer.step()
        
    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
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
    Get the state of the game as input for the neural network.
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
        food_right
    ]

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    best_score = 0
    
    # Initialize agent
    agent = DQNAgent(input_size=11, hidden_size=256, output_size=3)
    
    # Load best score if exists
    if os.path.exists("best_score.txt"):
        with open("best_score.txt", "r") as f:
            best_score = int(f.read())
    
    # Initialize game
    game = SnakeGame(w=GAME_WIDTH_BLOCKS, h=GAME_HEIGHT_BLOCKS)
    
    # Pygame setup for visualization (optional)
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("DQN Snake AI Training")
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
        
        # Train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        
        # Remember
        agent.remember(state_old, final_move, reward, state_new, done)
        
        if done:
            # Train long memory (experience replay)
            game.reset()
            agent.n_games += 1
            
            # Update target network periodically
            if agent.n_games % TARGET_UPDATE_FREQ == 0:
                agent.target_model.load_state_dict(agent.model.state_dict())
            
            agent.train_long_memory()
            
            if score > record:
                record = score
                best_score = score
                save_best_score(best_score)
                # Save model
                torch.save(agent.model.state_dict(), 'dqn_model.pth')
                
            print('Game', agent.n_games, 'Score', score, 'Record:', record, 'Epsilon:', agent.epsilon)
            
            # Decay epsilon
            if agent.epsilon > EPSILON_END:
                agent.epsilon *= EPSILON_DECAY
                agent.epsilon = max(EPSILON_END, agent.epsilon)
                
        # Draw game
        screen.fill((0, 0, 0))
        game.draw(screen)
        
        # Draw info
        score_text = font.render(f"Score: {game.score} | Record: {record} | Game: {agent.n_games}", True, (255, 255, 255))
        screen.blit(score_text, [10, 10])
        
        epsilon_text = font.render(f"Epsilon: {agent.epsilon:.3f}", True, (255, 255, 255))
        screen.blit(epsilon_text, [10, 30])
        
        pygame.display.flip()
        clock.tick(60)  # 60 FPS

if __name__ == '__main__':
    train()