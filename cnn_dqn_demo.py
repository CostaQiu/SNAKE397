import pygame
import torch
import sys
import os

# Add the outdated directory to the path for SnakeGame
sys.path.append(os.path.join(os.path.dirname(__file__), 'outdated'))

from outdated.snake_game import SnakeGame, Direction
from cnn_dqn import CNNQNetwork

# --- Constants ---
WIDTH = 600
HEIGHT = 600
GAME_WIDTH_BLOCKS = 20
GAME_HEIGHT_BLOCKS = 20

def demonstrate():
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("CNN DQN Snake AI Demonstration")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 25)
    
    # Initialize game
    game = SnakeGame(w=GAME_WIDTH_BLOCKS, h=GAME_HEIGHT_BLOCKS)
    
    # Initialize model
    model = CNNQNetwork(GAME_WIDTH_BLOCKS, 3)  # 3 actions: left, straight, right
    
    # Load trained model
    if os.path.exists('cnn_dqn_model.pth'):
        model.load_state_dict(torch.load('cnn_dqn_model.pth', map_location=torch.device('cpu')))
        model.eval()
        print("Model loaded successfully!")
    else:
        print("No trained model found. Please train the model first.")
        return
    
    # Game loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
        
        # Create state representation
        grid = np.zeros((3, GAME_WIDTH_BLOCKS, GAME_HEIGHT_BLOCKS), dtype=np.float32)
        
        # Add snake head
        head_x, head_y = int(game.head.x), int(game.head.y)
        if 0 <= head_x < GAME_WIDTH_BLOCKS and 0 <= head_y < GAME_HEIGHT_BLOCKS:
            grid[0, head_y, head_x] = 1.0
            
        # Add snake body (excluding head)
        for i, segment in enumerate(game.snake[1:]):
            x, y = int(segment.x), int(segment.y)
            if 0 <= x < GAME_WIDTH_BLOCKS and 0 <= y < GAME_HEIGHT_BLOCKS:
                grid[1, y, x] = 1.0
                
        # Add food
        food_x, food_y = int(game.food.x), int(game.food.y)
        if 0 <= food_x < GAME_WIDTH_BLOCKS and 0 <= food_y < GAME_HEIGHT_BLOCKS:
            grid[2, food_y, food_x] = 1.0
        
        # Get prediction
        state_tensor = torch.tensor(grid, dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            q_values = model(state_tensor)
        
        # Get move
        action_idx = q_values.argmax().item()
        final_move = [0, 0, 0]
        final_move[action_idx] = 1
        
        # Perform move
        reward, done, score = game.play_step(final_move)
        
        if done:
            print(f"Game Over! Final Score: {score}")
            game.reset()
        
        # Draw game
        screen.fill((0, 0, 0))
        game.draw(screen)
        
        # Draw score
        score_text = font.render(f"Score: {game.score}", True, (255, 255, 255))
        screen.blit(score_text, [10, 10])
        
        pygame.display.flip()
        clock.tick(10)  # Slower for demonstration
    
    pygame.quit()

if __name__ == '__main__':
    demonstrate()