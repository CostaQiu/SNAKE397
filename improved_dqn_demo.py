import pygame
import torch
import sys
import os

# Add the outdated directory to the path for SnakeGame
sys.path.append(os.path.join(os.path.dirname(__file__), 'outdated'))

from outdated.snake_game import SnakeGame, Direction
from improved_dqn import ImprovedDQN, get_improved_state

# --- Constants ---
WIDTH = 600
HEIGHT = 600
GAME_WIDTH_BLOCKS = 20
GAME_HEIGHT_BLOCKS = 20

def demonstrate():
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Improved DQN Snake AI Demonstration")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 25)
    
    # Initialize game
    game = SnakeGame(w=GAME_WIDTH_BLOCKS, h=GAME_HEIGHT_BLOCKS)
    
    # Initialize model
    model = ImprovedDQN(15, 512, 3)
    
    # Load trained model
    if os.path.exists('improved_dqn_model.pth'):
        model.load_state_dict(torch.load('improved_dqn_model.pth', map_location=torch.device('cpu')))
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
        
        # Get current state
        state = get_improved_state(game)
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        
        # Get prediction
        with torch.no_grad():
            prediction = model(state_tensor)
        
        # Get move
        move = torch.argmax(prediction).item()
        final_move = [0, 0, 0]
        final_move[move] = 1
        
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