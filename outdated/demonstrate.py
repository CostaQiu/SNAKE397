# demonstrate.py
import pygame
import neat
import os
import time
# Import SnakeGame and necessary constants from snake_game
from snake_game import SnakeGame, BLACK, WHITE, RED, GREEN # Import what's needed
from utils import load_best_genome
# Removed Enum, namedtuple imports as they are handled within SnakeGame now

# Define font here, but initialize after pygame.init()
font = None

# --- Constants ---
WIDTH, HEIGHT = 600, 600 # Single screen for demo
GAME_WIDTH_BLOCKS = 20  # Can use a larger grid for demo
GAME_HEIGHT_BLOCKS = 20
BLOCK_SIZE = WIDTH // GAME_WIDTH_BLOCKS # BLOCK_SIZE seems unused? SnakeGame calculates it internally based on screen size. Let's remove the argument unless needed.

def run_demonstration(config_path):
    """Loads the best genome and runs the game controlled by it."""
    global font # Access the global font variable

    # --- Pygame Setup ---
    pygame.init() # <<< MOVE pygame.init() HERE
    font = pygame.font.Font(None, 25) # <<< Initialize font AFTER init
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("NEAT Snake AI Demonstration")
    clock = pygame.time.Clock()

    # Load NEAT config
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Load the best genome
    best_genome = load_best_genome()
    if not best_genome:
        print("Could not load best_genome.pkl. Train a model first.")
        pygame.quit() # Quit pygame if genome not loaded
        return

    # Create network from the genome
    net = neat.nn.FeedForwardNetwork.create(best_genome, config)

    # --- Game Setup ---
    # Pass the actual screen dimensions to SnakeGame if block_size is determined by it
    # Or let SnakeGame use its defaults if you prefer fixed grid size regardless of window
    game_w_pixels = BLOCK_SIZE * GAME_WIDTH_BLOCKS
    game_h_pixels = BLOCK_SIZE * GAME_HEIGHT_BLOCKS
    # Adjust game creation if block_size matters for drawing coordinates outside SnakeGame
    # Since drawing is inside SnakeGame, it likely just needs w/h in blocks
    game = SnakeGame(w=GAME_WIDTH_BLOCKS, h=GAME_HEIGHT_BLOCKS) # Removed block_size argument

    # --- Game Loop ---
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                 if event.key == pygame.K_q:
                     running = False

        # Get state and AI action
        state = game.get_state()
        output = net.activate(state)
        action = [0, 0, 0]
        action_idx = output.index(max(output))
        action[action_idx] = 1

        # Perform game step
        reward, game_over, score = game.play_step(action)

        if game_over:
            print(f"Game Over! Final Score: {score}")
            time.sleep(2)
            game = SnakeGame(w=GAME_WIDTH_BLOCKS, h=GAME_HEIGHT_BLOCKS) # Reset

        # --- Drawing ---
        # Create a surface for the game area if smaller than window
        game_surface = pygame.Surface((WIDTH, HEIGHT)) # Assuming game fills window
        game.draw(game_surface, draw_grid=True) # Draw game onto its surface
        screen.blit(game_surface, (0,0)) # Blit game surface to main screen

        # Draw score on top using the initialized font
        score_text = font.render(f"Score: {game.score}", True, WHITE)
        screen.blit(score_text, [10, 10])

        pygame.display.flip()
        clock.tick(15)

    pygame.quit()