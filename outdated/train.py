# train.py
import pygame
import neat
import os
import time
import math
# Import necessary things from snake_game
from snake_game import SnakeGame, WHITE, BLACK, RED, GREEN, Direction
from utils import save_best_genome, save_best_score, load_best_score

# --- Pygame Visualization Constants ---
WIDTH = 800
HEIGHT = 600

# --- NEAT / Game Constants ---
BEST_SCORE = 0
GENERATIONS = 100  # Increase generations for better training
MAX_STEPS_WITHOUT_FOOD = 200  # Maximum steps allowed without eating food

# Global screen variable for drawing
screen = None
# Define fonts here, but initialize them after pygame.init()
info_font = None
mini_info_font = None


def eval_genomes(genomes, config):
    """
    Evaluate each genome by letting it control the snake.
    """
    global BEST_SCORE, screen, info_font, mini_info_font
    
    # Track the best genome in this generation
    best_genome = None
    best_fitness = 0
    
    # For each genome
    for genome_id, genome in genomes:
        # Reset fitness score
        genome.fitness = 0.0
        
        # Create neural network
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # Create a new snake game
        game = SnakeGame(20, 20)  # 20x20 grid
        
        # Variables to track game state
        steps_without_food = 0
        prev_distance = math.sqrt((game.food.x - game.head.x)**2 + (game.food.y - game.head.y)**2)
        
        # Game loop
        done = False
        while not done and steps_without_food < MAX_STEPS_WITHOUT_FOOD:
            # Handle events (close button, etc.)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            # Get state for neural network input
            # 1. Danger straight, left, right (3 inputs)
            # 2. Current direction (4 inputs)
            # 3. Food direction (4 inputs)
            state = get_state(game)
            
            # Get neural network output
            output = net.activate(state)
            
            # Determine direction based on output
            # [left, straight, right]
            direction = game.direction
            if output[0] > output[1] and output[0] > output[2]:
                # Turn left
                if direction == Direction.UP:
                    direction = Direction.LEFT
                elif direction == Direction.LEFT:
                    direction = Direction.DOWN
                elif direction == Direction.DOWN:
                    direction = Direction.RIGHT
                elif direction == Direction.RIGHT:
                    direction = Direction.UP
            elif output[2] > output[1] and output[2] > output[0]:
                # Turn right
                if direction == Direction.UP:
                    direction = Direction.RIGHT
                elif direction == Direction.RIGHT:
                    direction = Direction.DOWN
                elif direction == Direction.DOWN:
                    direction = Direction.LEFT
                elif direction == Direction.LEFT:
                    direction = Direction.UP
            
            # Move the snake
            game.direction = direction
            game_over, reward = move_snake(game)
            
            if game_over:
                done = True
            
            # Update fitness based on reward
            genome.fitness += reward
            
            # Reward for survival
            genome.fitness += 0.01
            
            # Check if snake is getting closer to food
            current_distance = math.sqrt((game.food.x - game.head.x)**2 + (game.food.y - game.head.y)**2)
            if current_distance < prev_distance:
                genome.fitness += 0.1  # Reward for getting closer to food
            else:
                genome.fitness -= 0.05  # Small penalty for getting further from food
            
            # Additional reward for eating food relative to snake length
            if reward > 0:  # Food was eaten
                genome.fitness += game.score * 0.5  # Reward increases with snake length
            
            prev_distance = current_distance
            
            # Count steps without food
            steps_without_food += 1
            if reward > 0:  # Food was eaten
                steps_without_food = 0
            
            # Draw game
            if screen is not None:
                screen.fill(BLACK)
                game.draw(screen)
                
                # Draw info
                info_text = f"Score: {game.score} | Fitness: {genome.fitness:.2f} | Steps w/o food: {steps_without_food}"
                text_surface = info_font.render(info_text, True, WHITE)
                screen.blit(text_surface, (10, 10))
                
                # Draw generation info
                gen_info = f"Genome ID: {genome_id} | Best Score: {BEST_SCORE}"
                gen_surface = mini_info_font.render(gen_info, True, WHITE)
                screen.blit(gen_surface, (10, HEIGHT - 20))
                
                pygame.display.update()
                pygame.time.delay(20)  # Slow down visualization
        
        # Update best score
        if game.score > BEST_SCORE:
            BEST_SCORE = game.score
            save_best_score(BEST_SCORE)
            save_best_genome(genome)
            
        # Track best genome in this generation
        if genome.fitness > best_fitness:
            best_fitness = genome.fitness
            best_genome = genome


def get_state(game):
    """
    Get the state of the game as input for the neural network.
    """
    # Get head position
    head_x, head_y = game.head.x, game.head.y
    
    # 1. Check for danger in three directions (straight, left, right)
    danger_straight = is_danger(game, game.direction)
    
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
    
    danger_left = is_danger(game, left_dir)
    
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
    
    danger_right = is_danger(game, right_dir)
    
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


def is_danger(game, direction):
    """
    Check if there's danger in the given direction (wall or body).
    """
    head_x, head_y = game.head.x, game.head.y
    
    # Calculate next position based on direction
    next_x, next_y = head_x, head_y
    if direction == Direction.UP:
        next_y -= 1
    elif direction == Direction.RIGHT:
        next_x += 1
    elif direction == Direction.DOWN:
        next_y += 1
    elif direction == Direction.LEFT:
        next_x -= 1
    
    # Check if out of bounds
    if next_x < 0 or next_x >= game.w or next_y < 0 or next_y >= game.h:
        return 1
    
    # Check if hitting itself
    for segment in game.snake[1:]:
        if segment.x == next_x and segment.y == next_y:
            return 1
    
    return 0


def move_snake(game):
    """
    Move the snake and return if game is over and reward.
    """
    # Determine next position based on direction
    head_x, head_y = game.head.x, game.head.y
    if game.direction == Direction.UP:
        head_y -= 1
    elif game.direction == Direction.RIGHT:
        head_x += 1
    elif game.direction == Direction.DOWN:
        head_y += 1
    elif game.direction == Direction.LEFT:
        head_x -= 1
    
    # Check if game over (hit wall or itself)
    if (head_x < 0 or head_x >= game.w or 
        head_y < 0 or head_y >= game.h):
        return True, -10  # Game over, negative reward
    
    for segment in game.snake:
        if segment.x == head_x and segment.y == head_y:
            return True, -10  # Game over, negative reward
    
    # Check if food is eaten
    reward = 0
    if game.food.x == head_x and game.food.y == head_y:
        reward = 10 + game.score * 0.5  # Reward increases with snake length
        game.score += 1
        game._place_food()
    else:
        game.snake.pop()  # Remove tail if no food eaten
    
    # Move head
    from collections import namedtuple
    Point = namedtuple('Point', 'x, y')
    game.head = Point(head_x, head_y)
    game.snake.insert(0, game.head)
    
    return False, reward


def run_neat(config_file):
    """
    Runs the NEAT algorithm using the provided configuration file.
    """
    global screen, BEST_SCORE, info_font, mini_info_font # Add fonts to global scope access

    # --- Pygame Setup ---
    pygame.init()
    # Initialize fonts AFTER pygame.init()
    info_font = pygame.font.Font(None, 22)
    mini_info_font = pygame.font.Font(None, 16)

    pygame.display.set_caption("NEAT Snake AI Training")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    BEST_SCORE = load_best_score() # Load best score at the beginning

    # --- NEAT Setup ---
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_file)
    
    # Create the population
    p = neat.Population(config)
    
    # Add reporters to show progress in the terminal
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run for a large number of generations
    try:
        # Run NEAT training
        winner = p.run(eval_genomes, GENERATIONS)  # Run for 50 generations
    except Exception as e:
         print(f"An error occurred during training: {e}")
         import traceback
         traceback.print_exc() # Print detailed traceback
    finally:
         pygame.quit()
         print(f"Training finished. Best score achieved during training: {BEST_SCORE}")

# The import statement at the top of train.py should look like:
# from snake_game import SnakeGame, WHITE, BLACK, RED