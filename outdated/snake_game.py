# snake_game.py
import pygame # Keep this import
import random
from enum import Enum
from collections import namedtuple

# REMOVE pygame.init() from here
# REMOVE font = pygame.font.Font(None, 25) from here

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# Colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

class SnakeGame:
    def __init__(self, w=20, h=20, block_size=20):
        self.w = w
        self.h = h
        self.block_size = block_size
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                     Point(self.head.x-1, self.head.y),
                     Point(self.head.x-2, self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()

    def _place_food(self):
        x = random.randint(0, self.w-1)
        y = random.randint(0, self.h-1)
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def draw(self, surface, draw_grid=True):
        """Draws the game onto the given Pygame surface."""
        surface.fill(BLACK)
        block_w = surface.get_width() // self.w
        block_h = surface.get_height() // self.h

        if draw_grid:
            for x in range(0, surface.get_width(), block_w):
                pygame.draw.line(surface, WHITE, (x, 0), (x, surface.get_height()))
            for y in range(0, surface.get_height(), block_h):
                pygame.draw.line(surface, WHITE, (0, y), (surface.get_width(), y))

        # Draw snake
        for i, pt in enumerate(self.snake):
            color = BLUE1 if i != 0 else GREEN # Green head
            pygame.draw.rect(surface, color, pygame.Rect(pt.x * block_w, pt.y * block_h, block_w, block_h))

        # Draw food
        pygame.draw.rect(surface, RED, pygame.Rect(self.food.x * block_w, self.food.y * block_h, block_w, block_h))

    def get_state(self):
        """
        Get the state of the game as input for the neural network.
        """
        # Get head position
        head_x, head_y = self.head.x, self.head.y
        
        # 1. Check for danger in three directions (straight, left, right)
        danger_straight = self._is_danger(self.direction)
        
        # Calculate left direction
        left_dir = self.direction
        if left_dir == Direction.UP:
            left_dir = Direction.LEFT
        elif left_dir == Direction.LEFT:
            left_dir = Direction.DOWN
        elif left_dir == Direction.DOWN:
            left_dir = Direction.RIGHT
        elif left_dir == Direction.RIGHT:
            left_dir = Direction.UP
        
        danger_left = self._is_danger(left_dir)
        
        # Calculate right direction
        right_dir = self.direction
        if right_dir == Direction.UP:
            right_dir = Direction.RIGHT
        elif right_dir == Direction.RIGHT:
            right_dir = Direction.DOWN
        elif right_dir == Direction.DOWN:
            right_dir = Direction.LEFT
        elif right_dir == Direction.LEFT:
            right_dir = Direction.UP
        
        danger_right = self._is_danger(right_dir)
        
        # 2. Current direction
        dir_up = 1 if self.direction == Direction.UP else 0
        dir_right = 1 if self.direction == Direction.RIGHT else 0
        dir_down = 1 if self.direction == Direction.DOWN else 0
        dir_left = 1 if self.direction == Direction.LEFT else 0
        
        # 3. Food direction (relative to snake head)
        food_left = 1 if self.food.x < head_x else 0
        food_right = 1 if self.food.x > head_x else 0
        food_up = 1 if self.food.y < head_y else 0
        food_down = 1 if self.food.y > head_y else 0
        
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
        
    def _is_danger(self, direction):
        """
        Check if there's danger in the given direction (wall or body).
        """
        head_x, head_y = self.head.x, self.head.y
        
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
        if next_x < 0 or next_x >= self.w or next_y < 0 or next_y >= self.h:
            return 1
        
        # Check if hitting itself
        for segment in self.snake[1:]:
            if segment.x == next_x and segment.y == next_y:
                return 1
        
        return 0
        
    def play_step(self, action):
        """
        Play one step of the game based on the action.
        Action is an array: [left, straight, right]
        """
        # Determine direction based on action
        direction = self.direction
        if action[0] == 1:  # Turn left
            if direction == Direction.UP:
                direction = Direction.LEFT
            elif direction == Direction.LEFT:
                direction = Direction.DOWN
            elif direction == Direction.DOWN:
                direction = Direction.RIGHT
            elif direction == Direction.RIGHT:
                direction = Direction.UP
        elif action[2] == 1:  # Turn right
            if direction == Direction.UP:
                direction = Direction.RIGHT
            elif direction == Direction.RIGHT:
                direction = Direction.DOWN
            elif direction == Direction.DOWN:
                direction = Direction.LEFT
            elif direction == Direction.LEFT:
                direction = Direction.UP
        
        # Move the snake
        self.direction = direction
        game_over, reward = self._move_snake()
        
        return reward, game_over, self.score
        
    def _move_snake(self):
        """
        Move the snake and return if game is over and reward.
        """
        # Determine next position based on direction
        head_x, head_y = self.head.x, self.head.y
        if self.direction == Direction.UP:
            head_y -= 1
        elif self.direction == Direction.RIGHT:
            head_x += 1
        elif self.direction == Direction.DOWN:
            head_y += 1
        elif self.direction == Direction.LEFT:
            head_x -= 1
        
        # Check if game over (hit wall or itself)
        if (head_x < 0 or head_x >= self.w or 
            head_y < 0 or head_y >= self.h):
            return True, -10  # Game over, negative reward
        
        for segment in self.snake:
            if segment.x == head_x and segment.y == head_y:
                return True, -10  # Game over, negative reward
        
        # Check if food is eaten
        reward = 0
        if self.food.x == head_x and self.food.y == head_y:
            reward = 10 + self.score * 0.5  # Reward increases with snake length
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()  # Remove tail if no food eaten
        
        # Move head
        from collections import namedtuple
        Point = namedtuple('Point', 'x, y')
        self.head = Point(head_x, head_y)
        self.snake.insert(0, self.head)
        
        return False, reward 