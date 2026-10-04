> Older README, kept for the earlier NEAT / DQN experiments (the outdated folder and the *_dqn*.py scripts).

# Snake AI with NEAT and Advanced DQN

This project uses five different approaches to train an AI to play the classic Snake game:
1. NeuroEvolution of Augmenting Topologies (NEAT)
2. Deep Q-Network (DQN) - A reinforcement learning approach
3. Advanced DQN - A highly sophisticated reinforcement learning approach with multiple advanced techniques
4. CNN DQN - A convolutional neural network approach that can better understand spatial relationships
5. Improved DQN - A specifically tuned DQN with better reward shaping and architecture

## Files

- `outdated/snake_game.py`: Contains the SnakeGame class with game logic
- `outdated/train.py`: Implements the NEAT training process
- `outdated/demonstrate.py`: Demonstrates the trained NEAT AI
- `outdated/config-feedforward.txt`: NEAT configuration file
- `outdated/utils.py`: Utility functions for saving/loading genomes and scores
- `dqn_train.py`: Implements the basic DQN training process
- `dqn_demonstrate.py`: Demonstrates the trained basic DQN AI
- `advanced_dqn.py`: Implements the advanced DQN training process with multiple enhancements
- `advanced_dqn_demo.py`: Demonstrates the trained advanced DQN AI
- `cnn_dqn.py`: Implements the CNN-based DQN training process
- `cnn_dqn_demo.py`: Demonstrates the trained CNN DQN AI
- `improved_dqn.py`: Implements the improved DQN training process with better reward shaping
- `improved_dqn_demo.py`: Demonstrates the trained improved DQN AI
- `main.py`: Main entry point for training or demonstration

## Improvements Made

### NEAT Approach:
1. Enhanced state representation for the neural network (11 inputs instead of 8)
2. Improved reward system to encourage better performance
3. Increased maximum steps without food to allow for longer survival strategies
4. Increased population size and generations for better training
5. Improved neural network configuration for better learning

### Basic DQN Approach:
1. Uses a Deep Q-Network with experience replay for more stable learning
2. Implements epsilon-greedy exploration strategy that decays over time
3. Uses a target network to stabilize training
4. More efficient learning compared to NEAT for this specific problem

### Advanced DQN Approach:
1. **Noisy Networks**: Replaces epsilon-greedy with parameter noise for better exploration
2. **Prioritized Experience Replay**: Focuses learning on important experiences
3. **Multi-step Learning**: Considers rewards over multiple steps for better credit assignment
4. **Enhanced State Representation**: Adds distance to food and snake length features
5. **Larger Network**: Uses 512 hidden units for better representation capacity
6. **Higher Discount Factor**: Uses γ=0.95 for better long-term planning

### CNN DQN Approach:
1. **Spatial Understanding**: Uses a convolutional neural network to process the game grid directly
2. **Better Feature Representation**: Represents the game state as a 3-channel image (snake head, snake body, food)
3. **Improved Pattern Recognition**: CNN can recognize spatial patterns and relationships
4. **Eliminates Manual Feature Engineering**: No need to manually design state features

### Improved DQN Approach (Recommended for quick results):
1. **Better Reward Shaping**: 
   - Reward for getting closer to food
   - Small reward for survival
   - Penalty for time to encourage efficiency
   - Stronger penalty for self-collision
2. **Enhanced State Representation**: 15 features including Manhattan distance
3. **Deeper Network**: 4 hidden layers for better representation
4. **Fine-tuned Hyperparameters**: Optimized learning rate, batch size, and decay rates

## How to Run

1. Install required packages:
   ```
   pip install pygame neat-python torch numpy
   ```

2. Train the AI:
   ```
   # Using NEAT (original approach)
   python main.py train --method neat
   
   # Using basic DQN
   python main.py train --method dqn
   
   # Using advanced DQN
   python main.py train --method advanced_dqn
   
   # Using CNN DQN
   python main.py train --method cnn_dqn
   
   # Using Improved DQN (recommended for quick results)
   python main.py train --method improved_dqn
   ```

3. Demonstrate the trained AI:
   ```
   # Using NEAT (original approach)
   python main.py demo --method neat
   
   # Using basic DQN
   python main.py demo --method dqn
   
   # Using advanced DQN
   python main.py demo --method advanced_dqn
   
   # Using CNN DQN
   python main.py demo --method cnn_dqn
   
   # Using Improved DQN (recommended for quick results)
   python main.py demo --method improved_dqn
   ```

## Controls for Demonstration

- Press 'Q' or close the window to quit the demonstration

## Expected Performance

The Improved DQN approach should provide good results quickly:
- Basic DQN: Scores around 10-50
- NEAT: Scores around 20-100
- Advanced DQN: Scores of 100-200+
- CNN DQN: Scores of 200+ with sufficient training
- Improved DQN: Scores of 100+ relatively quickly, with potential for 200+ with more training

The Improved DQN addresses the core issues that were causing poor performance:
1. **Better Reward Shaping**: More nuanced rewards guide the snake toward better behavior
2. **Enhanced State Representation**: More informative features help the snake make better decisions
3. **Fine-tuned Hyperparameters**: Optimized settings for faster learning
## RL Snake with search + 3D cartoon viewer

Files (newer than the sections above):

- `snake_env.py`: fast headless environment with 27 space-aware features (flood-fill area, tail/food reachability)
- `rl_train.py`: Double + Dueling DQN trainer with a "tail must stay reachable" safety mask (`python rl_train.py`)
- `rl_search.py`: food-directed lookahead on top of the trained network (`python rl_search.py --goal`)
- `rl_demo.py`: pygame demo of the network + search (`python rl_demo.py`)
- `record_games.py`: records AI games into `replays.js`
- `snake3d.html`: 3D cartoon replay viewer with synthesized sound effects and music

Results (32 games, 20x20 board, max 397): network + safety mask averages ~289; with food-directed
search ~345-353 (median ~390), and occasionally a perfect 397. The remaining failures are free cells
sealed off by the snake's own body earlier in the game; only a Hamiltonian-cycle strategy guarantees 397.

View the 3D replays:

```
python -m http.server 8765      # or just open snake3d.html from disk
# then open http://127.0.0.1:8765/snake3d.html
```

Screensaver mode: double-click `screensaver.html` (opens in your default browser; press F11 for full screen)
or `start_screensaver.bat` (full-screen Edge, close with Alt+F4), or open `snake3d.html#saver` directly. All UI is hidden, no click is needed, the cursor is
hidden, and it loops through 10 perfect 397-point games (`replays_wins.js`, recorded with
`python record_games.py --wins 10 --seed0 20000 --out replays_wins.js`), switching 8 seconds after each
game ends. It is silent by design (browsers only allow sound after a click). To use it as a real Windows
screensaver it still needs a wrapper (a `.scr` launcher or a wallpaper app that can show an HTML page).

Real Windows screensaver (`screensaver/SnakeSaver.scr`, built with `python build_screensaver.py`, uses the
C# compiler that ships with Windows). It shows `snake_classic.html`: black background, Nokia-style grey snake body,
a green triangle head pointing the way it moves, a red triangle tail pointing away from the body, white numbers
(score, steps, game time, clock; English text), plain 2D canvas redrawn only when something changes. The snake moves
one cell per step at a constant 2x speed (16 steps/s), no smoothing.
One full-screen Edge window per connected monitor; all screens show the same perfect game in lockstep (the frame is
derived from a shared clock passed in the URL). Only the primary screen makes sound (square-wave blips). Any
mouse/keyboard input quits it. Progress is saved in `%LOCALAPPDATA%\SnakeSaver\progress.txt` and resumed next time;
`launcher.log` and `heartbeat.txt` next to it are diagnostics. The 3D version (`snake3d.html`) is kept for viewing in
the browser. The project path is compiled in, so rebuild after moving the folder.
Measured on the real Windows-triggered run (user away): GPU acceleration ON gave ~6-12% GPU and a few % of one CPU core;
`--disable-gpu` made it worse (~19% GPU and ~100% of one CPU core), so acceleration stays on.
Test without waiting for the idle timer: `SnakeSaver.scr /s /test 15` (in Git Bash set `MSYS_NO_PATHCONV=1`,
otherwise `/s` is rewritten to a path). Test hooks (environment variables): `SNAKESAVER_EXTRA` (extra URL
parameters), `SNAKESAVER_EDGEFLAGS` (extra browser flags), `SNAKESAVER_NOTOP=1`.

Live mode (the AI computes every move on the spot while the page plays it):

```
python live_server.py
# then open http://127.0.0.1:8770/snake3d.html#live  (or click the "Live compute" button)
```

Regenerate the replays with `python record_games.py`. Tests: `python -m pytest`.
