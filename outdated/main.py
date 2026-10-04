# main.py
import sys
import os
from train import run_neat
from demonstrate import run_demonstration

CONFIG_PATH = "config-feedforward.txt"

def print_usage():
    print("Usage: python main.py <mode>")
    print("Modes:")
    print("  train       - Start training the NEAT model")
    print("  demo        - Demonstrate the best trained model")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print_usage()
        sys.exit(1)

    mode = sys.argv[1].lower()

    if not os.path.exists(CONFIG_PATH):
         print(f"Error: NEAT configuration file not found at {CONFIG_PATH}")
         sys.exit(1)

    if mode == "train":
        print("Starting AI training...")
        # Create checkpoints directory if it doesn't exist
        if not os.path.exists("checkpoints"):
            try:
                os.makedirs("checkpoints")
                print("Created 'checkpoints' directory.")
            except OSError as e:
                print(f"Error creating 'checkpoints' directory: {e}")
                sys.exit(1)
        run_neat(CONFIG_PATH)
    elif mode == "demo":
        print("Starting demonstration of the best model...")
        run_demonstration(CONFIG_PATH)
    else:
        print(f"Error: Unknown mode '{mode}'")
        print_usage()
        sys.exit(1)