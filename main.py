import os
import sys

# Add the outdated directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'outdated'))

def train(method="neat"):
    """Run the training process."""
    if method == "neat":
        from outdated.train import run_neat
        config_path = os.path.join(os.path.dirname(__file__), 'outdated', 'config-feedforward.txt')
        run_neat(config_path)
    elif method == "dqn":
        from dqn_train import train as dqn_train
        dqn_train()
    elif method == "advanced_dqn":
        from advanced_dqn import train as advanced_dqn_train
        advanced_dqn_train()
    elif method == "cnn_dqn":
        from cnn_dqn import train as cnn_dqn_train
        cnn_dqn_train()
    elif method == "improved_dqn":
        from improved_dqn import train as improved_dqn_train
        improved_dqn_train()

def demonstrate(method="neat"):
    """Run the demonstration of the trained AI."""
    if method == "neat":
        from outdated.demonstrate import run_demonstration
        config_path = os.path.join(os.path.dirname(__file__), 'outdated', 'config-feedforward.txt')
        run_demonstration(config_path)
    elif method == "dqn":
        from dqn_demonstrate import demonstrate as dqn_demonstrate
        dqn_demonstrate()
    elif method == "advanced_dqn":
        from advanced_dqn_demo import demonstrate as advanced_dqn_demonstrate
        advanced_dqn_demonstrate()
    elif method == "cnn_dqn":
        from cnn_dqn_demo import demonstrate as cnn_dqn_demonstrate
        cnn_dqn_demonstrate()
    elif method == "improved_dqn":
        from improved_dqn_demo import demonstrate as improved_dqn_demonstrate
        improved_dqn_demonstrate()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train or demonstrate the Snake AI")
    parser.add_argument("mode", choices=["train", "demo"], 
                        help="Mode to run: 'train' for training, 'demo' for demonstration")
    parser.add_argument("--method", choices=["neat", "dqn", "advanced_dqn", "cnn_dqn", "improved_dqn"], default="neat",
                        help="Method to use: 'neat' for NeuroEvolution, 'dqn' for Deep Q-Network, 'advanced_dqn' for Advanced DQN, 'cnn_dqn' for CNN-based DQN, 'improved_dqn' for Improved DQN")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train(args.method)
    elif args.mode == "demo":
        demonstrate(args.method)