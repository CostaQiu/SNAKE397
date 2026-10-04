# utils.py
import pickle
import os

BEST_SCORE_FILE = "best_score.txt"
BEST_GENOME_FILE = "best_genome.pkl"

def save_best_genome(genome, filename=BEST_GENOME_FILE):
    """Saves the best genome using pickle."""
    with open(filename, "wb") as f:
        pickle.dump(genome, f)
    print(f"Best genome saved to {filename}")

def load_best_genome(filename=BEST_GENOME_FILE):
    """Loads the best genome using pickle."""
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            try:
                genome = pickle.load(f)
                print(f"Best genome loaded from {filename}")
                return genome
            except Exception as e:
                print(f"Error loading genome from {filename}: {e}")
                return None
    else:
        print(f"Best genome file not found: {filename}")
        return None

def save_best_score(score, filename=BEST_SCORE_FILE):
    """Saves the best score to a text file."""
    try:
        with open(filename, "w") as f:
            f.write(str(score))
        # print(f"Best score {score} saved to {filename}")
    except IOError as e:
        print(f"Error saving best score to {filename}: {e}")

def load_best_score(filename=BEST_SCORE_FILE):
    """Loads the best score from a text file."""
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                score = int(f.read())
                # print(f"Best score {score} loaded from {filename}")
                return score
        except (IOError, ValueError) as e:
            print(f"Error loading or parsing best score from {filename}: {e}")
            return 0
    else:
        return 0