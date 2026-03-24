from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.generator import save_dataset

if __name__ == "__main__":
    path = save_dataset("data/sample_candidates.json", n_fraud=500, n_genuine=500)
    print(f"Saved dataset to {path}")
