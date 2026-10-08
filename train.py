"""Run:  python train.py            (all 3 models, saves the deployed one)
        python train.py data/my.csv"""
import sys
from src.trainer import run_training

if __name__ == "__main__":
    run_training(sys.argv[1] if len(sys.argv) > 1 else None)
