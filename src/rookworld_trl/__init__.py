"""
RookWorld GRPO Training with HuggingFace Transformers and TRL

A complete implementation of Group Relative Policy Optimization (GRPO) for chess
using the RookWorld dataset and sophisticated chess-accurate reward functions.
"""

__version__ = "0.1.0"

from .train import main as train_main
from .rewards import ChessRewardScorer, create_reward_function, find_stockfish_path
from .dataset import RookWorldDataGenerator

__all__ = [
    "train_main",
    "ChessRewardScorer", 
    "create_reward_function",
    "find_stockfish_path",
    "RookWorldDataGenerator",
]