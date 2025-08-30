"""
Dataset loading and preprocessing for RookWorld GRPO training.

Self-contained implementation that handles both P: (Policy) and A: (Environment) tasks
from the RookWorld dataset without external dependencies.
"""

import re
import logging
from typing import List, Tuple, Dict, Optional
from datasets import load_dataset
import random

logger = logging.getLogger(__name__)


def preprocess_sample(text: str) -> str:
    """
    Preprocess a raw dataset sample.
    
    CRITICAL: If text doesn't start with "P: ", it's an A: task and needs "A: " prefix.
    
    Args:
        text: Raw text from dataset
        
    Returns:
        Preprocessed text with proper task prefix
    """
    text = text.strip()
    
    # If it already has a proper prefix, keep it
    if text.startswith("P: ") or text.startswith("A: "):
        return text
    
    # Otherwise, it's an A: task that needs the prefix
    logger.debug(f"Adding 'A: ' prefix to sample")
    return "A: " + text


def parse_p_task(text: str) -> Tuple[str, str, Dict]:
    """
    Parse a P: (Policy) task into prompt and completion.
    
    Format:
    - Input: "P: [FEN] M: [moves] E: [evals] B: [best]"
    - Prompt: "P: [FEN]"
    - Completion: "M: [moves] E: [evals] B: [best]"
    
    Args:
        text: Preprocessed P: task text
        
    Returns:
        (prompt, completion, parsed_data) where parsed_data contains:
        - fen: The chess position
        - moves: List of top moves (if present)
        - evals: List of evaluations (if present)
        - best_move: Best move (if present)
    """
    if not text.startswith("P: "):
        raise ValueError(f"Not a P: task: {text[:50]}...")
    
    parsed_data = {}
    
    # Find where completion starts (at M:)
    if "M:" in text:
        # Split into prompt and completion
        parts = text.split("M:", 1)
        prompt = parts[0].strip()  # "P: [FEN]"
        completion = "M:" + parts[1].strip()  # "M: ... E: ... B: ..."
        
        # Extract FEN from prompt
        fen = prompt[3:].strip()  # Remove "P: " prefix
        parsed_data['fen'] = fen
        
        # Parse completion components
        # Extract moves (M: section)
        moves_match = re.search(r'M:\s*([a-h][1-8][a-h][1-8]\w*(?:\s+[a-h][1-8][a-h][1-8]\w*)*)', completion)
        if moves_match:
            moves = moves_match.group(1).split()
            parsed_data['moves'] = moves
        
        # Extract evaluations (E: section)
        evals_match = re.search(r'E:\s*([-\d\.]+(?:\s+[-\d\.]+)*)', completion)
        if evals_match:
            try:
                evals = [float(x) for x in evals_match.group(1).split()]
                parsed_data['evals'] = evals
            except ValueError:
                parsed_data['evals'] = []
        
        # Extract best move (B: section)
        best_match = re.search(r'B:\s*([a-h][1-8][a-h][1-8]\w*)', completion)
        if best_match:
            parsed_data['best_move'] = best_match.group(1)
    else:
        # No completion, just prompt
        prompt = text.strip()
        completion = ""
        
        # Extract FEN
        fen = prompt[3:].strip() if prompt.startswith("P: ") else ""
        parsed_data['fen'] = fen
    
    logger.debug(f"Parsed P: task - prompt: {prompt[:50]}..., completion: {completion[:50] if completion else 'None'}")
    
    return prompt, completion, parsed_data


def parse_a_task(text: str) -> Tuple[str, str, Dict]:
    """
    Parse an A: (Environment) task into prompt and completion.
    
    Format:
    - Input: "A: [FEN]+[move]+[history]+[new_FEN]+[reward]+[terminated]+[truncated]"
    - Prompt: "A: [FEN]+[move]+[history]+"
    - Completion: "[new_FEN]+[reward]+[terminated]+[truncated]"
    
    Args:
        text: Preprocessed A: task text
        
    Returns:
        (prompt, completion, parsed_data) where parsed_data contains:
        - fen: Starting position
        - move: Move to make
        - history: Move history (comma-separated)
        - new_fen: Resulting position (if present)
        - reward: Reward value (if present)
        - terminated: Game ended flag (if present)
        - truncated: Illegal move flag (if present)
    """
    if not text.startswith("A: "):
        raise ValueError(f"Not an A: task: {text[:50]}...")
    
    parsed_data = {}
    
    # Remove "A: " prefix for parsing
    content = text[3:].strip()
    
    # Split by + delimiter
    components = content.split("+")
    
    if len(components) >= 7:
        # Full format with all components
        # Prompt: FEN + move + history +
        fen = components[0].strip()
        move = components[1].strip()
        history = components[2].strip()
        
        # Completion: new_FEN + reward + terminated + truncated
        new_fen = components[3].strip()
        reward = components[4].strip()
        terminated = components[5].strip()
        truncated = components[6].strip()
        
        prompt = f"A: {fen}+{move}+{history}+"
        completion = f"{new_fen}+{reward}+{terminated}+{truncated}"
        
        # Store parsed data
        parsed_data['fen'] = fen
        parsed_data['move'] = move
        parsed_data['history'] = history
        parsed_data['new_fen'] = new_fen
        parsed_data['reward'] = float(reward) if reward else 0.0
        parsed_data['terminated'] = terminated.lower() in ['true', '1']
        parsed_data['truncated'] = truncated.lower() in ['true', '1']
        
    elif len(components) >= 3:
        # Partial format (prompt only)
        fen = components[0].strip()
        move = components[1].strip()
        history = components[2].strip() if len(components) > 2 else ""
        
        prompt = f"A: {fen}+{move}+{history}+"
        completion = ""
        
        parsed_data['fen'] = fen
        parsed_data['move'] = move
        parsed_data['history'] = history
        
    else:
        # Malformed, but try to handle gracefully
        logger.warning(f"Malformed A: task with {len(components)} components")
        prompt = text
        completion = ""
        
        if components:
            parsed_data['fen'] = components[0].strip()
        if len(components) > 1:
            parsed_data['move'] = components[1].strip()
    
    logger.debug(f"Parsed A: task - prompt: {prompt[:50]}..., completion: {completion[:50] if completion else 'None'}")
    
    return prompt, completion, parsed_data


def load_and_prepare_samples(
    n_samples: int = 100,
    dataset_name: str = "jrahn/rookworld_7m",
    split: str = "train",
    seed: int = 42
) -> List[Tuple[str, str, str, Dict]]:
    """
    Load and prepare samples from the dataset.
    
    Args:
        n_samples: Number of samples to load
        dataset_name: HuggingFace dataset name
        split: Dataset split to use
        seed: Random seed for sampling
        
    Returns:
        List of (task_type, prompt, completion, parsed_data) tuples
    """
    logger.info(f"Loading {n_samples} samples from {dataset_name}")
    
    # Load dataset
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    dataset = dataset.shuffle(seed=seed)
    
    prepared_samples = []
    p_count = 0
    a_count = 0
    
    for i, sample in enumerate(dataset.take(n_samples)):
        text = sample.get('text', str(sample))
        
        # Step 1: Preprocess (add A: prefix if needed)
        text = preprocess_sample(text)
        
        # Step 2: Parse based on task type
        try:
            if text.startswith("P: "):
                task_type = "P"
                prompt, completion, parsed_data = parse_p_task(text)
                p_count += 1
            elif text.startswith("A: "):
                task_type = "A"
                prompt, completion, parsed_data = parse_a_task(text)
                a_count += 1
            else:
                logger.warning(f"Unknown task type: {text[:50]}...")
                continue
            
            prepared_samples.append((task_type, prompt, completion, parsed_data))
            
        except Exception as e:
            logger.error(f"Error parsing sample {i}: {e}")
            continue
    
    logger.info(f"Prepared {len(prepared_samples)} samples - P: {p_count}, A: {a_count}")
    
    return prepared_samples


def get_batch_by_type(
    samples: List[Tuple[str, str, str, Dict]],
    task_type: str,
    batch_size: int
) -> List[Tuple[str, str, str, Dict]]:
    """
    Get a batch of samples of a specific task type.
    
    This avoids mixing P: and A: tasks which have very different lengths.
    
    Args:
        samples: List of prepared samples
        task_type: "P" or "A"
        batch_size: Number of samples to return
        
    Returns:
        List of samples of the specified type
    """
    typed_samples = [s for s in samples if s[0] == task_type]
    
    if len(typed_samples) < batch_size:
        logger.warning(f"Only {len(typed_samples)} {task_type} samples available, requested {batch_size}")
    
    return typed_samples[:batch_size]


class RookWorldDataGenerator:
    """Generate chess training data from RookWorld dataset."""
    
    def __init__(self, max_length: int = 256, dataset_size: int = 1000):
        self.max_length = max_length
        self.dataset_size = dataset_size
        self.samples = self._load_rookworld_samples()
    
    def _load_rookworld_samples(self) -> List[Tuple[str, str, str, Dict]]:
        """Load samples from RookWorld dataset."""
        print(f"Loading {self.dataset_size} samples from RookWorld dataset...")
        try:
            samples = load_and_prepare_samples(
                n_samples=self.dataset_size,
                dataset_name="jrahn/rookworld_7m",
                split="train",
                seed=42
            )
            print(f"✓ Loaded {len(samples)} mixed task samples")
            return samples
        except Exception as e:
            print(f"⚠️ Failed to load RookWorld dataset: {e}")
            print("Falling back to synthetic data...")
            return self._create_fallback_samples()
    
    def _create_fallback_samples(self) -> List[Tuple[str, str, str, Dict]]:
        """Create fallback synthetic samples if dataset loading fails."""
        fallback_samples = []
        positions = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 4",
            "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 b kq - 0 5",
        ]
        
        for i in range(min(50, self.dataset_size)):
            fen = random.choice(positions)
            if i % 2 == 0:  # P: task
                prompt = f"P: {fen}"
                completion = " M: e2e4 d2d4 g1f3 E: 0.3 0.35 0.28 B: e2e4"
                fallback_samples.append(("P", prompt, completion, {"fen": fen}))
            else:  # A: task
                prompt = f"A: {fen}+e2e4+,"
                completion = "+rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1+0.001+false+false"
                fallback_samples.append(("A", prompt, completion, {"fen": fen}))
        
        return fallback_samples
    
    def get_mixed_batch(self, batch_size: int) -> List[str]:
        """Get a mixed batch of prompts from both P: and A: tasks."""
        if not self.samples:
            return []
        
        # Sample random subset
        selected_samples = random.sample(
            self.samples, 
            min(batch_size, len(self.samples))
        )
        
        prompts = []
        for task_type, prompt, completion, parsed_data in selected_samples:
            prompts.append(prompt)
        
        return prompts
    
    def get_task_specific_batch(self, task_type: str, batch_size: int) -> List[str]:
        """Get a batch of specific task type (P or A)."""
        task_samples = get_batch_by_type(self.samples, task_type, batch_size)
        return [prompt for _, prompt, _, _ in task_samples]
    
    def get_samples_info(self) -> Dict[str, int]:
        """Get information about loaded samples."""
        if not self.samples:
            return {"total": 0, "P": 0, "A": 0}
        
        p_count = sum(1 for s in self.samples if s[0] == "P")
        a_count = sum(1 for s in self.samples if s[0] == "A")
        
        return {
            "total": len(self.samples),
            "P": p_count,
            "A": a_count
        }