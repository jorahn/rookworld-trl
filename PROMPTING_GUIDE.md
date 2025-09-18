# RookWorld-LM Prompting Guide

This document provides comprehensive guidance for prompting RookWorld-LM models for both Policy (P:) and Environment/Arbiter (A:) tasks.

## Model Overview

RookWorld-LM is a unified chess language model that combines:
- **Policy Task (P:)**: Generates chess moves with evaluations
- **Environment/Arbiter Task (A:)**: Simulates chess environment transitions

## Model Variants

1. **ROOK-LM**: Policy-only model
   - Uses raw FEN input (no prefix)
   - Output format: `P: [FEN] M: [moves] E: [evals] B: [best_move]`

2. **RookWorld-LM**: Unified model with both capabilities
   - Policy task: Uses `P:` prefix
   - Environment task: Uses `A:` prefix

## Policy Task (P:)

### Prompt Format
```
P: [FEN]
```
**Important**: Note the trailing space after the FEN!

### Example
```
Input:  P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
Output: M: d2d4 e2e4 g1f3 c2c4 b1c3 E: 0.29 0.32 0.24 0.24 0.21 B: e2e4
```

### Output Structure
- **M:**: Candidate moves in UCI notation (e.g., e2e4, g1f3)
- **E:**: Evaluation scores for each candidate move
- **B:**: Best move selection

### Generation Parameters
```python
# Recommended parameters for policy generation
temperature = 0.6
top_k = 5
do_sample = True
max_new_tokens = 100
```

### Validation Checklist
- ✅ All three fields (M:, E:, B:) must be present
- ✅ Number of moves should match number of evaluations
- ✅ Best move should be from the candidate list
- ✅ Moves must be in valid UCI format (4-5 characters)
- ✅ Moves must be legal in the given position

## Environment/Arbiter Task (A:)

### Prompt Format
```
A: [state]+[action]+[move_history]+
```

### Critical Requirements
1. **Move History Format**:
   - Maximum 10 moves (space-separated)
   - **MUST include the current move as the LAST entry**
   - Chronological order (oldest to newest)
   - Example: `e2e4 e7e5 g1f3 b8c6 f1c4` (where f1c4 is the current move)

2. **Field Delimiters**:
   - Use `+` to separate all fields
   - Prompt must end with `+`

### Example
```
Input:  A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+e2e4+
Output: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1+0.001+0+0
```

### Output Structure
Format: `[new_state]+[reward]+[terminated]+[truncated]+`

1. **new_state**: FEN after move is applied
2. **reward**: Float value (typically 0.001 for normal moves, 1.0/-1.0 for game end)
3. **terminated**: 0 or 1 (game ended by checkmate/stalemate)
4. **truncated**: 0 or 1 (game ended by move limit)

### Generation Parameters
```python
# Recommended parameters for environment generation
temperature = 0.6
top_k = 5
do_sample = True
max_new_tokens = 100  # Environment output is typically ~80 chars
```

### Common Implementation Patterns

#### Building Move History
```python
from collections import deque

# Maintain move history with max length 10
move_history = deque(maxlen=10)

# After each move:
move_history.append(current_move)

# When creating prompt:
history_with_current = list(move_history)[-9:] + [action]
history_str = " ".join(history_with_current)
prompt = f"A: {state}+{action}+{history_str}+"
```

## Known Issues and Accuracy

### Overall Performance
- **Environment Transition Accuracy**: 97.7% (on 1000 diverse positions)
- **Policy Format Compliance**: ~93-94%

### Common Errors

1. **Early Game Issues** (Moves 1-2):
   - FEN formatting errors (missing digit counts, extra pieces)
   - Most errors occur in opening positions
   - Example: Generates `5p` instead of `5p2`, or `PPPPPP1PP` (9 squares) instead of `PPPPPP1P` (8 squares)

2. **Specific Problematic Moves**:
   - `d2d4` from starting position often generates malformed FENs
   - `e2e4` from starting position has ~30-40% success rate with sampling
   - Other opening moves work better once move history exists

3. **Error Distribution**:
   - 82.6% of errors occur in moves 1-10 (opening)
   - 13% in moves 11-30 (middlegame)
   - 4.3% in moves 31+ (endgame)

### Mitigation Strategies

1. **For Opening Positions**:
   - Consider using lower temperature (0.3) for first 2 moves
   - Validate FEN output and retry on malformed results
   - Use greedy decoding for known problematic moves

2. **For General Use**:
   - Always validate FEN strings before use
   - Implement retry logic for malformed outputs
   - After move 10, accuracy improves significantly

## Implementation Examples

### Complete Self-Play Example

```python
from transformers import pipeline
import chess
from collections import deque

# Load model
pipe = pipeline(
    "text-generation",
    model="jrahn/RookWorld-LM-124M",
    device="cuda",
    torch_dtype=torch.bfloat16
)
pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id

def play_game():
    board = chess.Board()
    move_history = deque(maxlen=10)

    while not board.is_game_over():
        # Get policy move
        fen = board.fen()
        policy_prompt = f"P: {fen} "

        policy_result = pipe(
            policy_prompt,
            max_new_tokens=100,
            temperature=0.6,
            top_k=5,
            do_sample=True,
            return_full_text=False,
            pad_token_id=pipe.tokenizer.eos_token_id
        )

        # Parse best move
        output = policy_result[0]['generated_text']
        if "B:" in output:
            best_move = output.split("B:")[-1].strip().split()[0]
        else:
            break  # Invalid format

        # Validate move
        try:
            move = chess.Move.from_uci(best_move)
            if move not in board.legal_moves:
                break
        except:
            break

        # Get environment transition
        state_before = board.fen()
        history_with_current = list(move_history)[-9:] + [best_move]
        history_str = " ".join(history_with_current)
        env_prompt = f"A: {state_before}+{best_move}+{history_str}+"

        env_result = pipe(
            env_prompt,
            max_new_tokens=100,
            temperature=0.6,
            top_k=5,
            do_sample=True,
            return_full_text=False,
            pad_token_id=pipe.tokenizer.eos_token_id
        )

        # Parse environment response
        env_output = env_result[0]['generated_text']
        parts = env_output.split("+")

        if len(parts) >= 4:
            new_state = parts[0].strip()
            terminated = parts[2].strip() == "1"

            # Validate transition
            board.push(move)
            if board.fen() != new_state:
                print(f"Warning: FEN mismatch at move {len(move_history)+1}")

        # Update history
        move_history.append(best_move)

        if terminated:
            break

    return board
```

## Testing and Validation

### Recommended Test Suite

1. **Format Compliance Test**:
   - Verify M:, E:, B: fields present
   - Check field counts match
   - Validate UCI format

2. **Environment Accuracy Test**:
   - Generate 1000+ diverse positions
   - Compare predicted vs actual FEN transitions
   - Track error patterns by game phase

3. **Edge Cases**:
   - Starting position transitions
   - Castling moves
   - En passant captures
   - Pawn promotions
   - Positions with few legal moves

## Model Availability

- **HuggingFace**: `jrahn/RookWorld-LM-124M`
- **Weights**: bfloat16 precision
- **Size**: 124M parameters
- **Architecture**: GPT-2 based

## References

- Original paper: [ROOK: Automated Reasoning and Tool-use for Chess](https://laion.ai/notes/rook/)
- Training code: [RookWorld Repository](https://github.com/jrahn/rookworld)
- Evaluation scripts: `dev/eval/` directory in RookWorld repo