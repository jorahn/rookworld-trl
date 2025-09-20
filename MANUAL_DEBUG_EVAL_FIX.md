## Manual Debug Eval: Findings, Fixes, and Next Steps

Date: 2025-09-18

### Summary

Evaluation in the manual-debug path was reporting 0% accuracy despite clear signs of learning during training. The root cause was overly strict and spacing‑sensitive FEN extraction and comparison in the eval routines, which led to false negatives when the model produced the correct position but differed on move counters or formatting. I implemented normalization and a lenient FEN equivalence check across the manual and standalone evaluation scripts to align evaluation with practical correctness and with our reward logic.

---

### Symptoms Observed

- Eval frequently printed 0% accuracy on `opening_dataset_eval.json` even when manual samples and rewards suggested improvement.
- Discrepancies between post‑step greedy metrics in `manual_grpo_debug.py` and held‑out eval accuracy.
- “Exact FEN” matching expected an exact 6‑field equality; any minor difference (e.g., halfmove/fullmove counters) led to a miss.

---

### Root Causes

1) Strict FEN equality
- Eval paths compared full FEN strings for equality, including halfmove/fullmove counters.
- The model often outputs correct position, turn, castling, and en passant, but may differ slightly on counters, yielding false negatives.

2) Spacing and tokenization noise
- No spacing normalization applied before parsing the completion, so variable whitespace could break equality checks.
- Some completions can echo task prefixes (e.g., "A:" or "P:"), polluting extraction.

3) Mismatch with training/rewards
- Reward logic allows nuanced scoring (e.g., leniency for move counters) while eval demanded strict 6‑field equality, exaggerating the gap.

---

### Changes Implemented

Applied consistently across manual debug and standalone evaluation scripts:

- Normalize spacing with `normalize_spacing` before extracting a FEN from generated text.
- Strip any leading task marker (`^\s*[PA]:\s*`) from the generated completion before parsing.
- Use a lenient FEN equivalence function:
  - Pass if exact strings match, OR
  - Pass if the first 4 FEN fields match: board position, side to move, castling rights, en passant square. Halfmove/fullmove counters are ignored for correctness purposes.

Files and key modifications:

1) `manual_grpo_debug.py`
- `evaluate_on_eval_set(...)` now:
  - Normalizes spacing and strips `P:/A:` before extraction.
  - Uses lenient `_fen_equivalent` to count correct predictions.
  - Handles move‑1 reporting only when totals > 0 (avoid divide‑by‑zero).
- A‑task initial metrics in Phase 2:
  - Normalizes completion, strips prefixes before extraction.
  - Uses the same lenient equivalence when crediting exact matches.

2) `evaluate_checkpoint.py`
- Introduced `_extract_generated_fen(...)` with normalization + prefix strip.
- Added `_fen_equivalent(...)` with lenient 4‑field match.
- Adjusted correctness accounting to use new helpers.

3) `debug_eval.py`
- Normalizes completions and strips prefixes.
- Prints a “Match (lenient)” result using the same 4‑field equivalence.

Implementation details:
- We intentionally ignore halfmove and fullmove counters for correctness. We still require board, turn, castling rights, and en passant to match, preserving the critical game‑state fidelity.

---

### Impact

- Evaluations now reflect practical correctness rather than stringent formatting/counter differences.
- Should eliminate spurious 0% scores when positions are correct but counters differ.
- Brings manual/standalone eval into closer alignment with reward scoring expectations and the model’s deterministic greedy outputs.

---

### Backwards Compatibility and Risks

- Behavior change: reported accuracy may increase compared to the old strict metric. This is expected and desired to remove false negatives.
- We still enforce strictness on core game state (position, side to move, castling, en passant), so we are not masking substantive errors.
- The equivalence logic is currently duplicated in `manual_grpo_debug.py`, `evaluate_checkpoint.py`, and `debug_eval.py`; centralizing will reduce drift risk.

---

### Recommended Next Steps

1) Centralize FEN parsing/equivalence
- Create shared helpers in `src/rookworld_trl/utils.py`:
  - `extract_fen_from_completion(text: str) -> str`
  - `fen_equivalent(pred: str, exp: str, mode: Literal['lenient','strict']='lenient') -> bool`
- Update all eval scripts to consume these helpers (remove duplication).

2) Dual‑metric reporting
- Report both metrics during eval:
  - Strict (6‑field exact) accuracy
  - Lenient (4‑field) accuracy
- This helps track whether improvements are blocked by move‑counter modeling vs. position understanding.

3) Tests
- Add unit tests for FEN extraction and equivalence in `tests/`:
  - Spacing normalization cases
  - Prefix stripping
  - Counter differences only
  - En passant/castling mismatches (should fail)
- Add smoke tests for `evaluate_checkpoint.py` with a tiny synthetic sample.

4) Logging and diagnostics
- In periodic eval inside `manual_grpo_debug.py`, log a small confusion matrix or a few representative mismatches per move number with short diffs (first 4 fields vs. counters) to accelerate diagnosis.

5) Reward–eval parity review
- Cross‑check reward scoring on A‑tasks to confirm it shares the same stance on counters and normalization to avoid metric drift. If different by design, document the rationale.

6) Optional: stricter modes for research
- Add a `--strict_eval` flag to toggle strict 6‑field equality, and record both when enabled.

---

### How to Validate

- Quick spot‑check of parsing/leniency:
  - `uv run python debug_eval.py`
  - Confirm “Match (lenient)” shows True for samples where only counters differ.

- Standalone eval on base model or a checkpoint:
  - `uv run python evaluate_checkpoint.py jrahn/RookWorld-LM-124M`
  - Or pass a local checkpoint dir.

- Manual GRPO with periodic eval:
  - `uv run python manual_grpo_debug.py --task_type A --steps 5 --batch_size 4 --eval_every 1 --seed 42`
  - Watch the periodic “Eval accuracy” output; it should be non‑zero when positions are correct.

---

### Appendix: Why Ignore Halfmove/Fullmove Counters?

- The training objective and reward are primarily about correct position transitions and move legality/quality.
- Halfmove/fullmove counters depend on move history and are frequently miscounted by generative models despite having the correct position state.
- Requiring strict equality on counters penalizes correct chess positions, distorting measured progress.
- We still require en passant and castling rights to match, which are essential for position correctness.

---

### Files Changed in This Patch

- `manual_grpo_debug.py`: normalized completion parsing; lenient FEN equivalence for eval; safer move‑1 reporting; A‑task metric alignment.
- `evaluate_checkpoint.py`: normalization + prefix stripping; lenient FEN equivalence helpers.
- `debug_eval.py`: normalization + prefix stripping; lenient match indicator.

If you want, I can proceed to centralize the parsing/equivalence helpers in `src/rookworld_trl/utils.py` and add tests in `tests/`.

