"""
Verifiable reward system for chess GRPO training.
Combines format validation and content scoring.
"""

import re
import chess
import chess.engine
from typing import List, Dict, Any, Optional, Tuple, Union
import torch
import numpy as np
import time
import shutil
import os
from functools import lru_cache
import Levenshtein
from .utils import normalize_spacing


def find_stockfish_path() -> Optional[str]:
    """Automatically find Stockfish on the system."""
    # Common Stockfish installation paths
    common_paths = [
        "/usr/games/stockfish",
        "/usr/bin/stockfish", 
        "/usr/local/bin/stockfish",
        "/opt/homebrew/bin/stockfish",
        "stockfish"
    ]
    
    # Check each path
    for path in common_paths:
        if shutil.which(path) or (os.path.exists(path) and os.access(path, os.X_OK)):
            return path
    
    return None


class ChessRewardScorer:
    """Enhanced chess reward scorer with sophisticated Stockfish evaluation."""
    
    def __init__(self, stockfish_path: Optional[str] = None, depth: int = 10, time_limit: float = 0.1):
        self.stockfish_path = stockfish_path
        self.engine = None
        self.depth = depth
        self.time_limit = time_limit
        self.evaluation_cache = {}  # Simple position cache
        
        # Try to initialize Stockfish
        if stockfish_path:
            try:
                self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
                # Don't configure MultiPV here - it will be set per analysis
                self.engine.configure({"Threads": 1, "Hash": 32})  # Conservative settings
                print(f"✓ Stockfish initialized at {stockfish_path} (depth={depth}, time={time_limit}s)")
            except Exception as e:
                print(f"Warning: Could not initialize Stockfish: {e}")
                self.engine = None
        else:
            print("⚠️ No Stockfish path provided - using fallback scoring")
    
    def __del__(self):
        if self.engine:
            try:
                self.engine.quit()
            except:
                pass
    
    def get_stockfish_analysis(self, board: chess.Board, multipv: int = 5) -> Dict[str, Any]:
        """Get comprehensive Stockfish analysis for a position."""
        if not self.engine or not board.is_valid():
            return {"best_moves": [], "evaluations": [], "error": "No engine or invalid position"}
        
        # Use position FEN as cache key
        fen = board.fen()
        cache_key = f"{fen}_{multipv}"
        
        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key]
        
        try:
            # Deterministic analysis: use fixed depth rather than time-based limit
            limit = chess.engine.Limit(depth=self.depth)
            
            # Analyze with multiple principal variations
            infos = self.engine.analyse(board, limit, multipv=multipv)
            
            best_moves = []
            evaluations = []
            
            # Handle the results (infos is a list when multipv > 1)
            if isinstance(infos, list):
                # Multiple PV results
                for info in infos:
                    if 'pv' in info and info['pv']:
                        best_moves.append(info['pv'][0])
                        score = info.get('score')
                        if score and score.relative:
                            eval_cp = score.relative.score(mate_score=10000)
                            evaluations.append(eval_cp)
                        else:
                            evaluations.append(0)
            else:
                # Single PV result
                if 'pv' in infos and infos['pv']:
                    best_moves.append(infos['pv'][0])
                    score = infos.get('score')
                    if score and score.relative:
                        eval_cp = score.relative.score(mate_score=10000)
                        evaluations.append(eval_cp)
                    else:
                        evaluations.append(0)
            
            result = {
                "best_moves": best_moves,
                "evaluations": evaluations,
                "error": None
            }
            
            # Cache the result
            self.evaluation_cache[cache_key] = result
            return result
            
        except Exception as e:
            return {"best_moves": [], "evaluations": [], "error": str(e)}
    
    def batch_evaluate_positions(self, boards: List[chess.Board], max_time_per_position: float = None) -> List[Dict[str, Any]]:
        """Evaluate multiple positions efficiently with optimized batching."""
        if not self.engine:
            return [{"best_moves": [], "evaluations": [], "error": "No engine"}] * len(boards)
        
        if max_time_per_position is None:
            max_time_per_position = self.time_limit
        
        results = []
        
        # Check cache first for all positions
        uncached_indices = []
        for i, board in enumerate(boards):
            if board.is_valid():
                fen = board.fen()
                cache_key = f"{fen}_5"  # Top 5 moves
                if cache_key in self.evaluation_cache:
                    results.append(self.evaluation_cache[cache_key])
                else:
                    results.append(None)  # Placeholder
                    uncached_indices.append(i)
            else:
                results.append({"best_moves": [], "evaluations": [], "error": "Invalid position"})
        
        # Evaluate uncached positions with reduced time per position
        if uncached_indices:
            for idx in uncached_indices:
                board = boards[idx]
                
                # Get top 5 moves analysis (deterministic depth-based)
                try:
                    limit = chess.engine.Limit(depth=self.depth)
                    infos = self.engine.analyse(board, limit, multipv=5)
                    
                    best_moves = []
                    evaluations = []
                    
                    # Handle multipv results
                    if isinstance(infos, list):
                        for info in infos:
                            if 'pv' in info and info['pv']:
                                best_moves.append(info['pv'][0])
                                score = info.get('score')
                                if score and score.relative:
                                    eval_cp = score.relative.score(mate_score=10000)
                                    evaluations.append(eval_cp)
                                else:
                                    evaluations.append(0)
                    else:
                        # Single result fallback
                        if 'pv' in infos and infos['pv']:
                            best_moves.append(infos['pv'][0])
                            score = infos.get('score')
                            if score and score.relative:
                                eval_cp = score.relative.score(mate_score=10000)
                                evaluations.append(eval_cp)
                            else:
                                evaluations.append(0)
                    
                    result = {
                        "best_moves": best_moves,
                        "evaluations": evaluations,
                        "error": None
                    }
                    
                    # Cache and store
                    fen = board.fen()
                    cache_key = f"{fen}_5"
                    self.evaluation_cache[cache_key] = result
                    results[idx] = result
                    
                except Exception as e:
                    results[idx] = {"best_moves": [], "evaluations": [], "error": str(e)}
        
        return results
    
    def batch_evaluate_moves_after_playing(self, board: chess.Board, moves: List[chess.Move]) -> List[Optional[int]]:
        """Evaluate multiple moves from the same position efficiently."""
        if not self.engine or not board.is_valid():
            return [None] * len(moves)
        
        evaluations = []
        # Deterministic depth for post-move evaluation
        
        for move in moves:
            if move not in board.legal_moves:
                evaluations.append(None)  # Illegal move
                continue
            
            try:
                # Make move and evaluate resulting position
                board_copy = board.copy()
                board_copy.push(move)
                
                # Fixed-depth evaluation for determinism
                limit = chess.engine.Limit(depth=self.depth)
                info = self.engine.analyse(board_copy, limit)
                
                if 'score' in info and info['score'] and info['score'].relative:
                    # Flip sign because it's opponent's turn after the move
                    eval_cp = -info['score'].relative.score(mate_score=10000)
                    evaluations.append(eval_cp)
                else:
                    evaluations.append(0)
                    
            except Exception:
                evaluations.append(None)
        
        return evaluations
    
    def evaluate_move_after_playing(self, board: chess.Board, move: chess.Move) -> Optional[int]:
        """Evaluate position after playing a specific move."""
        if not self.engine or not board.is_valid():
            return None
            
        try:
            # Make a copy and play the move
            board_copy = board.copy()
            if move not in board_copy.legal_moves:
                return None  # Illegal move
                
            board_copy.push(move)
            
            # Get evaluation of resulting position
            analysis = self.get_stockfish_analysis(board_copy, multipv=1)
            if analysis["evaluations"]:
                # Return evaluation from the perspective of the player who just moved
                # (flip sign because it's now the opponent's turn)
                return -analysis["evaluations"][0]
            return None
            
        except Exception:
            return None
    
    def score_responses(self, prompts: List[str], responses: List[str]) -> List[float]:
        """Score a batch of responses."""
        scores = []
        for prompt, response in zip(prompts, responses):
            score = self._score_single_response(prompt, response)
            scores.append(score)
        return scores
    
    def _score_single_response(self, prompt: str, response: str) -> float:
        """Score a single response with format + content validation for RookWorld mixed tasks."""
        # Determine task type and extract relevant information
        if prompt.startswith("P: "):
            return self._score_policy_task(prompt, response)
        elif prompt.startswith("A: "):
            return self._score_environment_task(prompt, response)
        else:
            # Fallback: try to extract FEN and score as policy task
            return self._score_generic_chess_task(prompt, response)
    
    def _score_policy_task(self, prompt: str, response: str) -> float:
        """Score P: (Policy) task with sophisticated chess accuracy evaluation."""
        # Extract FEN from P: task format
        fen_match = re.search(r'P:\s*([^\s]+(?:\s+[^\s]+)*)', prompt)
        if not fen_match:
            return -0.5  # Strong penalty for malformed prompt
        
        fen_candidate = fen_match.group(1).strip()
        
        # Parse board - handle long FEN strings
        try:
            # Take first part that looks like a FEN (until space or end)
            fen_parts = fen_candidate.split()
            if len(fen_parts) >= 4:  # Minimum FEN parts
                fen = ' '.join(fen_parts[:6]) if len(fen_parts) >= 6 else ' '.join(fen_parts)
            else:
                fen = fen_candidate
            
            board = chess.Board(fen)
        except (ValueError, IndexError):
            return -0.5  # Strong penalty for invalid FEN
        
        # Extract sections from response
        sections = self._parse_policy_response(response)
        
        # If no engine available, fall back to basic scoring
        if not self.engine:
            return self._fallback_policy_scoring(sections, board)
        
        # Get Stockfish ground truth
        try:
            stockfish_analysis = self.get_stockfish_analysis(board, multipv=5)
            if stockfish_analysis["error"]:
                return self._fallback_policy_scoring(sections, board)
        except Exception:
            return self._fallback_policy_scoring(sections, board)
        
        # Calculate sophisticated scoring components
        best_move_score = self._score_best_move(sections.get("B", ""), board, stockfish_analysis)
        move_candidates_score = self._score_move_candidates(sections.get("M", []), board, stockfish_analysis) 
        evaluation_score = self._score_evaluations(sections.get("E", []), sections.get("M", []), board, stockfish_analysis)
        
        # Weighted combination: Best move (50%), Candidates (30%), Evaluations (20%)
        total_score = (0.5 * best_move_score + 
                      0.3 * move_candidates_score + 
                      0.2 * evaluation_score)
        
        # Clamp to [-1.0, 1.0] range
        return max(-1.0, min(1.0, total_score))
    
    def _parse_policy_response(self, response: str) -> Dict[str, Any]:
        """Parse P: task response into structured sections."""
        sections = {"B": "", "M": [], "E": []}
        
        # Extract best move (B: section)
        best_match = re.search(r'B:\s*([a-h][1-8][a-h][1-8][qrbnQRBN]?)', response)
        if best_match:
            sections["B"] = best_match.group(1).strip()
        
        # Extract move candidates (M: section) 
        moves_match = re.search(r'M:\s*([^E]*?)(?=E:|$)', response)
        if moves_match:
            moves_text = moves_match.group(1).strip()
            # Extract UCI moves
            move_patterns = re.findall(r'[a-h][1-8][a-h][1-8][qrbnQRBN]?', moves_text)
            sections["M"] = move_patterns[:5]  # Max 5 moves
        
        # Extract evaluations (E: section)
        evals_match = re.search(r'E:\s*([-+]?\d*\.?\d+(?:\s+[-+]?\d*\.?\d+)*)', response)
        if evals_match:
            eval_text = evals_match.group(1).strip()
            try:
                sections["E"] = [float(x) for x in eval_text.split()]
            except ValueError:
                sections["E"] = []
        
        return sections
    
    def _score_best_move(self, predicted_best: str, board: chess.Board, stockfish_analysis: Dict) -> float:
        """Score the predicted best move against Stockfish."""
        if not predicted_best or not stockfish_analysis["best_moves"]:
            return -0.5  # Strong penalty for missing best move
        
        try:
            # Parse predicted move
            predicted_move = chess.Move.from_uci(predicted_best)
            if predicted_move not in board.legal_moves:
                return -1.0  # Illegal move gets maximum penalty
            
            stockfish_best_moves = stockfish_analysis["best_moves"]
            
            # Exact match with Stockfish #1 move
            if predicted_move == stockfish_best_moves[0]:
                return 1.0
            
            # Check if in top moves with decreasing reward
            for i, sf_move in enumerate(stockfish_best_moves):
                if predicted_move == sf_move:
                    # Linear decay: 2nd best = 0.8, 3rd = 0.6, 4th = 0.4, 5th = 0.2
                    return 1.0 - (i * 0.2)
            
            # Legal move but not in top 5 - small positive score
            return 0.1
            
        except (ValueError, chess.InvalidMoveError):
            return -1.0  # Invalid move format
    
    def _score_move_candidates(self, predicted_moves: List[str], board: chess.Board, stockfish_analysis: Dict) -> float:
        """Score the quality of move candidates using set overlap."""
        if not predicted_moves:
            return -0.3  # Penalty for no moves
        
        if not stockfish_analysis["best_moves"]:
            return 0.0  # No ground truth available
        
        # Parse and validate predicted moves
        valid_predicted = []
        illegal_count = 0
        
        for move_str in predicted_moves:
            try:
                move = chess.Move.from_uci(move_str)
                if move in board.legal_moves:
                    valid_predicted.append(move)
                else:
                    illegal_count += 1
            except:
                illegal_count += 1
        
        # Strong penalty for illegal moves
        illegal_penalty = illegal_count * 0.3
        
        if not valid_predicted:
            return -illegal_penalty  # Only illegal moves
        
        # Calculate set intersection with Stockfish top moves
        stockfish_moves = set(stockfish_analysis["best_moves"][:5])
        predicted_set = set(valid_predicted)
        intersection = stockfish_moves & predicted_set
        
        # Jaccard similarity with bonus for move order
        jaccard = len(intersection) / len(stockfish_moves | predicted_set)
        
        # Bonus for getting top moves in correct order
        order_bonus = 0.0
        for i, pred_move in enumerate(valid_predicted[:3]):
            if i < len(stockfish_analysis["best_moves"]) and pred_move == stockfish_analysis["best_moves"][i]:
                order_bonus += 0.1 * (3 - i) / 3  # Higher bonus for earlier positions
        
        score = jaccard + order_bonus - illegal_penalty
        return max(-1.0, min(1.0, score))
    
    def _score_evaluations(self, predicted_evals: List[float], predicted_moves: List[str], 
                          board: chess.Board, stockfish_analysis: Dict) -> float:
        """Score evaluation accuracy using MAE regression."""
        if not predicted_evals or not predicted_moves:
            return -0.2  # Penalty for missing evaluations
        
        # Check count consistency
        if len(predicted_evals) != len(predicted_moves):
            return -0.4  # Strong penalty for count mismatch
        
        # Parse and validate all moves first
        parsed_moves = []
        for move_str in predicted_moves:
            try:
                move = chess.Move.from_uci(move_str)
                if move in board.legal_moves:
                    parsed_moves.append(move)
                else:
                    return -0.5  # Illegal move in evaluations
            except:
                return -0.5  # Invalid move format
        
        # Get ground truth evaluations using batch method (returns centipawns)
        true_evals_cp = self.batch_evaluate_moves_after_playing(board, parsed_moves)
        
        # Filter out None values and replace with fallback, then convert to pawns
        true_evals_pawn = [
            (eval_cp / 100.0) if (eval_cp is not None) else 0.0
            for eval_cp in true_evals_cp
        ]
        
        if not true_evals_pawn:
            return 0.0  # No evaluations possible
        
        # Calculate MAE in pawn units and convert to reward
        mae = np.mean([abs(pred - true) for pred, true in zip(predicted_evals, true_evals_pawn)])
        
        # Check for sign errors (position misunderstanding) in pawn units
        sign_errors = 0
        for pred, true in zip(predicted_evals, true_evals_pawn):
            # Penalize sign mismatch when true advantage is meaningful (> 0.2 pawns)
            if abs(true) > 0.2 and np.sign(pred) != np.sign(true):
                sign_errors += 1
        
        # Convert MAE to reward (lower MAE = higher reward) in pawns
        # MAE of 0.0 pawns => 1.0, MAE of 1.0 pawn => 0.0
        mae_reward = max(0.0, 1.0 - mae / 1.0)
        
        # Strong penalty for sign errors
        sign_penalty = sign_errors * 0.4
        
        return max(-1.0, mae_reward - sign_penalty)
    
    def _fallback_policy_scoring(self, sections: Dict[str, Any], board: chess.Board) -> float:
        """Fallback scoring when Stockfish is not available."""
        score = 0.0
        
        # Basic format scoring
        if sections["B"]:
            score += 0.3
        if sections["M"]:
            score += 0.2 * min(len(sections["M"]), 5) / 5
        if sections["E"]:
            score += 0.2
        
        # Basic legality check
        try:
            if sections["B"]:
                move = chess.Move.from_uci(sections["B"])
                if move not in board.legal_moves:
                    score -= 0.5
            
            for move_str in sections["M"]:
                move = chess.Move.from_uci(move_str)
                if move not in board.legal_moves:
                    score -= 0.1
        except:
            score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    def _score_environment_task(self, prompt: str, response: str) -> float:
        """Score A: (Environment) task with asymmetric Levenshtein-based rewards.

        Reward structure:
        - Schema validation: 0 to +0.3 (30%)
        - FEN accuracy: +0.5 for perfect, -0.5 to 0 for errors (50%)
        - FEN validity: -0.2 to +0.2 (20%)

        Total range: -1.0 to +1.0
        """
        # Extract FEN and move from A: task format: "A: FEN+move+history+"
        env_match = re.search(r'A:\s*([^+]+)\+([^+]+)\+([^+]*)\+?', prompt)
        if not env_match:
            return -0.5  # Strong penalty for malformed prompt

        fen_candidate = env_match.group(1).strip()
        move_candidate = env_match.group(2).strip()
        history = env_match.group(3).strip() if env_match.group(3) else ""

        try:
            # Parse the board and move
            board = chess.Board(fen_candidate)
            move = chess.Move.from_uci(move_candidate)

            if move not in board.legal_moves:
                return -0.8  # Strong penalty for illegal move in prompt

            # Calculate expected FEN after move
            board_copy = board.copy()
            board_copy.push(move)
            expected_fen = board_copy.fen()

            # Score the response
            schema_score = self._score_environment_schema(response)
            fen_score = self._score_environment_fen_accuracy(response, expected_fen)
            validity_score = self._score_environment_validity(response, board_copy)

            total_score = schema_score + fen_score + validity_score

            return max(-1.0, min(1.0, total_score))

        except (ValueError, AttributeError, chess.InvalidMoveError) as e:
            # Invalid board or move format
            return -0.5

    def _score_environment_schema(self, response: str) -> float:
        """Score schema compliance for A: task response (0 to +0.03)."""
        score = 0.0

        # Check for correct delimiter structure
        parts = response.split("+")

        # Correct field count (4 fields: fen, reward, terminated, truncated)
        if len(parts) >= 4:
            score += 0.01

            # Validate field types
            # Field 1: FEN-like string (contains "/" and pieces)
            if re.search(r'[rnbqkpRNBQKP1-8]+/[rnbqkpRNBQKP1-8/]+', parts[0]):
                score += 0.005

            # Field 2: Float reward
            try:
                float(parts[1])
                score += 0.005
            except ValueError:
                pass

            # Field 3: 0/1 for terminated
            if parts[2].strip() in ['0', '1', 'true', 'false']:
                score += 0.005

            # Field 4: 0/1 for truncated
            if len(parts) > 3 and parts[3].strip() in ['0', '1', 'true', 'false']:
                score += 0.005

        return min(0.03, score)

    def _score_environment_fen_accuracy(self, response: str, expected_fen: str) -> float:
        """Score FEN accuracy with asymmetric rewards (+0.5 for perfect, -0.7 for ANY errors).

        IMPORTANT: Only the COMPLETE FEN (up to move numbers) must be correct for positive reward.
        This includes: board position, turn, castling rights, en passant, halfmove, fullmove.
        """
        # Extract FEN from response
        parts = response.split("+")
        if not parts or not parts[0].strip():
            return -0.7  # No FEN found

        predicted_fen = parts[0].strip()

        # Check if it's a complete FEN (should have 6 space-separated parts)
        predicted_parts = predicted_fen.split()
        expected_parts = expected_fen.split()

        if len(predicted_parts) < 6:
            # Incomplete FEN - missing turn, castling, en passant, or move counts
            return -0.7  # Severe penalty for incomplete response

        # Compare the full FEN up to move numbers
        # This ensures en passant, castling rights, etc. are all correct

        # Option 1: Exact match on everything = +0.5
        if predicted_fen == expected_fen:
            return 0.5

        # Option 2: If move numbers differ slightly but everything else is perfect
        # Compare first 5 parts (excluding fullmove number which might vary)
        if len(predicted_parts) >= 6 and len(expected_parts) >= 6:
            # Check if everything except maybe move numbers is correct
            if (predicted_parts[0] == expected_parts[0] and  # Board position
                predicted_parts[1] == expected_parts[1] and  # Turn
                predicted_parts[2] == expected_parts[2] and  # Castling
                predicted_parts[3] == expected_parts[3] and  # En passant
                predicted_parts[4] == expected_parts[4]):    # Halfmove clock
                # Everything except fullmove is correct - still reward
                if predicted_parts[5] == expected_parts[5]:
                    return 0.5  # Perfect match
                else:
                    # Only fullmove differs - minor penalty
                    return -0.3

        # ANY other mismatch (including missing en passant) gets maximum penalty
        # This is crucial for training the model to handle en passant correctly
        return -0.7

    def _score_environment_validity(self, response: str, expected_board: chess.Board) -> float:
        """Score FEN validity and game state preservation (-0.2 to +0.01)."""
        parts = response.split("+")
        if not parts or not parts[0].strip():
            return -0.2

        predicted_fen = parts[0].strip()

        try:
            # Try to parse as valid FEN
            predicted_board = chess.Board(predicted_fen)

            # Valid parse gets minimal base score
            score = 0.005

            # Check for common errors
            # Count kings - should be exactly one per color
            white_kings = str(predicted_board).count('K')
            black_kings = str(predicted_board).count('k')

            if white_kings != 1 or black_kings != 1:
                return -0.2  # Penalize extra/missing kings heavily

            # Check if turn and castling rights are preserved correctly
            expected_fen_parts = expected_board.fen().split()
            predicted_fen_parts = predicted_fen.split()

            if len(predicted_fen_parts) >= 2 and len(expected_fen_parts) >= 2:
                # Correct turn
                if predicted_fen_parts[1] == expected_fen_parts[1]:
                    score += 0.0025

                # Correct castling rights (if specified)
                if len(predicted_fen_parts) >= 3 and len(expected_fen_parts) >= 3:
                    if predicted_fen_parts[2] == expected_fen_parts[2]:
                        score += 0.0025

            return min(0.01, score)

        except (ValueError, AttributeError):
            # Invalid FEN that can't be parsed
            return -0.2

    def _score_generic_chess_task(self, prompt: str, response: str) -> float:
        """Fallback scoring for unclear task format."""
        # Look for any FEN-like pattern
        fen_patterns = [
            r'([rnbqkpRNBQKP1-8]+/[rnbqkpRNBQKP1-8/]+\s+[wb]\s+[KQkq-]+\s+[a-h1-8-]+\s+\d+\s+\d+)',
            r'([a-h1-8]+/[a-h1-8/]+)',
        ]
        
        for pattern in fen_patterns:
            fen_match = re.search(pattern, prompt)
            if fen_match:
                try:
                    board = chess.Board(fen_match.group(1))
                    # Basic format + content scoring
                    format_score = self._score_basic_format(response)
                    content_score = self._score_basic_content(board, response)
                    return 0.5 * format_score + 0.5 * content_score
                except ValueError:
                    continue
        
        # No valid chess position found - minimal score for any reasonable text
        return 0.1 if len(response.strip()) > 10 else 0.0
    
    def _score_format(self, response: str) -> float:
        """Score format adherence (0.0 to 1.0)."""
        score = 0.0
        
        # Check for move notation
        if re.search(r'\b[a-h][1-8][a-h][1-8]\b|[NBRQK]?[a-h]?[1-8]?x?[a-h][1-8]', response):
            score += 0.3
        
        # Check for evaluation
        if re.search(r'[+-]?\d+\.?\d*', response):
            score += 0.3
        
        # Check for reasonable structure
        if len(response.strip()) > 10 and len(response.strip()) < 200:
            score += 0.2
        
        # Bonus for chess keywords
        chess_keywords = ['move', 'best', 'analysis', 'position', 'evaluation']
        if any(keyword in response.lower() for keyword in chess_keywords):
            score += 0.2
        
        return min(score, 1.0)
    
    def _score_content(self, board: chess.Board, response: str) -> float:
        """Score content accuracy using basic chess validation."""
        if not board.is_valid():
            return 0.0
        
        score = 0.0
        
        # Extract moves from response
        moves = self._extract_moves(response)
        legal_moves = list(board.legal_moves)
        
        if not moves:
            return 0.2  # Minimal score for no moves
        
        # Check if suggested moves are legal
        valid_moves = 0
        for move_str in moves[:3]:  # Check first 3 moves
            try:
                move = board.parse_san(move_str) if len(move_str) < 6 else chess.Move.from_uci(move_str)
                if move in legal_moves:
                    valid_moves += 1
            except (ValueError, chess.InvalidMoveError):
                continue
        
        if moves:
            score += 0.5 * (valid_moves / min(len(moves), 3))
        
        # Stockfish comparison (if available)
        if self.engine and valid_moves > 0:
            try:
                info = self.engine.analyse(board, chess.engine.Limit(time=0.1))
                best_move = info['pv'][0] if 'pv' in info else None
                
                if best_move:
                    best_move_str = board.san(best_move)
                    if best_move_str in response or str(best_move) in response:
                        score += 0.5
            except Exception:
                pass
        else:
            # Basic positional scoring without engine
            score += 0.3
        
        return min(score, 1.0)
    
    def _extract_moves(self, text: str) -> List[str]:
        """Extract chess moves from text."""
        # UCI format
        uci_moves = re.findall(r'\b[a-h][1-8][a-h][1-8][qrbnQRBN]?\b', text)
        
        # SAN format  
        san_moves = re.findall(r'\b(?:[NBRQK]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?|O-O(?:-O)?)\b', text)
        
        return uci_moves + san_moves
    
    
    def _score_environment_format(self, response: str) -> float:
        """Score format for A: task responses."""
        score = 0.0
        
        # Look for FEN-like patterns
        if re.search(r'[rnbqkpRNBQKP1-8]+/[rnbqkpRNBQKP1-8/]+', response):
            score += 0.4
        
        # Look for reward patterns
        if re.search(r'[-+]?\d*\.?\d+', response):
            score += 0.3
        
        # Look for boolean patterns (true/false)
        if re.search(r'\b(?:true|false)\b', response.lower()):
            score += 0.3
        
        return min(score, 1.0)
    
    def _score_environment_content(self, board: chess.Board, move: chess.Move, response: str) -> float:
        """Score content for A: task responses."""
        score = 0.0
        
        # Simulate the move to get expected result
        try:
            board_copy = board.copy()
            board_copy.push(move)
            expected_fen = board_copy.fen()
            
            # Check if response contains similar FEN
            if expected_fen.split()[0] in response:  # Just check piece positions
                score += 0.6
            
            # Check for reasonable reward values
            reward_match = re.search(r'([-+]?\d*\.?\d+)', response)
            if reward_match:
                reward_val = float(reward_match.group(1))
                if -1.0 <= reward_val <= 1.0:  # Reasonable reward range
                    score += 0.4
        except:
            pass
        
        return min(score, 1.0)
    
    def _score_basic_format(self, response: str) -> float:
        """Basic format scoring for fallback cases."""
        score = 0.0
        
        if re.search(r'\b[a-h][1-8][a-h][1-8]\b', response):
            score += 0.3
        
        if re.search(r'[-+]?\d*\.?\d+', response):
            score += 0.3
        
        if len(response.strip()) > 20:
            score += 0.2
        
        chess_words = ['move', 'position', 'piece', 'king', 'queen', 'rook', 'bishop', 'knight', 'pawn']
        if any(word in response.lower() for word in chess_words):
            score += 0.2
        
        return min(score, 1.0)
    
    def _score_basic_content(self, board: chess.Board, response: str) -> float:
        """Basic content scoring for fallback cases."""
        if not board.is_valid():
            return 0.0
        
        moves = self._extract_moves(response)
        if not moves:
            return 0.3  # Some score for non-move responses
        
        legal_moves = list(board.legal_moves)
        valid_count = 0
        
        for move_str in moves[:3]:
            try:
                move = board.parse_san(move_str) if len(move_str) < 6 else chess.Move.from_uci(move_str)
                if move in legal_moves:
                    valid_count += 1
            except:
                continue
        
        return valid_count / min(len(moves), 3) if moves else 0.3


def create_reward_function(stockfish_path: Optional[str] = None):
    """Create reward function for GRPO training."""
    # Auto-detect Stockfish if no path provided
    if stockfish_path is None:
        stockfish_path = find_stockfish_path()
        if stockfish_path:
            print(f"✓ Auto-detected Stockfish at: {stockfish_path}")
        else:
            print("⚠️ Stockfish not found - will use fallback scoring")
    
    scorer = ChessRewardScorer(stockfish_path)
    
    def reward_function(completions: List[str], **kwargs) -> List[float]:
        """
        GRPO reward function that takes completions and returns rewards.
        
        Args:
            completions: List of generated completions/responses
            **kwargs: Additional context (may include prompts)
        
        Returns:
            List of reward scores for each completion
        """
        # Normalize spacing in completions to prevent KL divergence inflation
        normalized_completions = [normalize_spacing(completion) for completion in completions]
        
        # Extract prompts from kwargs if available, otherwise create dummy prompts
        prompts = kwargs.get("prompts", None)
        if prompts is None:
            # Create dummy prompts - in actual GRPO training, prompts should be provided
            prompts = ["P:rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 M:"] * len(completions)
        
        # Ensure prompts and completions have the same length
        if len(prompts) != len(completions):
            # Repeat prompts to match completions if needed
            prompts = prompts * (len(completions) // len(prompts) + 1)
            prompts = prompts[:len(completions)]
        
        return scorer.score_responses(prompts, normalized_completions)
    
    return reward_function
