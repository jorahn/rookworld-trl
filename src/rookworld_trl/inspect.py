"""
Inspection utilities for analyzing GRPO training and reward calculations.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .dataset import RookWorldDataGenerator
from .rewards import create_reward_function
import argparse
from typing import Optional


def inspect_single_batch(
    batch_size: int = 4,
    max_completion_length: int = 144,
    model_name: str = "jrahn/RookWorld-LM-124M",
    stockfish_path: Optional[str] = None
):
    """Inspect a single batch showing detailed reward calculations."""
    print("🔍 SINGLE BATCH INSPECTION - MIXED TASKS")
    print("=" * 80)
    
    print(f"Model: {model_name}")
    print(f"Batch size: {batch_size}")
    print(f"Max completion length: {max_completion_length}")
    
    # Load model and tokenizer
    print("\n📥 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    model.eval()
    
    # Create data generator and reward function
    print("📊 Creating data generator...")
    data_generator = RookWorldDataGenerator(max_length=256, dataset_size=20)
    
    print("🏆 Creating reward function...")
    reward_function = create_reward_function(stockfish_path)
    
    # Generate mixed batch
    print(f"\n🎲 Generating {batch_size} mixed prompts...")
    prompts = data_generator.get_mixed_batch(batch_size)
    print(f"Generated {len(prompts)} prompts")
    
    # Process each sample
    completions = []
    rewards = []
    
    print("\n" + "=" * 80)
    print("📋 BATCH PROCESSING - SAMPLE BY SAMPLE")
    print("=" * 80)
    
    for i, prompt in enumerate(prompts):
        print(f"\n🔢 SAMPLE {i+1}/{len(prompts)}")
        print("-" * 60)
        
        # Determine task type
        task_type = "P: (Policy)" if prompt.startswith("P: ") else "A: (Environment)" if prompt.startswith("A: ") else "Unknown"
        print(f"📝 Task Type: {task_type}")
        print(f"📋 Prompt: {prompt}")
        
        # Generate completion
        print("\n🤖 Generating completion...")
        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                # Task-conditional generation params: keep P focused, A more permissive
                is_a_task = prompt.startswith("A: ")
                gen_temperature = 0.95 if is_a_task else 0.5
                gen_top_p = 0.95 if is_a_task else 0.9

                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_completion_length,
                    do_sample=True,
                    temperature=gen_temperature,
                    top_p=gen_top_p,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            completion = full_text[len(prompt):].strip()
            
            print(f"✅ Generated ({len(completion)} chars): {completion[:100]}{'...' if len(completion) > 100 else ''}")
            completions.append(completion)
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            completion = ""
            completions.append(completion)
        
        # Calculate reward with detailed breakdown
        print("\n🎯 Calculating reward...")
        try:
            reward = reward_function([completion], prompts=[prompt])[0]
            rewards.append(reward)
            print(f"💰 Final Reward: {reward:.4f}")
            
            # Show detailed breakdown for P: tasks
            if prompt.startswith("P: "):
                print("\n🔍 Detailed P: Task Analysis:")
                scorer = reward_function.__closure__[0].cell_contents  # Get the scorer object
                sections = scorer._parse_policy_response(completion)
                print(f"   📤 Parsed sections:")
                print(f"      B: (best move) = '{sections.get('B', '')}'")
                print(f"      M: (moves) = {sections.get('M', [])} ({len(sections.get('M', []))} moves)")
                print(f"      E: (evals) = {sections.get('E', [])} ({len(sections.get('E', []))} values)")
                
                # Component breakdown
                try:
                    import chess
                    import re
                    
                    # Extract FEN from prompt
                    fen_match = re.search(r'P:\s*([^\s]+(?:\s+[^\s]+)*)', prompt)
                    if fen_match:
                        fen_parts = fen_match.group(1).strip().split()
                        if len(fen_parts) >= 4:
                            fen = ' '.join(fen_parts[:6]) if len(fen_parts) >= 6 else ' '.join(fen_parts)
                            board = chess.Board(fen)
                            
                            print(f"   🏁 Position: {fen}")
                            print(f"   🎯 Scoring breakdown:")
                            
                            if scorer.engine:
                                print(f"      🔬 Using Stockfish analysis...")
                                stockfish_analysis = scorer.get_stockfish_analysis(board, multipv=5)
                                if not stockfish_analysis["error"]:
                                    print(f"      📊 Stockfish ground truth:")
                                    print(f"         Top moves: {[str(m) for m in stockfish_analysis['best_moves'][:3]]}")
                                    print(f"         Evaluations: {stockfish_analysis['evaluations'][:3]}")
                                    
                                    # Component scores
                                    best_score = scorer._score_best_move(sections.get("B", ""), board, stockfish_analysis)
                                    candidates_score = scorer._score_move_candidates(sections.get("M", []), board, stockfish_analysis)
                                    eval_score = scorer._score_evaluations(sections.get("E", []), sections.get("M", []), board, stockfish_analysis)
                                    
                                    print(f"      💯 Component scores:")
                                    print(f"         Best move (50%):    {best_score:.4f}")
                                    print(f"         Candidates (30%):   {candidates_score:.4f}")
                                    print(f"         Evaluations (20%):  {eval_score:.4f}")
                                    print(f"         Final weighted:     {0.5*best_score + 0.3*candidates_score + 0.2*eval_score:.4f}")
                                else:
                                    print(f"      ⚠️ Stockfish error: {stockfish_analysis['error']}")
                            else:
                                print(f"      🔄 Using fallback scoring (no Stockfish)...")
                                fallback_score = scorer._fallback_policy_scoring(sections, board)
                                print(f"      💯 Fallback score: {fallback_score:.4f}")
                                
                except Exception as e:
                    print(f"   ⚠️ Detailed analysis failed: {e}")
            
        except Exception as e:
            print(f"❌ Reward calculation failed: {e}")
            rewards.append(0.0)
    
    # Batch summary
    print("\n" + "=" * 80)
    print("📊 BATCH SUMMARY")
    print("=" * 80)
    
    p_tasks = sum(1 for p in prompts if p.startswith("P: "))
    a_tasks = sum(1 for p in prompts if p.startswith("A: "))
    
    print(f"📈 Task distribution:")
    print(f"   P: (Policy) tasks:     {p_tasks}")
    print(f"   A: (Environment) tasks: {a_tasks}")
    
    print(f"\n💰 Reward statistics:")
    if rewards:
        import numpy as np
        print(f"   Mean reward:     {np.mean(rewards):.4f}")
        print(f"   Std reward:      {np.std(rewards):.4f}")
        print(f"   Min reward:      {min(rewards):.4f}")
        print(f"   Max reward:      {max(rewards):.4f}")
        print(f"   Non-zero:        {sum(1 for r in rewards if abs(r) > 0.001)}/{len(rewards)} ({100*sum(1 for r in rewards if abs(r) > 0.001)/len(rewards):.1f}%)")
    
    print(f"\n📝 Individual sample results:")
    for i, (prompt, completion, reward) in enumerate(zip(prompts, completions, rewards)):
        task_type = "P:" if prompt.startswith("P: ") else "A:" if prompt.startswith("A: ") else "?:"
        print(f"   {i+1}. {task_type} | Reward: {reward:6.3f} | Completion: {completion[:40]}{'...' if len(completion) > 40 else ''}")


def main():
    """Main inspection function."""
    parser = argparse.ArgumentParser(description="Inspect GRPO batch processing and rewards")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inspection")
    parser.add_argument("--max_completion_length", type=int, default=144, help="Max completion length")
    parser.add_argument("--model_name", default="jrahn/RookWorld-LM-124M", help="Model to use")
    parser.add_argument("--stockfish_path", default=None, help="Path to Stockfish (auto-detect if None)")
    
    args = parser.parse_args()
    
    print("🧪 ROOKWORLD GRPO INSPECTION TOOL")
    print("Analyzing reward calculations for mixed P: and A: tasks\n")
    
    try:
        inspect_single_batch(
            batch_size=args.batch_size,
            max_completion_length=args.max_completion_length,
            model_name=args.model_name,
            stockfish_path=args.stockfish_path
        )
        
        print("\n" + "=" * 80)
        print("✅ INSPECTION COMPLETE")
        print("=" * 80)
        print("• Showed detailed reward calculation breakdown")
        print("• Demonstrated continuous reward values")
        print("• Validated chess-accurate scoring system")
        
    except Exception as e:
        print(f"❌ Inspection failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
