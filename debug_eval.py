#!/usr/bin/env python3
"""Debug script to test evaluation function and see actual model outputs."""

import json
import re
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from src.rookworld_trl.utils import normalize_spacing

def debug_eval():
    # Load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained("jrahn/RookWorld-LM-124M")
    tokenizer = GPT2Tokenizer.from_pretrained("jrahn/RookWorld-LM-124M")
    tokenizer.pad_token = tokenizer.eos_token

    # Load eval data
    with open("opening_dataset_eval.json") as f:
        eval_data = json.load(f)

    # Test first 3 samples
    for i, sample in enumerate(eval_data[:3]):
        print(f"\n=== Sample {i+1} (Move {sample['move_num']}) ===")
        print(f"Prompt: {sample['prompt']}")
        print(f"Expected: {sample['expected_fen']}")

        # Generate
        inputs = tokenizer(sample["prompt"], return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_len = inputs.input_ids.shape[1]
        completion = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)

        print(f"Raw completion: '{completion}'")

        # Normalize and parse FEN
        norm = normalize_spacing(completion)
        norm = re.sub(r"^\s*[PA]:\s*", "", norm)
        generated_fen = norm.split("+")[0].strip() if norm else ""

        def fen_equivalent(pred: str, exp: str) -> bool:
            if pred == exp:
                return True
            p_parts = pred.split()
            e_parts = exp.split()
            if len(p_parts) >= 4 and len(e_parts) >= 4:
                return p_parts[0] == e_parts[0] and p_parts[1] == e_parts[1] and p_parts[2] == e_parts[2] and p_parts[3] == e_parts[3]
            return False

        print(f"Parsed FEN: '{generated_fen}'")
        print(f"Match (lenient): {fen_equivalent(generated_fen, sample['expected_fen'])}")

        # Character-by-character comparison if no match
        if not fen_equivalent(generated_fen, sample['expected_fen']):
            print("Diff:")
            print(f"  Gen: {repr(generated_fen)}")
            print(f"  Exp: {repr(sample['expected_fen'])}")

if __name__ == "__main__":
    debug_eval()
