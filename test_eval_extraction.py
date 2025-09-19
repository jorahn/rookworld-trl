#!/usr/bin/env python3
"""Test evaluation FEN extraction and matching."""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from src.rookworld_trl.utils import normalize_spacing

def test_evaluation():
    # Load model
    model_name = "jrahn/RookWorld-LM-124M"
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.eval()

    # Load eval dataset
    with open("opening_dataset_eval.json") as f:
        eval_data = json.load(f)

    # Test first 10 samples
    samples = eval_data[:10]

    def _extract_generated_fen(text: str) -> str:
        # Normalize spacing and strip task markers
        norm = normalize_spacing(text)
        norm = re.sub(r"^\s*[PA]:\s*", "", norm)
        # Extract FEN before first +
        return norm.split("+")[0].strip()

    exact_matches = 0

    print("\nTesting FEN extraction and matching:")
    print("=" * 80)

    for i, sample in enumerate(samples):
        prompt = sample["prompt"]
        expected_fen = sample["expected_fen"]

        # Generate completion
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=100,
                do_sample=False,  # Greedy decoding
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Extract generated text
        prompt_len = inputs.input_ids.shape[1]
        completion = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)

        # Extract FEN from completion
        generated_fen = _extract_generated_fen(completion)

        # Check exact match
        is_match = generated_fen == expected_fen
        if is_match:
            exact_matches += 1

        print(f"\nSample {i+1}:")
        print(f"  Completion: {completion[:100]}...")
        print(f"  Extracted FEN: {generated_fen}")
        print(f"  Expected FEN:  {expected_fen}")
        print(f"  Exact match: {is_match}")

        if not is_match and generated_fen:
            # Check what's different
            gen_parts = generated_fen.split()
            exp_parts = expected_fen.split()
            if len(gen_parts) >= 4 and len(exp_parts) >= 4:
                if gen_parts[:4] == exp_parts[:4]:
                    print(f"  -> Position matches, counters differ")
                else:
                    print(f"  -> Position differs")

    print("\n" + "=" * 80)
    print(f"Summary: {exact_matches}/10 exact matches ({exact_matches*10}%)")

if __name__ == "__main__":
    test_evaluation()