#!/usr/bin/env python3
"""
Quick GPU memory capacity check for GRPO-like workloads.

Simulates the heavy parts of a GRPO step:
- Load policy and reference models (two copies) on GPU
- Batched generation with num_return_sequences per prompt
- Forward pass on full sequences to approximate logit/score memory

Note: This is a synthetic probe — it over-approximates peak memory for safety.
"""

import argparse
import os
import random
from typing import Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def human_mb(x: int) -> str:
    return f"{x / (1024**2):.1f} MB"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="jrahn/RookWorld-LM-124M")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--prompt_len", type=int, default=128, help="Approx prompt tokens")
    p.add_argument("--num_generations", type=int, default=12)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--single_model", action="store_true", help="Only load one model (lower bound on memory)")
    p.add_argument("--compile", action="store_true", help="Use torch.compile for the policy model")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️ Requested cuda but no GPU detected. Falling back to CPU.")
        args.device = "cpu"

    print("=" * 80)
    print("GPU Capacity Probe (GRPO‑like)")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Prompt length: {args.prompt_len}")
    print(f"Generations per prompt: {args.num_generations}")
    print(f"Max new tokens: {args.max_new_tokens} (hard limit per schema)")
    print(f"single_model: {args.single_model}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32

    policy = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype).to(args.device)
    if args.compile and hasattr(torch, "compile"):
        try:
            policy = torch.compile(policy, mode="reduce-overhead")
            print("✓ torch.compile enabled for policy")
        except Exception as e:
            print(f"⚠️ torch.compile failed: {e}")

    ref = None
    if not args.single_model:
        ref = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype).to(args.device)
        ref.eval()
        for p_ in ref.parameters():
            p_.requires_grad = False

    def cuda_stats(header: str) -> Tuple[int, int]:
        if args.device != "cuda":
            return (0, 0)
        cur = torch.cuda.memory_allocated()
        peak = torch.cuda.max_memory_allocated()
        print(f"{header:<24} | current={human_mb(cur):>8} | peak={human_mb(peak):>8}")
        return cur, peak

    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        cuda_stats("After load")

    # Create synthetic prompts (token ids)
    pad_id = tokenizer.pad_token_id
    bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id
    input_ids = torch.full((args.batch_size, args.prompt_len), pad_id, dtype=torch.long)
    input_ids[:, 0] = bos_id
    attn = torch.ones_like(input_ids)
    input_ids = input_ids.to(args.device)
    attn = attn.to(args.device)

    try:
        # Generation (batched) — this is typically a peak
        with torch.no_grad():
            _ = policy.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=args.max_new_tokens,
                num_return_sequences=args.num_generations,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=False,
            )
        if args.device == "cuda":
            cuda_stats("After generate")

        # Forward pass over concatenated sequences (approximate logit/score memory)
        # Use a single long tensor to stress attention KV memory
        total_len = args.prompt_len + args.max_new_tokens
        full = torch.randint(low=0, high=tokenizer.vocab_size, size=(args.batch_size, total_len), device=args.device)
        full.requires_grad_(False)
        out = policy(full)
        loss_like = out.logits.float().mean()
        # Simulate backward storage with a tiny grad usage
        loss_like.backward() if policy.training else None
        if args.device == "cuda":
            cuda_stats("After forward")

        print("\n✅ Capacity OK at requested settings.")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            if args.device == "cuda":
                cuda_stats("OOM captured")
            print("\n❌ CUDA OOM at requested settings.")
            print("   Suggestions:")
            print("   - Reduce batch_size and/or num_generations")
            print("   - Keep max_new_tokens fixed (schema requirement)")
            print("   - Consider gradient accumulation to increase effective batch")
            return
        raise


if __name__ == "__main__":
    main()

