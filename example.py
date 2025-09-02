#!/usr/bin/env python3
"""
Example usage of RookWorld TRL package with task-conditional generation
and explicit attention masks to avoid tokenizer warnings.
"""

from rookworld_trl.rewards import create_reward_function
from rookworld_trl.dataset import RookWorldDataGenerator
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def main():
    print("🧪 RookWorld TRL Package Example")
    print("=" * 50)
    
    # 1. Create reward function
    print("\n🏆 Creating reward function...")
    reward_fn = create_reward_function()
    
    # 2. Test reward calculation (pawn units for E:)
    print("\n🎯 Testing reward calculation...")
    prompt = "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    # Evaluations are expressed in pawns (dataset notation), e.g., 0.30 = +0.30 pawns
    completion = "M: e2e4 d2d4 g1f3 E: 0.30 0.35 0.28 B: e2e4"
    
    score = reward_fn([completion], prompts=[prompt])[0]
    print(f"Prompt: {prompt}")
    print(f"Completion: {completion}")
    print(f"Reward: {score:.3f}")
    
    # 3. Load dataset
    print("\n📊 Loading RookWorld dataset...")
    generator = RookWorldDataGenerator(dataset_size=10)
    info = generator.get_samples_info()
    print(f"Dataset info: {info}")
    
    # 4. Get mixed batch
    print("\n🎲 Getting mixed batch...")
    prompts = generator.get_mixed_batch(3)
    for i, prompt in enumerate(prompts):
        task_type = "P:" if prompt.startswith("P: ") else "A:" if prompt.startswith("A: ") else "?:"
        print(f"  {i+1}. {task_type} {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
    
    # 5. Generate and score one completion per prompt with task-conditional params
    print("\n🤖 Generating and scoring sample completions...")
    model_name = "jrahn/RookWorld-LM-124M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    model.eval()

    for i, p in enumerate(prompts):
        inputs = tokenizer(p, return_tensors="pt")
        is_a = p.startswith("A: ")
        temperature = 0.95 if is_a else 0.5
        top_p = 0.95 if is_a else 0.9
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=144,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        gen_completion = full_text[len(p):].strip()
        gen_score = reward_fn([gen_completion], prompts=[p])[0]
        print(f"  {i+1}. temp={temperature}, top_p={top_p} | reward={gen_score:.3f} | {gen_completion[:60]}{'...' if len(gen_completion) > 60 else ''}")
    
    print("\n✅ Package working correctly!")

if __name__ == "__main__":
    main()
