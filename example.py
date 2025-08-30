#!/usr/bin/env python3
"""
Example usage of RookWorld TRL package.
"""

from rookworld_trl.rewards import create_reward_function
from rookworld_trl.dataset import RookWorldDataGenerator

def main():
    print("🧪 RookWorld TRL Package Example")
    print("=" * 50)
    
    # 1. Create reward function
    print("\n🏆 Creating reward function...")
    reward_fn = create_reward_function()
    
    # 2. Test reward calculation
    print("\n🎯 Testing reward calculation...")
    prompt = "P: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    completion = "M: e2e4 d2d4 g1f3 E: 30 35 28 B: e2e4"
    
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
    
    print("\n✅ Package working correctly!")

if __name__ == "__main__":
    main()