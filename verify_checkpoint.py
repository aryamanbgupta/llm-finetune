import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
import os
from pathlib import Path
import glob

# Same constants as training script
EXTRA_BASE = 40
OUTCOMES = ["0", "1", "2", "3", "4", "6", "WICKET"]
OUTCOME2TOK = {o: f"<|extra_{EXTRA_BASE+i}|>" for i, o in enumerate(OUTCOMES)}
TOK2OUTCOME = {v: k for k, v in OUTCOME2TOK.items()}

def setup_model_and_tokenizer(base_model_name="Qwen/Qwen1.5-1.8B", low_memory=False):
    """Setup base model and tokenizer with special tokens"""
    print(f"Loading base model: {base_model_name}")
    
    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Add special tokens
    special_tokens_to_add = list(OUTCOME2TOK.values())
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens_to_add})
    
    # Load base model with memory optimization for concurrent use
    attn_implementation = "flash_attention_2" if torch.cuda.is_available() else "eager"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    # Use less aggressive device mapping when training is running
    if low_memory:
        device_map = None  # Load to CPU first, move manually
        print("Low memory mode: Loading to CPU first")
    else:
        device_map = "auto" if torch.cuda.is_available() else None
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=False,
        attn_implementation=attn_implementation,
        low_cpu_mem_usage=True  # Reduce CPU memory during loading
    )
    
    # Move to GPU manually if needed
    if low_memory and torch.cuda.is_available():
        print("Moving model to GPU...")
        model = model.cuda()
    
    # Resize embeddings for special tokens
    model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the directory"""
    checkpoint_pattern = os.path.join(checkpoint_dir, "checkpoint-*")
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        return None
    
    # Extract step numbers and find the latest
    checkpoint_steps = []
    for cp in checkpoints:
        try:
            step = int(os.path.basename(cp).split('-')[1])
            checkpoint_steps.append((step, cp))
        except (IndexError, ValueError):
            continue
    
    if not checkpoint_steps:
        return None
    
    latest_step, latest_checkpoint = max(checkpoint_steps, key=lambda x: x[0])
    return latest_checkpoint, latest_step

def predict_dist(prompt: str, model, tokenizer):
    """Generate probability distribution over outcomes"""
    model.eval()
    prompt_with_space = prompt.rstrip() + " "
    inputs = tokenizer(prompt_with_space, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        last_token_logits = logits[0, -1, :]
        probabilities = torch.softmax(last_token_logits, dim=-1)

        outcome_token_ids = tokenizer.convert_tokens_to_ids(list(OUTCOME2TOK.values()))
        outcome_probs = probabilities[outcome_token_ids].cpu().tolist()
        
        dist = {outcome: prob for outcome, prob in zip(OUTCOMES, outcome_probs)}
        return dist

def main():
    parser = argparse.ArgumentParser(description="Check cricket model checkpoints")
    parser.add_argument("--checkpoint_dir", required=True, help="Directory containing checkpoints")
    parser.add_argument("--base_model", default="Qwen/Qwen1.5-1.8B", help="Base model name")
    parser.add_argument("--all_checkpoints", action="store_true", help="Test all checkpoints, not just latest")
    parser.add_argument("--low_memory", action="store_true", help="Use memory optimization for concurrent training")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint_dir):
        print(f"Error: Checkpoint directory {args.checkpoint_dir} not found")
        return
    
    # Test prompts
    test_prompts = [
        "0.3: 0/0 Need191@9.7 | Recent: ------0-0 | P:0@2b(0.0rr) | Ahmed(Econ0.0) vs Arya(new,0 Runs @ 0 SR) | PP 0.3 | Chennai Super Kings vs Punjab Kings, MA Chidambaram",
        "12.1: 74/2 | Recent: 1-1-0-0-1 | P:23@20b(6.9rr) | Ashwin(Econ5.3,1W) vs Kohli(set,31 Runs @ 94 SR) | Chennai Super Kings vs Royal Challengers Bangalore, MA Chidambaram",
        "18.1: 178/5 | Recent: 6-0-W-4-1 | P:6@2b(18.0rr) | Chahal(Econ12.0) vs Dhoni(new,5 Runs @ 167 SR) | Death 18.1 | Chennai Super Kings vs Punjab Kings, MA Chidambaram",
        "5.3: 42/1 | Recent: 0-4-0-1-1 | P:33@22b(9.0rr) | Kaul(Econ6.0) vs Shaw(set,30 Runs @ 150 SR) | PP 5.3 | Delhi Daredevils vs Sunrisers Hyderabad, RGI"
    ]
    
    # Setup base model
    base_model, tokenizer = setup_model_and_tokenizer(args.base_model, low_memory=args.low_memory)
    
    if args.all_checkpoints:
        # Test all checkpoints
        checkpoint_pattern = os.path.join(args.checkpoint_dir, "checkpoint-*")
        checkpoints = glob.glob(checkpoint_pattern)
        checkpoint_list = []
        for cp in checkpoints:
            try:
                step = int(os.path.basename(cp).split('-')[1])
                checkpoint_list.append((step, cp))
            except (IndexError, ValueError):
                continue
        checkpoint_list.sort(key=lambda x: x[0])
    else:
        # Just test latest
        latest_info = find_latest_checkpoint(args.checkpoint_dir)
        if latest_info is None:
            print(f"No checkpoints found in {args.checkpoint_dir}")
            return
        latest_checkpoint, latest_step = latest_info
        checkpoint_list = [(latest_step, latest_checkpoint)]
    
    print(f"Found {len(checkpoint_list)} checkpoint(s) to test\n")
    
    for step, checkpoint_path in checkpoint_list:
        print(f"{'='*60}")
        print(f"Testing checkpoint: {os.path.basename(checkpoint_path)} (Step {step})")
        print(f"{'='*60}")
        
        try:
            # Load PEFT model
            peft_model = PeftModel.from_pretrained(base_model, checkpoint_path)
            peft_model.eval()
            
            for i, prompt in enumerate(test_prompts, 1):
                print(f"\n--- Test Prompt {i} ---")
                print(f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
                
                dist = predict_dist(prompt, peft_model, tokenizer)
                
                # Sort by probability for easier reading
                sorted_outcomes = sorted(dist.items(), key=lambda x: x[1], reverse=True)
                
                print("Predictions:")
                for outcome, prob in sorted_outcomes:
                    print(f"  {outcome:7s}: {prob:.4f} {'*' if prob == max(dist.values()) else ''}")
                
                # Show confidence metrics
                max_prob = max(dist.values())
                entropy = -sum(p * torch.log2(torch.tensor(p + 1e-8)) for p in dist.values() if p > 0)
                print(f"  Confidence: {max_prob:.3f}, Entropy: {entropy:.3f}")
            
            # Cleanup to free memory aggressively
            del peft_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Wait for cleanup to complete
                
        except Exception as e:
            print(f"Error loading checkpoint {checkpoint_path}: {e}")
            # Cleanup on error too
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
    
    print(f"\n{'='*60}")
    print("Checkpoint testing complete!")
    
    # Final cleanup
    del base_model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

if __name__ == "__main__":
    main()