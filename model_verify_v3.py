#!/usr/bin/env python3
"""
Test your cricket model checkpoint locally on Mac Mini
Usage: python cricket_model_tester.py
"""
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# --- Setup ---
model_name = 'Qwen/Qwen1.5-1.8B'  # Updated to match your training script
adapter_path = '/Users/aryamangupta/CricML/checkpoint-18500'  # Update with your actual path
device = "mps" if torch.backends.mps.is_available() else "cpu"  # Use Apple Silicon GPU if available

print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")

# Check if adapter path exists
if not os.path.exists(adapter_path):
    print(f"❌ Adapter path not found: {adapter_path}")
    print("Please update the adapter_path variable to point to your downloaded checkpoint")
    exit(1)

print(f"Loading base model: {model_name}")

# Load base model and tokenizer  
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Load model with appropriate settings for Mac
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    trust_remote_code=False,
    torch_dtype=torch.float32,  # Use float32 for CPU compatibility
    device_map=None  # Load to CPU first
).to(device)

# --- CRITICAL: Setup special tokens exactly as in training ---
print("Setting up cricket outcome tokens...")
EXTRA_BASE = 40
OUTCOMES = ["0", "1", "2", "3", "4", "6", "WICKET"]
OUTCOME2TOK = {o: f"<|extra_{EXTRA_BASE+i}|>" for i, o in enumerate(OUTCOMES)}
TOK2OUTCOME = {v: k for k, v in OUTCOME2TOK.items()}

# Add special tokens to tokenizer
special_tokens_to_add = list(OUTCOME2TOK.values())
tokenizer.add_special_tokens({'additional_special_tokens': special_tokens_to_add})

# Resize model embeddings to match training
print(f"Resizing vocabulary from {len(tokenizer) - len(special_tokens_to_add)} to {len(tokenizer)}")
model.resize_token_embeddings(len(tokenizer))

# Verify special tokens work
print("Verifying special tokens...")
for outcome, token in OUTCOME2TOK.items():
    ids = tokenizer(token)["input_ids"]
    assert len(ids) == 1, f"{token} splits into multiple tokens"
    print(f"  {outcome} -> {token} -> ID: {ids[0]}")

print("✅ Special tokens setup complete")

# NOW load the LoRA adapter
print(f"Loading LoRA adapter from: {adapter_path}")
model = PeftModel.from_pretrained(model, adapter_path)
print('✅ LoRA adapter loaded successfully!')

model.eval()

def predict_cricket_outcomes(prompt: str, model, tokenizer):
    """Generate probability distribution over cricket outcomes"""
    prompt_with_space = prompt.rstrip() + " "
    inputs = tokenizer(prompt_with_space, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        last_token_logits = logits[0, -1, :]
        probabilities = torch.softmax(last_token_logits, dim=-1)

        # Get probabilities for cricket outcome tokens
        outcome_token_ids = tokenizer.convert_tokens_to_ids(list(OUTCOME2TOK.values()))
        outcome_probs = probabilities[outcome_token_ids].cpu().tolist()
        
        # Create distribution
        dist = {outcome: prob for outcome, prob in zip(OUTCOMES, outcome_probs)}
        return dist, probabilities, last_token_logits

def show_top_k_tokens(prompt: str, model, tokenizer, k=30):
    """Show top-k token predictions"""
    prompt_with_space = prompt.rstrip() + " "
    inputs = tokenizer(prompt_with_space, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        last_token_logits = logits[0, -1, :]
        probabilities = torch.softmax(last_token_logits, dim=-1)
        
        top_k_probs, top_k_indices = torch.topk(probabilities, k)
        top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_indices)
        
        return top_k_tokens, top_k_probs

# --- Test Prompts ---
test_prompts = [
   "0.3: 0/0 Need191@9.7 | Recent: ------0-0 | P:0@2b(0.0rr) | Ahmed(Econ0.0) vs Arya(new,0 Runs @ 0 SR) | PP 0.3 | Chennai Super Kings vs Punjab Kings, MA Chidambaram",
    "12.1: 74/2 | Recent: 1-1-0-0-1 | P:23@20b(6.9rr) | Ashwin(Econ5.3,1W) vs Kohli(set,31 Runs @ 94 SR) | Chennai Super Kings vs Royal Challengers Bangalore, MA Chidambaram",
    "18.1: 178/5 | Recent: 6-0-W-4-1 | P:6@2b(18.0rr) | Chahal(Econ12.0) vs Dhoni(new,5 Runs @ 167 SR) | Death 18.1 | Chennai Super Kings vs Punjab Kings, MA Chidambaram",
    "5.3: 42/1 | Recent: 0-4-0-1-1 | P:33@22b(9.0rr) | Kaul(Econ6.0) vs Shaw(set,30 Runs @ 150 SR) | PP 5.3 | Delhi Daredevils vs Sunrisers Hyderabad, RGI"
]

print(f"\n{'='*80}")
print("🏏 CRICKET MODEL TESTING")
print(f"{'='*80}")

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n--- Test Prompt {i} ---")
    print(f"Context: {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
    
    # Get cricket outcome predictions
    cricket_dist, all_probs, logits = predict_cricket_outcomes(prompt, model, tokenizer)
    
    print("\n🎯 Cricket Outcome Predictions:")
    sorted_outcomes = sorted(cricket_dist.items(), key=lambda x: x[1], reverse=True)
    for outcome, prob in sorted_outcomes:
        print(f"  {outcome:7s}: {prob:.4f} {'🏆' if prob == max(cricket_dist.values()) else ''}")
    
    # Calculate confidence metrics
    max_prob = max(cricket_dist.values())
    entropy = -sum(p * torch.log2(torch.tensor(p + 1e-8)) for p in cricket_dist.values() if p > 0)
    print(f"  Confidence: {max_prob:.3f}, Entropy: {entropy:.3f}")
    
    # Show top general tokens too
    print("\n📊 Top 15 General Token Predictions:")
    top_tokens, top_probs = show_top_k_tokens(prompt, model, tokenizer, k=15)
    for token, prob in zip(top_tokens, top_probs):
        # Clean up token display
        if isinstance(token, bytes):
            token_str = token.decode('utf-8', errors='replace')
        else:
            token_str = token
        
        # Highlight if it's a cricket outcome token
        is_cricket = token_str in OUTCOME2TOK.values()
        marker = "🏏" if is_cricket else "  "
        print(f"  {marker} {token_str:<20} | {prob.item():.5f}")

print(f"\n{'='*80}")
print("✅ Testing complete!")
print(f"{'='*80}")

# Interactive mode
print("\n🎮 Interactive Mode - Enter your own prompts!")
print("Type 'quit' to exit")

while True:
    try:
        user_prompt = input("\nEnter cricket prompt: ").strip()
        if user_prompt.lower() in ['quit', 'exit', 'q']:
            break
        if not user_prompt:
            continue
            
        cricket_dist, _, _ = predict_cricket_outcomes(user_prompt, model, tokenizer)
        
        print("\n🏏 Predictions:")
        sorted_outcomes = sorted(cricket_dist.items(), key=lambda x: x[1], reverse=True)
        for outcome, prob in sorted_outcomes:
            print(f"  {outcome:7s}: {prob:.4f}")
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        break
    except Exception as e:
        print(f"Error: {e}")