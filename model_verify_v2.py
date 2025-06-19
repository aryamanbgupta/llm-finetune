# script to get logits over top k tokens after prompt
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Setup ---
model_name = 'Qwen/Qwen-1_8B'
adapter_path = '/Users/aryamangupta/CricML/qwen-cricket-peft'
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

# Load base model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


# --- START OF THE CRITICAL FIX ---
# The training script effectively used a larger vocabulary. We MUST replicate
# that exact change here BEFORE loading the LoRA adapter.

print("Resizing vocabulary to match training setup...")
# Define the exact same special tokens used during training
EXTRA_BASE = 40
OUTCOMES = ["0", "1", "2", "3", "4", "6", "WICKET"]
outcome_tokens = [f"<|extra_{EXTRA_BASE+i}|>" for i, o in enumerate(OUTCOMES)]

# Add the special tokens to the tokenizer
tokenizer.add_special_tokens({"additional_special_tokens": outcome_tokens})

# Resize the model's token embedding layer to accommodate the new tokens
model.resize_token_embeddings(len(tokenizer))
print(f"✅ Vocabulary resized. New size: {len(tokenizer)}")
# --- END OF THE CRITICAL FIX ---


# NOW, with the architectures matched, it is safe to load the LoRA adapter
model = PeftModel.from_pretrained(model, adapter_path)
print('✅ LoRA adapter loaded successfully!')
print(f'Model type: {type(model)}')

model.eval()

# --- The rest of your script remains the same ---
demo = "M Theekshana bowling to YBK Jaiswal, 7/0 after 1.5 overs, 1st innings, Pallekele International Cricket Stadium, 2024-07-30"
prompt_with_space = demo.rstrip() + " "
inputs = tokenizer(prompt_with_space, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

last_token_logits = logits[0, -1, :]
probabilities = torch.softmax(last_token_logits, dim=-1)
top_k_probs, top_k_indices = torch.topk(probabilities, 50)

# This line should no longer cause an error
top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_indices)

# (Printing logic follows)
print(f"\nTop 50 next token predictions for the prompt:\n'{demo}'")
print("-" * 50)
print(f"{'Token':<25} | {'Probability':<20}")
print("-" * 50)
for token, prob in zip(top_k_tokens, top_k_probs):
    if isinstance(token, bytes):
        token_str = token.decode('utf-8', errors='replace')
    else:
        token_str = token
    print(f"{token_str:<25} | {prob.item():.5f}")