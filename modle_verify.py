import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 1. Setup and Configuration ---
model_name = 'Qwen/Qwen-1_8B'
# NOTE: Ensure this path points to your fine-tuned LoRA adapter
adapter_path = '/Users/aryamangupta/CricML/qwen-cricket-peft'
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the exact same special tokens used during training
EXTRA_BASE = 40
OUTCOMES = ["0", "1", "2", "3", "4", "6", "WICKET"]
outcome_tokens = [f"<|extra_{EXTRA_BASE+i}|>" for i, o in enumerate(OUTCOMES)]

print(f"Using device: {device}")

# --- 2. Load Base Model and Tokenizer ---
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True
).to(device)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

# --- 3. CRITICAL FIX: Synchronize Vocabulary and Model Size ---
# The model was trained with a resized vocabulary. We MUST replicate
# that exact change here BEFORE loading the LoRA adapter.

print("Resizing vocabulary to match training setup...")

# Set the pad token - using the end-of-sequence token is standard
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Add the special outcome tokens to the tokenizer's vocabulary
tokenizer.add_special_tokens({"additional_special_tokens": outcome_tokens})

# Resize the model's token embedding layer to accommodate the new tokens
model.resize_token_embeddings(len(tokenizer))

print(f"✅ Vocabulary resized. New size: {len(tokenizer)}")

# --- 4. Load the Fine-Tuned LoRA Adapter ---
# Now that the base model's architecture matches the trained model, it's safe to load.
model = PeftModel.from_pretrained(model, adapter_path)
model.eval() # Set the model to evaluation mode

print('✅ LoRA adapter loaded successfully!')
print(f'Model type: {type(model)}')


# --- 5. Perform Prediction ---
demo = "M Theekshana bowling to YBK Jaiswal, 7/0 after 1.5 overs, 1st innings, Pallekele International Cricket Stadium, 2024-07-30"

# Add the trailing space to the prompt to predict the *next* token
prompt_with_space = demo.rstrip() + " "
inputs = tokenizer(prompt_with_space, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    # Get the logits for the very last position in the sequence
    last_token_logits = logits[0, -1, :]
    # Convert logits to probabilities
    probabilities = torch.softmax(last_token_logits, dim=-1)

    # Get the token IDs for our specific outcome tokens
    outcome_token_ids = tokenizer.convert_tokens_to_ids(outcome_tokens)

    # Extract the probabilities for just those tokens
    outcome_probs = probabilities[outcome_token_ids].cpu().tolist()

    # Create a clean dictionary of the results
    dist = {outcome: prob for outcome, prob in zip(OUTCOMES, outcome_probs)}


# --- 6. Print Results ---
print("\n" + "="*50)
print(f"Prediction for prompt: \n'{demo}'")
print("="*50)
for k, v in dist.items():
    print(f" {k:7s} : {v:.4f}")
print("="*50)