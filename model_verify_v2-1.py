#script to see auto-regressive model output
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Setup ---
model_name = 'Qwen/Qwen-1_8B'
adapter_path = '/Users/aryamangupta/CricML/qwen-cricket-peft'
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Set the pad token if it's not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

print('✅ Model loaded successfully!')

demo = "M Theekshana bowling to YBK Jaiswal, 7/0 after 1.5 overs, 1st innings, Pallekele International Cricket Stadium, 2024-07-30"

# --- Autoregressive Generation ---

# Tokenize the input prompt
inputs = tokenizer(demo, return_tensors="pt").to(device)

print("\n--- Generating Output ---")

# Use the model's generate function to produce tokens until the EOS token is generated
# We use torch.no_grad() for efficiency as we aren't training
with torch.no_grad():
    output_ids = model.generate(
        inputs["input_ids"],
        max_new_tokens=50,  # A safety limit to prevent infinite loops
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False      # Use greedy decoding (always pick the most likely token)
    )

# Decode the generated token IDs back to a string
# skip_special_tokens=False will ensure we can see the EOS token if it's generated
full_output = tokenizer.decode(output_ids[0], skip_special_tokens=False)

# To see only the newly generated part, you can do this:
generated_part = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)


print(f"\nPrompt: '{demo}'")
print(f"\nGenerated Text: '{generated_part}'")
print(f"\nFull Output (including prompt and special tokens):")
print(full_output)