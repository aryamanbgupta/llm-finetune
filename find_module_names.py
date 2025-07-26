import torch
from transformers import AutoModelForCausalLM

# 1. Set your model's repository ID here
# The model from your traceback appears to be a Qwen model.
# Using "Qwen/Qwen1.5-1.8B" as an example.
MODEL_ID = "Qwen/Qwen1.5-1.8B"

print(f"Loading model '{MODEL_ID}' to inspect its layers...\n")

# 2. Load the model
# Using low_cpu_mem_usage to avoid high RAM consumption
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, low_cpu_mem_usage=True)

# 3. Print all parameter names
print("--- All Model Parameter Names ---")
for name, _ in model.named_parameters():
    print(name)
print("---------------------------------")

# 3. Find all modules that are instances of torch.nn.Linear
linear_layer_names = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        linear_layer_names.append(name)

# 4. Print the results
print("--- Found Linear Layers ---")
if linear_layer_names:
    for name in linear_layer_names:
        print(name)

    # Print in a copy-paste friendly format for your LoraConfig
    print("\n--- Python list for LoraConfig ---")
    print(f"target_modules={linear_layer_names}")
else:
    print("No linear layers found with these names.")

print("----------------------------")
