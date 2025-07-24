import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import argparse, json, random, os, warnings, math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score


print(f"PyTorch version: {torch.__version__}")

# Your existing code to load model and tokenizer
model_name  = "Qwen/Qwen1.5-1.8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print(tokenizer.pad_token)
print (tokenizer.eos_token)
# Your existing outcome mapping
EXTRA_BASE = 40
OUTCOMES = ["0", "1", "2", "3", "4", "6", "WICKET"]
OUTCOME2TOK = {o:f"<|extra_{EXTRA_BASE+i}|>" for i,o in enumerate(OUTCOMES)}

# --- FIX STARTS HERE ---

# 1. Add your custom tokens to the Qwen1.5 tokenizer vocabulary
print("Adding special tokens to the tokenizer...")
special_tokens_to_add = list(OUTCOME2TOK.values())
tokenizer.add_special_tokens({'additional_special_tokens': special_tokens_to_add})

# 2. Resize the model's token embeddings to include the new tokens
print(f"Resizing token embeddings from {len(tokenizer) - len(special_tokens_to_add)} to {len(tokenizer)}")
model.resize_token_embeddings(len(tokenizer))

# --- FIX ENDS HERE ---

# The sanity check will now pass
print("Running sanity check on new tokens...")
for tok in OUTCOME2TOK.values():
    ids = tokenizer(tok)["input_ids"]
    assert len(ids) == 1, f"{tok} splits into multiple tokens"
print("Sanity check passed!")