#!/usr/bin/env python3
"""
Quick script to verify label masking is working correctly
"""
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
import json

# Constants from your main script
EXTRA_BASE = 40
OUTCOMES = ["0", "1", "2", "3", "4", "6", "WICKET"]
OUTCOME2TOK = {o: f"<|extra_{EXTRA_BASE+i}|>" for i, o in enumerate(OUTCOMES)}
TOK2OUTCOME = {v: k for k, v in OUTCOME2TOK.items()}

def setup_tokenizer():
    """Setup tokenizer same as main script"""
    model_name = "Qwen/Qwen1.5-1.8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    special_tokens_to_add = list(OUTCOME2TOK.values())
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens_to_add})
    
    return tokenizer

def original_masking(example, tokenizer, max_len=96):
    """Original masking logic (potentially buggy)"""
    tgt_tok = OUTCOME2TOK[example["target"]]
    prompt_text = example["prompt"].rstrip() + " "
    full_text = prompt_text + tgt_tok

    enc = tokenizer(full_text, truncation=True, padding="max_length", max_length=max_len)
    labels = [-100] * max_len
    target_token_id = tokenizer.convert_tokens_to_ids([tgt_tok])[0]
    
    # Original logic - find first occurrence
    for idx, token_id in enumerate(enc["input_ids"]):
        if token_id == target_token_id:
            labels[idx] = token_id
            break
    
    enc["labels"] = labels
    return enc

def fixed_masking(example, tokenizer, max_len=96):
    """Fixed masking logic (search from right)"""
    tgt_tok = OUTCOME2TOK[example["target"]]
    prompt_text = example["prompt"].rstrip() + " "
    full_text = prompt_text + tgt_tok

    enc = tokenizer(full_text, truncation=True, padding="max_length", max_length=max_len)
    labels = [-100] * max_len
    target_token_id = tokenizer.convert_tokens_to_ids([tgt_tok])[0]
    
    # Fixed logic - find from right with attention mask check
    target_found = False
    for idx in reversed(range(len(enc["input_ids"]))):
        if enc["input_ids"][idx] == target_token_id and enc["attention_mask"][idx] == 1:
            labels[idx] = target_token_id
            target_found = True
            break
    
    if not target_found:
        raise ValueError(f"Target token {tgt_tok} not found in expected position")
    
    enc["labels"] = labels
    return enc

def analyze_example(example, tokenizer, max_len=96):
    """Analyze a single example with both masking approaches"""
    print(f"\n{'='*80}")
    print(f"TARGET: {example['target']}")
    print(f"PROMPT: {example['prompt']}")
    print(f"TARGET TOKEN: {OUTCOME2TOK[example['target']]}")
    
    target_token_id = tokenizer.convert_tokens_to_ids([OUTCOME2TOK[example['target']]])[0]
    print(f"TARGET TOKEN ID: {target_token_id}")
    
    # Test both approaches
    try:
        orig_enc = original_masking(example, tokenizer, max_len)
        print(f"\n--- ORIGINAL MASKING ---")
        analyze_encoding(orig_enc, tokenizer, target_token_id)
    except Exception as e:
        print(f"\n--- ORIGINAL MASKING FAILED ---")
        print(f"Error: {e}")
    
    try:
        fixed_enc = fixed_masking(example, tokenizer, max_len)
        print(f"\n--- FIXED MASKING ---")
        analyze_encoding(fixed_enc, tokenizer, target_token_id)
    except Exception as e:
        print(f"\n--- FIXED MASKING FAILED ---")
        print(f"Error: {e}")

def analyze_encoding(enc, tokenizer, expected_target_id):
    """Analyze the encoding and labels"""
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    labels = enc["labels"]
    
    # Find non-padding positions
    non_pad_positions = [i for i, mask in enumerate(attention_mask) if mask == 1]
    print(f"Non-padding positions: {len(non_pad_positions)} tokens")
    
    # Find label positions (non -100)
    label_positions = [i for i, label in enumerate(labels) if label != -100]
    print(f"Label positions: {label_positions}")
    
    # Find all occurrences of target token
    target_occurrences = [i for i, token_id in enumerate(input_ids) if token_id == expected_target_id]
    print(f"Target token occurrences: {target_occurrences}")
    
    # Show the relevant tokens around label positions
    for pos in label_positions:
        token = tokenizer.convert_ids_to_tokens([input_ids[pos]])[0]
        label_id = labels[pos]
        is_correct = label_id == expected_target_id
        in_attention = attention_mask[pos] == 1
        
        print(f"  Position {pos}: token='{token}', label_id={label_id}, correct={is_correct}, in_attention={in_attention}")
    
    # Show the last few non-padding tokens
    if non_pad_positions:
        print(f"\nLast 5 non-padding tokens:")
        start_idx = max(0, len(non_pad_positions) - 5)
        for i in range(start_idx, len(non_pad_positions)):
            pos = non_pad_positions[i]
            token = tokenizer.convert_ids_to_tokens([input_ids[pos]])[0]
            has_label = labels[pos] != -100
            print(f"  Position {pos}: '{token}' (labeled: {has_label})")

def main():
    # Setup
    tokenizer = setup_tokenizer()
    
    # Load a few examples from your dataset
    # Update this path to your actual data file
    data_file = input("Enter path to your JSONL data file: ").strip()
    
    try:
        raw_ds = load_dataset("json", data_files=data_file, split="train")
        print(f"Loaded {len(raw_ds)} examples")
        
        # Test first few examples
        num_examples = min(3, len(raw_ds))
        print(f"\nTesting first {num_examples} examples...")
        
        for i in range(num_examples):
            analyze_example(raw_ds[i], tokenizer)
            
        print(f"\n{'='*80}")
        print("SUMMARY:")
        print("1. Check that 'Label positions' contains exactly one position")
        print("2. Check that this position corresponds to the target token")
        print("3. Check that the label is at the END of non-padding content")
        print("4. Compare original vs fixed - they should be the same if no conflicts")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure the file path is correct and the file exists")

if __name__ == "__main__":
    main()