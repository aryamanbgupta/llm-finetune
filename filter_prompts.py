#!/usr/bin/env python3
"""
Script to filter JSONL files by removing lines where the tokenized sequence exceeds max_len.
Usage: python filter_jsonl.py input.jsonl output.jsonl [--max_len 128]
"""

import json
import argparse
from transformers import AutoTokenizer

# Same configuration as your main script
model_name = "Qwen/Qwen1.5-1.8B"
EXTRA_BASE = 40
OUTCOMES = ["0", "1", "2", "3", "4", "6", "WICKET"]
OUTCOME2TOK = {o: f"<|extra_{EXTRA_BASE+i}|>" for i, o in enumerate(OUTCOMES)}

def setup_tokenizer():
    """Setup tokenizer with special tokens, same as main script"""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    special_tokens_to_add = list(OUTCOME2TOK.values())
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens_to_add})
    
    return tokenizer

def get_sequence_length(example, tokenizer):
    """Get the tokenized length of prompt + target, same logic as main script"""
    tgt_tok = OUTCOME2TOK[example["target"]]
    prompt_text = example["prompt"].rstrip() + " "
    full_text = prompt_text + tgt_tok
    
    enc = tokenizer(full_text, add_special_tokens=True)
    return len(enc["input_ids"])

def filter_jsonl(input_path, output_path, max_len):
    """Filter JSONL file by sequence length"""
    tokenizer = setup_tokenizer()
    
    kept_count = 0
    removed_count = 0
    
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line_num, line in enumerate(infile, 1):
            try:
                example = json.loads(line.strip())
                seq_len = get_sequence_length(example, tokenizer)
                
                if seq_len <= max_len:
                    outfile.write(line)
                    kept_count += 1
                else:
                    removed_count += 1
                    
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                removed_count += 1
    
    print(f"Filtering complete:")
    print(f"  Kept: {kept_count} lines")
    print(f"  Removed: {removed_count} lines (> {max_len} tokens)")
    print(f"  Output saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Filter JSONL file by sequence length")
    parser.add_argument("input", help="Input JSONL file path")
    parser.add_argument("output", help="Output JSONL file path")
    parser.add_argument("--max_len", type=int, default=128, help="Maximum sequence length (default: 128)")
    
    args = parser.parse_args()
    
    filter_jsonl(args.input, args.output, args.max_len)

if __name__ == "__main__":
    main()