#script to verify if masking prompt is working correctly
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
import random

# --- Step 1: Copy the necessary components from your training script ---

# This dictionary maps your plain-text outcomes to the special tokens
OUTCOME2TOK = {
    "0": "<|extra_40|>",
    "1": "<|extra_41|>",
    "2": "<|extra_42|>",
    "3": "<|extra_43|>",
    "4": "<|extra_44|>",
    "6": "<|extra_45|>",
    "WICKET": "<|extra_46|>"
}

# This is the new, corrected data processing function we want to test
def build_dataset(jsonl_path: str, tokenizer, max_len: int = 64):
    """
    Processes the JSONL dataset and applies the loss mask to the labels.
    """
    raw_ds = load_dataset("json", data_files=jsonl_path, split="train")

    def _to_features(example):
        tgt_tok = OUTCOME2TOK[example["target"]]
        prompt_text = example["prompt"].rstrip() + " "
        full_text = prompt_text + tgt_tok

        enc = tokenizer(full_text, truncation=True, padding="max_length", max_length=max_len)

        # --- The Critical Fix Logic ---
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        prompt_length = len(prompt_ids)
        labels = [-100] * max_len # Start with all labels ignored

        # Un-ignore ONLY the tokens that correspond to the target outcome
        for i in range(prompt_length, max_len):
            if enc["input_ids"][i] != tokenizer.pad_token_id:
                labels[i] = enc["input_ids"][i]
            else:
                break
        
        enc["labels"] = labels
        return enc

    return raw_ds.map(_to_features, remove_columns=raw_ds.column_names)


# --- Step 2: Main execution block to run the test ---

if __name__ == "__main__":
    # IMPORTANT: Replace this with the actual path to your data file
    jsonl_data_path = 'cricket_prompts.jsonl' #<--- CHANGE THIS PATH

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-1_8B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = '<|endoftext|>'
    # --- START OF THE FIX ---
    # The Qwen tokenizer doesn't have a default pad token, so we need to set one.
    # Using the end-of-sequence token as the pad token is a standard practice.
    if tokenizer.pad_token is None:
        print("Tokenizer does not have a pad_token, setting it to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    # --- END OF THE FIX ---

    print(f"Processing data from '{jsonl_data_path}'...")
    # We run the function we want to test
    processed_dataset = build_dataset(jsonl_data_path, tokenizer, max_len=64)

    # Grab the very first example from the processed dataset
    sample = processed_dataset[0]
    
    # ... (the rest of your script remains the same) ...
    tokens = tokenizer.convert_ids_to_tokens(sample['input_ids'])
    labels = sample['labels']

    print("\n--- SANITY CHECK: Inspecting a Single Processed Example ---")
    print("This table shows if the loss mask is working correctly.\n")
    print(f"{'TOKEN':<25} | {'LABEL'}")
    print("-" * 40)

    for token, label in zip(tokens, labels):
        # Handle byte tokens for clean printing
        if isinstance(token, bytes):
            token = token.decode('utf-8', 'replace')
        print(f"{token:<25} | {label}")

    print("\n--- VERIFICATION ---")
    print("Look at the table above. The fix is working if:")
    print("✅ 1. The 'LABEL' column is -100 for all tokens in the prompt.")
    print("✅ 2. The 'LABEL' column shows a real number (the token's ID) ONLY for the final outcome token.")
    print("✅ 3. The 'LABEL' column is -100 for all trailing padding tokens (which are now eos_tokens).")