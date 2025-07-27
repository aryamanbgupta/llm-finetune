import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import argparse, json, random, os, warnings, math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

print(f"PyTorch version: {torch.__version__}")
model_name  = "Qwen/Qwen1.5-1.8B"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = False)
# Qwen tokenizer does not have a pad token, so we set it to the end-of-text token.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
# Simplified CUDA-focused device handling
attn_implementation = "flash_attention_2" if torch.cuda.is_available() else "eager"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
device_map = "auto" if torch.cuda.is_available() else None

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype = dtype,
    device_map = device_map,
    trust_remote_code = False,
    attn_implementation = attn_implementation
)

#use placeholder tokens for the outcomes
EXTRA_BASE = 40
OUTCOMES = ["0", "1", "2", "3", "4", "6", "WICKET"]
OUTCOME_INDICES = {o: i for i, o in enumerate(OUTCOMES)}

OUTCOME2TOK = {o:f"<|extra_{EXTRA_BASE+i}|>" for i,o in enumerate(OUTCOMES)}
TOK2OUTCOME = {v:k for k,v in OUTCOME2TOK.items()}

print("Adding special tokens to the tokenizer...")
special_tokens_to_add = list(OUTCOME2TOK.values())
tokenizer.add_special_tokens({'additional_special_tokens': special_tokens_to_add})

# 2. Resize the model's token embeddings to include the new tokens
print(f"Resizing token embeddings from {len(tokenizer) - len(special_tokens_to_add)} to {len(tokenizer)}")
model.resize_token_embeddings(len(tokenizer))

#sanity check for tokenizer working
for tok in  OUTCOME2TOK.values():
    ids = tokenizer(tok)["input_ids"]
    #print(tok,ids)
    assert len(ids) == 1, f"{tok} splits into multiple tokens"

def build_dataset(jsonl_path: str, tokenizer, max_len: int =96, eval_split: float=0.1):
    """
    Prepares the dataset for training with the critical loss masking logic.
    Labels for the prompt tokens are set to -100 to be ignored by the loss function.
    If eval_split > 0, returns (train_dataset, eval_dataset).
    Otherwise, returns (full_dataset, None).
    """
    raw_ds = load_dataset("json", data_files=jsonl_path, split = "train")
    
    def _to_features(example):
        tgt_tok = OUTCOME2TOK[example["target"]]
        prompt_text = example["prompt"].rstrip() + " "
        full_text = prompt_text + tgt_tok

        # Tokenize the full sequence
        enc = tokenizer(full_text, truncation=True, padding="max_length", max_length=max_len)
        
        # Initialize labels with -100 (the ignore index)
        labels = [-100] * max_len
        
        # Find the outcome token in the input_ids
        target_token_id = tokenizer.convert_tokens_to_ids([tgt_tok])[0]
        
        target_found = False
        for idx in reversed(range(len(enc["input_ids"]))):
            if enc["input_ids"][idx] == target_token_id and enc["attention_mask"][idx] == 1:
                labels[idx] = target_token_id  # Fixed: was token_id in their example
                target_found = True
                break

        if not target_found:
            raise ValueError(f"Target token {tgt_tok} not found in expected position")

        
        enc["labels"] = labels
        return enc

    processed_ds = raw_ds.map(_to_features, remove_columns=raw_ds.column_names)
    if eval_split > 0:
        split_ds = processed_ds.train_test_split(test_size=eval_split, seed=42)
        print(f"Dataset split into {len(split_ds['train'])} training and {len(split_ds['test'])} evaluation examples.")
        return split_ds["train"], split_ds["test"]
    else:
        print(f"Using full dataset with {len(processed_ds)} examples for training.")
        return processed_ds, None

#training loop
def train_qwen (model, train_dataset, eval_dataset, out_dir, epochs:int, batch: int, resume: bool, grad_accum_steps: int, grad_checkpoint: bool, learning_rate: float):
    args = TrainingArguments(
        output_dir= out_dir,
        per_device_train_batch_size= batch,
        per_device_eval_batch_size = batch,
        num_train_epochs= epochs,
        gradient_accumulation_steps = grad_accum_steps,
        learning_rate= learning_rate,
        lr_scheduler_type = "cosine",
        warmup_ratio = 0.05,
        weight_decay = 0.01,
        bf16 = torch.cuda.is_available(),
        eval_strategy = "steps" if eval_dataset else "no",
        eval_steps = 500,
        logging_steps= 25,
        save_strategy= "steps",
        save_steps = 500,
        load_best_model_at_end = False,  # Disable since eval_loss isn't available
        metric_for_best_model = None,
        save_total_limit = 3,
        report_to= "tensorboard",
        dataloader_num_workers= max(1, (os.cpu_count() or 8) - 4),
        dataloader_pin_memory = True,
        dataloader_persistent_workers = True,
        seed = 42,
        gradient_checkpointing = grad_checkpoint,
        optim = "adamw_torch_fused",
        remove_unused_columns= False
    )
    trainer = Trainer(
        model=model, 
        args=args, 
        train_dataset= train_dataset, 
        eval_dataset = eval_dataset)

    trainer.train(resume_from_checkpoint = resume) 
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    return trainer

def predict_dist(prompt:str, model, tokenizer):
    #Generates a probability distribution over the possible outcomes for a given prompt.
    model.eval()
    # Add the trailing space to the prompt to predict the *next* token
    prompt_with_space = prompt.rstrip() + " "
    inputs = tokenizer(prompt_with_space, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        # Get the logits for the very last position in the sequence
        last_token_logits = logits[0, -1, :]
        # Convert all logits to probabilities
        probabilities = torch.softmax(last_token_logits, dim=-1)

        # Get the token IDs for our specific outcome tokens
        outcome_token_ids = tokenizer.convert_tokens_to_ids(list(OUTCOME2TOK.values()))

        # Extract the probabilities for just those tokens
        outcome_probs = probabilities[outcome_token_ids].cpu().tolist()

        # Create a clean dictionary of the results
        dist = {outcome: prob for outcome, prob in zip(OUTCOMES, outcome_probs)}
        return dist

   

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required = True)
    ap.add_argument("--epochs", type = int, default = 3)
    ap.add_argument("--batch", type=int, default = 64)
    ap.add_argument("--lora_r", type=int, default=16, help="LoRA rank (capacity).")
    ap.add_argument("--out", default="qwen-cricket-peft-2", help="Output directory for the model.")
    ap.add_argument("--resume", action='store_true', help="Resume training from the last checkpoint.")
    ap.add_argument("--eval_split", type=float, default=0.1, help="Fraction of data for evaluation. Set to 0 to disable.")
    ap.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps (increase if low GPU memory).")
    ap.add_argument("--grad_checkpoint", action='store_true', help="Enable gradient checkpointing (saves memory, slower training).")
    ap.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate.")
    ap.add_argument("--max_len", type=int, default=96, help="Maximum sequence length (reduce if GPU memory limited).")

    args = ap.parse_args()

    # Setup LoRA configuration based on command-line arguments
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2, # Common practice to set alpha to 2*r
        target_modules="all-linear",
        lora_dropout=0.0,
        bias="none",
        modules_to_save=["lm_head", "model.embed_tokens"]
    )
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()

    # if torch.cuda.is_available():
    #     try:
    #         print("Compiling model with torch.compile(mode='reduce-overhead')...")
    #         peft_model = torch.compile(peft_model, mode="reduce-overhead")
    #     except Exception as e:
    #         print(f"torch.compile failed with exception: {e}")
    #         print("Continuing with the un-compiled model.")

    train_dataset, eval_dataset = build_dataset(args.data, tokenizer, max_len = args.max_len, eval_split = args.eval_split)
    
    # Debug: Check if any labels are not -100
    if eval_dataset:
        print("\nChecking evaluation dataset labels...")
        num_valid_labels = 0
        for i in range(min(10, len(eval_dataset))):  # Check first 10 examples
            labels = eval_dataset[i]['labels']
            valid_positions = [j for j, label in enumerate(labels) if label != -100]
            if valid_positions:
                num_valid_labels += 1
                label_id = labels[valid_positions[0]]
                # Convert label ID to token
                token = tokenizer.convert_ids_to_tokens([label_id])[0]
                # Check if it's one of our outcome tokens
                is_outcome = token in OUTCOME2TOK.values()
                outcome = TOK2OUTCOME.get(token, "UNKNOWN")
                
                print(f"Example {i}: Valid label at position {valid_positions[0]}, label_id={label_id}, token='{token}', outcome='{outcome}', is_outcome_token={is_outcome}")
            else:
                print(f"Example {i}: No valid labels (all -100)")
        print(f"Total examples with valid labels: {num_valid_labels}/10")
        
        # Also print our expected outcome tokens and their IDs
        print("\nExpected outcome tokens and their IDs:")
        for outcome, token in OUTCOME2TOK.items():
            token_id = tokenizer.convert_tokens_to_ids([token])[0]
            print(f"  {outcome} -> {token} -> ID: {token_id}")
    
    trainer = train_qwen(peft_model, train_dataset, eval_dataset, args.out, args.epochs, args.batch, args.resume, args.grad_accum_steps, args.grad_checkpoint, args.learning_rate)

     # --- Plot and save the loss curve after training ---
    print("Generating loss curve...")
    history = trainer.state.log_history
    
    training_loss = [log['loss'] for log in history if 'loss' in log]
    eval_loss = [log['eval_loss'] for log in history if 'eval_loss' in log]
    train_steps = [log['step'] for log in history if 'loss' in log]
    eval_steps = [log['step'] for log in history if 'eval_loss' in log]

    if not train_steps:
        print("No training logs found. Skipping loss curve generation.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, training_loss, label="Training Loss")
    if eval_steps:
        plt.plot(eval_steps, eval_loss, label = "Evaluation Loss")
    plt.title("Training & Eval Loss Curve")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    plot_path = Path(args.out) / "loss_curve.png"
    plt.savefig(plot_path)
    print(f"Loss curve saved to {plot_path}")

    # --- Demo prediction ---
    demo_prompt = "2.4: 12/1 | Recent: 4-W-1-0-0 | P:1@3b(2.0rr) | Wickramasinghe(Econ2.7) vs Samson(new,0 Runs @ 0 SR) | PP 2.4 | India vs Sri Lanka, PIC"
    dist = predict_dist(demo_prompt, peft_model, tokenizer)
    print("\n--- Demo Prediction ---")
    for k, v in dist.items():
        print(f" {k:7s} : {v:.4f}")
    demo_prompt = "2.5: 12/1 | Recent: W-1-0-0-0 | P:1@4b(1.5rr) | Wickramasinghe(Econ2.4) vs Samson(new,0 Runs @ 0 SR) | PP 2.5 | India vs Sri Lanka, PIC"
    dist = predict_dist(demo_prompt, peft_model, tokenizer)
    print("\n--- Demo Prediction ---")
    for k, v in dist.items():
        print(f" {k:7s} : {v:.4f}")

if __name__ == "__main__":
    main()