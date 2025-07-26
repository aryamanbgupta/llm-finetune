import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import argparse, json, random, os, warnings, math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

os.environ["TOKENIZERS_PARALLELISM"]= "false"
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
        full_text= prompt_text + tgt_tok

        # Tokenize the full sequence
        enc = tokenizer(full_text, truncation= True, padding = "max_length", max_length = max_len)
        # Tokenize prompt only to find its length (without special tokens)
        prompt_ids = tokenizer(prompt_text, add_special_tokens = True)["input_ids"]
        prompt_length = len(prompt_ids)

        # Initialize labels with -100 (the ignore index)
        labels = [-100]* max_len
    
        # Unmask the labels only for the target token
        if prompt_length < max_len:
            labels[prompt_length]= enc["input_ids"][prompt_length]
        enc["labels"] = labels
        return enc

    processed_ds = raw_ds.map(_to_features, remove_columns= raw_ds.column_names)
    if eval_split>0:
        split_ds = processed_ds.train_test_split(test_size=eval_split, seed=42)
        print(f"Dataset split into {len(split_ds['train'])} training and {len(split_ds['test'])} evaluation examples.")
        return split_ds["train"], split_ds["test"]
    else:
        print(f"Using full dataset with {len(processed_ds)} examples for training.")
        return processed_ds, None

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    
    outcome_token_ids = tokenizer.convert_tokens_to_ids(list(OUTCOME2TOK.values()))
    true_labels = []
    pred_probs = []

    for i in range(len(labels)):
        valid_indices = np.where(labels[i]!= -100)[0]
        if len(valid_indices)> 0:
            idx = valid_indices[0]

            outcome_logits = logits[i, idx, outcome_token_ids]
            probs = torch.softmax(torch.tensor(outcome_logits), dim=-1).numpy()
            pred_probs.append(probs)

            true_token_id = labels[i][idx]
            true_token = tokenizer.convert_ids_to_tokens([true_token_id])[0]
            true_outcome = TOK2OUTCOME.get(true_token,"0")
            true_labels.append(OUTCOME_INDICES[true_outcome])
    #compute metrics
    pred_probs = np.array(pred_probs) #(n_samples, 7)
    accuracy = accuracy_score(true_labels, np.argmax(pred_probs, axis=1))
    cross_entropy = log_loss(true_labels, pred_probs, labels = np.arange(len(OUTCOMES)))


    return {
        "accuracy": accuracy,
        "cross_entropy": cross_entropy,
        "neg_cross_entropy": -cross_entropy
    }

#training loop
def train_qwen (model, train_dataset, eval_dataset, out_dir, epochs:int, batch: int, resume: bool, grad_accum_steps: int, grad_checkpoint: bool, learning_rate: float):
    callbacks = []
    if eval_dataset:
        from transformers import EarlyStoppingCallback
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=5))

    args = TrainingArguments(
        output_dir= out_dir,
        per_device_train_batch_size= batch,
        per_device_eval_batch_size = int(batch/4),
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
        load_best_model_at_end = False,
        metric_for_best_model =  None,
        greater_is_better= True,
        save_total_limit = 3,
        report_to= "tensorboard",
        dataloader_num_workers= os.cpu_count()-4 or 4,
        dataloader_pin_memory = True,
        dataloader_persistent_workers = False,
        seed = 42,
        gradient_checkpointing = grad_checkpoint,
        optim = "adamw_torch_fused",
        remove_unused_columns=False,
        dataloader_drop_last = True,
    )
    trainer = Trainer(
        model=model, 
        args=args, 
        train_dataset= train_dataset, 
        eval_dataset = eval_dataset, 
        compute_metrics = compute_metrics,
        callbacks = callbacks)

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
        modules_to_save=["lm_head", "embed_tokens"]
    )
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()

    if torch.cuda.is_available():
        try:
            print("Compiling model with torch.compile(mode='max-autotune')...")
            peft_model = torch.compile(peft_model, mode="default")
        except Exception as e:
            print(f"torch.compile failed with exception: {e}")
            print("Continuing with the un-compiled model.")

    train_dataset, eval_dataset = build_dataset(args.data, tokenizer, max_len = args.max_len, eval_split = args.eval_split)
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