import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import argparse, json, random, os, warnings, math
from pathlib import Path
import matplotlib.pyplot as plt

print(f"PyTorch version: {torch.__version__}")
model_name  = "Qwen/Qwen-1_8B"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True)
# Qwen tokenizer does not have a pad token, so we set it to the end-of-text token.
if tokenizer.pad_token is None:
    tokenizer.pad_token = '<|endoftext|>'

# Simplified CUDA-focused device handling
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
device_map = "auto" if torch.cuda.is_available() else None

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype = dtype,
    device_map = device_map,
    trust_remote_code = True
)

#use placeholder tokens for the outcomes
EXTRA_BASE = 40
OUTCOMES = ["0", "1", "2", "3", "4", "6", "WICKET"]

OUTCOME2TOK = {o:f"<|extra_{EXTRA_BASE+i}|>" for i,o in enumerate(OUTCOMES)}
TOK2OUTCOME = {v:k for k,v in OUTCOME2TOK.items()}

for k,v in OUTCOME2TOK.items():
    print(f" {k:7s} -> {v}")

#sanity check for tokenizer working
for tok in  OUTCOME2TOK.values():
    ids = tokenizer(tok)["input_ids"]
    #print(tok,ids)
    assert len(ids) == 1, f"{tok} splits into multiple tokens"

def build_dataset(jsonl_path: str, tokenizer, max_len: int =64):
    """
    Prepares the dataset for training with the critical loss masking logic.
    Labels for the prompt tokens are set to -100 to be ignored by the loss function.
    """
    raw_ds = load_dataset("json", data_files=jsonl_path, split = "train")
    def _to_features(example):
        tgt_tok = OUTCOME2TOK[example["target"]]
        prompt_text = example["prompt"].rstrip() + " "
        full_text= prompt_text + tgt_tok

        # Tokenize the full sequence
        enc = tokenizer(full_text, truncation= True, padding = "max_length", max_length = max_len)
        # --- Critical Change: Loss Masking ---
        # Tokenize prompt only to find its length (without special tokens)
        prompt_ids = tokenizer(prompt_text, add_special_tokens = True)["input_ids"]
        prompt_length = len(prompt_ids)

        # Initialize labels with -100 (the ignore index)
        labels = [-100]* max_len
    
        # Unmask the labels only for the target token
        # This loop starts right after the prompt
        for i in range(prompt_length, max_len):
            if enc["input_ids"][i] == tokenizer.pad_token_id:
                break
            labels[i] = enc["input_ids"][i]
        enc["labels"] = labels
        return enc
    return raw_ds.map(_to_features, remove_columns= raw_ds.column_names)



#training loop
def train_qwen (model, dataset, out_dir, epochs:int, batch: int, resume: bool):
    args = TrainingArguments(
        output_dir= out_dir,
        per_device_train_batch_size= batch,
        num_train_epochs= epochs,
        gradient_accumulation_steps = 4,
        learning_rate= 3e-4,
        fp16 = torch.cuda.is_available(),
        logging_steps= 25,
        save_strategy= "steps",
        save_steps = 500,
        save_total_limit = 3,
        report_to= "tensorboard",
        dataloader_num_workers=2
    )
    trainer = Trainer(model=model, args=args, train_dataset= dataset)
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
    ap.add_argument("--data", required = False)
    ap.add_argument("--epochs", type = int, default = 3)
    ap.add_argument("--batch", type=int, default = 2)
    ap.add_argument("--lora_r", type=int, default=16, help="LoRA rank (capacity).")
    ap.add_argument("--out", default="qwen-cricket-peft-2", help="Output directory for the model.")
    ap.add_argument("--resume", action='store_true', help="Resume training from the last checkpoint.")
    args = ap.parse_args()

    if not args.data:
        print("Data not provided")
        return

    # Setup LoRA configuration based on command-line arguments
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2, # Common practice to set alpha to 2*r
        target_modules="all-linear",
        lora_dropout=0.0,
        bias="none",
        modules_to_save=["lm_head"]
    )
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()

    ds = build_dataset(args.data, tokenizer)
    trainer = train_qwen(peft_model, ds, args.out, args.epochs, args.batch, args.resume)

     # --- Plot and save the loss curve after training ---
    print("Generating loss curve...")
    history = trainer.state.log_history
    
    training_loss = [log['loss'] for log in history if 'loss' in log]
    steps = [log['step'] for log in history if 'loss' in log]

    if not steps:
        print("No training logs found. Skipping loss curve generation.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(steps, training_loss, label="Training Loss")
    plt.title("Training Loss Curve")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    plot_path = Path(args.out) / "loss_curve.png"
    plt.savefig(plot_path)
    print(f"Loss curve saved to {plot_path}")

    # --- Demo prediction ---
    demo_prompt = "M Theekshana bowling to YBK Jaiswal, 7/0 after 1.4 overs, 1st innings, Pallekele"
    dist = predict_dist(demo_prompt, peft_model, tokenizer)
    print("\n--- Demo Prediction ---")
    for k, v in dist.items():
        print(f" {k:7s} : {v:.4f}")

if __name__ == "__main__":
    main()