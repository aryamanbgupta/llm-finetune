import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import argparse, json, random, os, warnings, math
from pathlib import Path

model_name  = "Qwen/Qwen1.5-1_8B"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = '<|endoftext|>'

if torch.backends.mps.is_available():
    model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype = torch.float16,
    device_map = {"":"mps"},
    trust_remote_code = True
)

else:

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype = dtype,
        device_map = "auto",
        trust_remote_code = True
    )
'''
new_tokens = ["<O_0>", "<O_1>", "<O_2>", "<O_3>", "<O_4>",  "<O_6>",  "<O_W>"]
num_added = tokenizer._tokenize.add_tokens(new_tokens)
print(num_added)
tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
model.resize_token_embeddings(len(tokenizer))
print("Added tokens:", new_tokens)

ids = tokenizer.convert_tokens_to_ids(new_tokens)
if all(i != tokenizer.unk_token_id for i in ids):
    print("added successfully")
else:
    print("failed")

for token in new_tokens:
    if token not in tokenizer.get_vocab():
        print(f"Warning: Token {token} may not have been added properly")
'''
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

    raw_ds = load_dataset("json", data_files=jsonl_path, split = "train")
    def _to_features(example):
        tgt_tok = OUTCOME2TOK[example["target"]]
        text = example["prompt"].rstrip() + " " + tgt_tok
        enc = tokenizer(text, truncation= True, padding = "max_length", max_length = max_len)
        # Add labels for training (same as input_ids, but mask padding tokens)
        enc["labels"] = [
            token_id if token_id != tokenizer.pad_token_id else -100 
            for token_id in enc["input_ids"]
        ]
        return enc
    return raw_ds.map(_to_features, remove_columns= raw_ds.column_names)

# ds = build_dataset('/Users/aryamangupta/CricML/llm-finetune/cricket_prompts.jsonl', tokenizer, max_len=128)
# example = ds[0]
# print(example["input_ids"])
# print("attention mask:", example["attention_mask"])
# print ("labels", example["labels"])

#setup LoRA
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules= "all-linear", #["c_attn", "c_proj", "w1", "w2"],
    lora_dropout=0.0,
    bias="none"
)
model = get_peft_model(model, peft_config)

print(f"Model type: {type(model).__name__}")
model.print_trainable_parameters()

#training loop
def train_qwen (model, dataset, out_dir, epochs:int, batch: int):
    args = TrainingArguments(
        output_dir= out_dir,
        per_device_train_batch_size= batch,
        num_train_epochs= epochs,
        gradient_accumulation_steps = 4,
        learning_rate= 3e-4,
        fp16 = torch.cuda.is_available(),
        logging_steps= 25,
        save_strategy= "epoch",
        report_to= "none",
        dataloader_num_workers=2
    )
    trainer = Trainer(model=model, args=args, train_dataset= dataset)
    trainer.train() 
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    return trainer

def predict_dist(prompt:str, model, tokenizer):
    model.eval()
    ids = tokenizer(prompt, return_tensors = "pt").to(model.device)
    with torch.no_grad():
        logits = model(**ids).logits[0, -1, :]
    sel = tokenizer.convert_tokens_to_ids(list(OUTCOME2TOK.values()))
    probs = torch.softmax(logits[sel], dim =-1).cpu().tolist()
    return {o: p for o,p in zip(OUTCOMES, probs)}

#test
# input_ids = tokenizer("Vaibhav Arora bowling to Tristan Stubbs", return_tensors = "pt").input_ids.to(model.device)
# output = model.generate(input_ids)
# print (tokenizer.decode(output[0], skip_special_tokens = True))    

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required = False)
    ap.add_argument("--epochs", type = int, default = 3)
    ap.add_argument("--batch", type=int, default = 2)
    ap.add_argument("--out", default = "qwen-cricket-peft")
    args = ap.parse_args()

    if not args.data:
        print("Data not provided")
        return

    ds = build_dataset(args.data, tokenizer)
    qwen_trainer = train_qwen(model, ds, args.out, args.epochs, args.batch)

    #demo
    demo = "M Theekshana bowling to YBK Jaiswal, 7/0 after 1.4 overs, 1st innings, Pallekele International Cricket Stadium, 2024-07-30"
    dist = predict_dist (demo,model, tokenizer)
    for k,v in dist.items():
        print(f" {k:7s} : {v:.3f}")

if __name__ == "__main__":
    main()