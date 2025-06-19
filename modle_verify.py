from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from qwenv1 import predict_dist

# Load base model
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen-1_8B', trust_remote_code= True)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen-1_8B', trust_remote_code= True)

# Load your trained LoRA adapter
model = PeftModel.from_pretrained(model, '/Users/aryamangupta/CricML/qwen-cricket-peft')
print('✅ Model loaded successfully!')
print(f'Model type: {type(model)}')
model.eval()
demo = "M Theekshana bowling to YBK Jaiswal, 7/0 after 1.5 overs, 1st innings, Pallekele International Cricket Stadium, 2024-07-30"
dist = predict_dist (demo,model, tokenizer)
for k,v in dist.items():
    print(f" {k:7s} : {v:.3f}")