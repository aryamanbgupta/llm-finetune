import json
import torch
from transformers import AutoTokenizer

def find_max_seq_length(jsonl_path, model_name="Qwen/Qwen-1_8B"):
    # Check device availability
    if torch.backends.mps.is_available():
        device = "mps"
        print("🚀 Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = "cuda"
        print("🚀 Using CUDA GPU")
    else:
        device = "cpu"
        print("💻 Using CPU")
    
    print(f"Loading tokenizer: {model_name}...")
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("✅ Tokenizer loaded")
    
    # Define outcome tokens
    EXTRA_BASE = 40
    OUTCOMES = ["0", "1", "2", "3", "4", "6", "WICKET"]
    OUTCOME2TOK = {o: f"<|extra_{EXTRA_BASE+i}|>" for i, o in enumerate(OUTCOMES)}
    print(f"📝 Outcome tokens defined: {list(OUTCOME2TOK.values())}")
    
    print(f"📖 Reading file: {jsonl_path}")
    lengths = []
    max_length = 0
    max_example = ""
    
    with open(jsonl_path, 'r') as f:
        lines = f.readlines()
        total_lines = len(lines)
        print(f"📊 Found {total_lines} examples to process")
        
        for line_num, line in enumerate(lines, 1):
            # Progress indicator
            if line_num % 1000 == 0 or line_num == total_lines:
                print(f"⏳ Processing {line_num}/{total_lines} ({line_num/total_lines*100:.1f}%)")
            
            data = json.loads(line.strip())
            
            # Reconstruct the full text as it would be in training
            tgt_tok = OUTCOME2TOK[data["target"]]
            prompt_text = data["prompt"].rstrip() + " "
            full_text = prompt_text + tgt_tok
            
            # Tokenize
            tokens = tokenizer(full_text)["input_ids"]
            length = len(tokens)
            lengths.append(length)
            
            if length > max_length:
                max_length = length
                max_example = full_text
                print(f"🔥 New max length found: {max_length} tokens (line {line_num})")
    
    print("✅ Processing complete! Calculating statistics...")
    
    # Statistics
    avg_length = sum(lengths) / len(lengths)
    sorted_lengths = sorted(lengths)
    p95_length = sorted_lengths[int(len(lengths) * 0.95)]
    p99_length = sorted_lengths[int(len(lengths) * 0.99)]
    
    print("\n" + "="*50)
    print("📊 SEQUENCE LENGTH ANALYSIS")
    print("="*50)
    print(f"Total examples: {len(lengths):,}")
    print(f"Max length: {max_length}")
    print(f"Average length: {avg_length:.1f}")
    print(f"95th percentile: {p95_length}")
    print(f"99th percentile: {p99_length}")
    print(f"Min length: {min(lengths)}")
    print(f"\n🔍 Longest example ({max_length} tokens):")
    print(f"'{max_example}'")
    
    # Recommendations
    print(f"\n" + "="*50)
    print("💡 RECOMMENDATIONS")
    print("="*50)
    if p95_length <= 32:
        print("✅ Use max_len=32 (covers 95% of data)")
        recommended = 32
    elif p95_length <= 48:
        print("✅ Use max_len=48 (covers 95% of data)")
        recommended = 48
    elif p95_length <= 64:
        print("✅ Use max_len=64 (covers 95% of data)")
        recommended = 64
    else:
        recommended = p99_length
        print(f"⚠️  Consider max_len={p99_length} or {max_length}")
    
    current_waste = 64 - avg_length
    new_waste = recommended - avg_length
    print(f"\n📈 Efficiency gains:")
    print(f"Current max_len=64 wastes {current_waste:.1f} tokens on average")
    print(f"Recommended max_len={recommended} wastes {new_waste:.1f} tokens on average")
    if current_waste > new_waste:
        speedup = 64 / recommended
        print(f"🚀 Expected speedup: ~{speedup:.1f}x faster training!")
    
    print("\n✅ Analysis complete!")
    
    return max_length, avg_length, p95_length

if __name__ == "__main__":
    import sys
    print("🔍 Sequence Length Analyzer for Cricket Dataset")
    print("=" * 50)
    
    if len(sys.argv) != 2:
        print("❌ Usage: python find_max_seq_len.py <jsonl_file>")
        print("📝 Example: python find_max_seq_len.py cricket_prompts.jsonl")
        sys.exit(1)
    
    jsonl_file = sys.argv[1]
    print(f"🎯 Target file: {jsonl_file}")
    
    try:
        find_max_seq_length(jsonl_file)
    except FileNotFoundError:
        print(f"❌ Error: File '{jsonl_file}' not found")
    except Exception as e:
        print(f"❌ Error: {e}")