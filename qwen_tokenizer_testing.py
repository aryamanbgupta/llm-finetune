#!/usr/bin/env python3
"""
Find usable tokens in Qwen vocabulary for cricket outcomes (handles bytes tokens)
"""

from transformers import AutoTokenizer
import re

def find_usable_tokens():
    print("Loading Qwen tokenizer...")
    model_name = "Qwen/Qwen-1_8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")
    
    # Get the vocabulary
    vocab = tokenizer.get_vocab()
    
    # Convert tokens to strings (handle bytes if necessary)
    def safe_token_to_str(token):
        if isinstance(token, bytes):
            try:
                return token.decode('utf-8')
            except UnicodeDecodeError:
                return str(token)
        return str(token)
    
    # Get all tokens as strings
    all_tokens = []
    token_to_id = {}
    
    for token, token_id in vocab.items():
        str_token = safe_token_to_str(token)
        all_tokens.append(str_token)
        token_to_id[str_token] = token_id
    
    print(f"Successfully processed {len(all_tokens)} tokens")
    
    print("\n" + "="*60)
    print("SEARCHING FOR USABLE TOKENS")
    print("="*60)
    
    # 1. Look for special tokens with angle brackets
    print("1. Special tokens with angle brackets:")
    bracket_tokens = [token for token in all_tokens if token.startswith('<') and token.endswith('>')]
    print(f"   Found {len(bracket_tokens)} tokens")
    for token in sorted(bracket_tokens)[:20]:
        print(f"   {token} (ID: {token_to_id[token]})")
    if len(bracket_tokens) > 20:
        print(f"   ... and {len(bracket_tokens) - 20} more")
    
    # 2. Look for tokens with pipe brackets <|...|>
    print(f"\n2. Tokens with pipe brackets <|...|>:")
    pipe_tokens = [token for token in all_tokens if token.startswith('<|') and token.endswith('|>')]
    print(f"   Found {len(pipe_tokens)} tokens")
    for token in sorted(pipe_tokens)[:20]:
        print(f"   {token} (ID: {token_to_id[token]})")
    if len(pipe_tokens) > 20:
        print(f"   ... and {len(pipe_tokens) - 20} more")
    
    # 3. Look for unused/reserved tokens
    print(f"\n3. Tokens containing keywords (unused, extra, reserved, special):")
    unused_patterns = ['unused', 'extra', 'reserved', 'special']
    unused_tokens = []
    for pattern in unused_patterns:
        pattern_tokens = [token for token in all_tokens if pattern.lower() in token.lower()]
        unused_tokens.extend(pattern_tokens)
    
    unused_tokens = list(set(unused_tokens))  # Remove duplicates
    print(f"   Found {len(unused_tokens)} tokens")
    for token in sorted(unused_tokens)[:15]:
        print(f"   {token} (ID: {token_to_id[token]})")
    
    # 4. Look for numbered tokens that we could repurpose
    print(f"\n4. Numbered special tokens:")
    numbered_tokens = [token for token in all_tokens if re.search(r'<.*\d+.*>', token)]
    print(f"   Found {len(numbered_tokens)} tokens")
    for token in sorted(numbered_tokens)[:15]:
        print(f"   {token} (ID: {token_to_id[token]})")
    
    # 5. Test your original approach
    print(f"\n5. Testing your original token pattern:")
    EXTRA_BASE = 40
    OUTCOMES = ["0", "1", "2", "3", "4", "6", "WICKET"]
    
    for i, outcome in enumerate(OUTCOMES):
        test_token = f"<|extra_{EXTRA_BASE+i}|>"
        if test_token in all_tokens:
            print(f"   ✅ {outcome:7s} -> {test_token} (ID: {token_to_id[test_token]})")
        else:
            # Test encoding
            encoded = tokenizer.encode(test_token, add_special_tokens=False)
            print(f"   ❌ {outcome:7s} -> {test_token} -> {len(encoded)} tokens: {encoded}")
    
    # 6. Test simple alternatives
    print(f"\n6. Testing simple bracket alternatives:")
    simple_alternatives = ["[0]", "[1]", "[2]", "[3]", "[4]", "[6]", "[W]", "[WICKET]"]
    for alt in simple_alternatives:
        if alt in all_tokens:
            print(f"   ✅ {alt} exists (ID: {token_to_id[alt]})")
        else:
            encoded = tokenizer.encode(alt, add_special_tokens=False)
            print(f"   ❌ {alt} -> {len(encoded)} tokens: {encoded}")
    
    # 7. Test your original <O_*> pattern
    print(f"\n7. Testing <O_*> pattern:")
    for outcome in OUTCOMES:
        test_token = f"<O_{outcome}>"
        if test_token in all_tokens:
            print(f"   ✅ {outcome:7s} -> {test_token} (ID: {token_to_id[test_token]})")
        else:
            encoded = tokenizer.encode(test_token, add_special_tokens=False)
            decoded = tokenizer.decode(encoded)
            print(f"   📝 {outcome:7s} -> '{test_token}' -> {len(encoded)} tokens -> '{decoded}'")
    
    return pipe_tokens, bracket_tokens, unused_tokens, token_to_id

def create_working_mapping(pipe_tokens, token_to_id):
    """Create a working token mapping based on available pipe tokens"""
    
    # Filter out important system tokens
    system_keywords = ['im_start', 'im_end', 'system', 'user', 'assistant', 'tool', 'begin', 'end']
    suitable_tokens = []
    
    for token in pipe_tokens:
        is_system = any(keyword in token.lower() for keyword in system_keywords)
        if not is_system:
            suitable_tokens.append(token)
    
    outcomes = ["0", "1", "2", "3", "4", "6", "WICKET"]
    
    if len(suitable_tokens) >= 7:
        print(f"\n✅ WORKING MAPPING OPTION:")
        print("="*40)
        mapping = {}
        for i, outcome in enumerate(outcomes):
            token = suitable_tokens[i]
            mapping[outcome] = token
            print(f"   {outcome:7s} -> {token} (ID: {token_to_id[token]})")
        
        print(f"\nPython code:")
        print("OUTCOME2TOK = {")
        for i, outcome in enumerate(outcomes):
            token = suitable_tokens[i]
            print(f'    "{outcome}": "{token}",')
        print("}")
        
        return mapping
    else:
        print(f"\n❌ Not enough suitable single tokens found.")
        print(f"   Available: {len(suitable_tokens)}, Needed: {len(outcomes)}")
        return None

def recommend_approach(tokenizer):
    """Recommend the best approach based on findings"""
    print(f"\n" + "="*60)
    print("RECOMMENDED APPROACHES")
    print("="*60)
    
    print("🥇 RECOMMENDATION: Use multi-token approach (most reliable)")
    print("   Your original <O_*> tokens work fine as multi-token sequences:")
    
    outcomes = ["0", "1", "2", "3", "4", "6", "WICKET"]
    
    print(f"\n   OUTCOME2TOK = {{")
    for outcome in outcomes:
        print(f'       "{outcome}": "<O_{outcome}>",')
    print(f"   }}")
    
    print(f"\n   Benefits:")
    print(f"   ✅ No dependency on specific vocabulary tokens")
    print(f"   ✅ Model will learn these patterns during training")
    print(f"   ✅ Easy to read and debug")
    print(f"   ✅ Consistent token length (mostly 3-4 tokens each)")
    
    # Test the multi-token approach
    print(f"\n   Token breakdown:")
    for outcome in outcomes:
        token_str = f"<O_{outcome}>"
        encoded = tokenizer.encode(token_str, add_special_tokens=False)
        decoded = tokenizer.decode(encoded)
        print(f"   {outcome:7s}: '{token_str}' -> {encoded} -> '{decoded}'")

if __name__ == "__main__":
    try:
        pipe_tokens, bracket_tokens, unused_tokens, token_to_id = find_usable_tokens()
        
        # Try to create a working mapping from existing tokens
        if pipe_tokens:
            working_mapping = create_working_mapping(pipe_tokens, token_to_id)
        else:
            working_mapping = None
        
        # Always show the multi-token recommendation
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-1_8B", trust_remote_code=True)
        recommend_approach(tokenizer)
        
        print(f"\n🎯 BOTTOM LINE:")
        if working_mapping:
            print(f"   You can either use existing pipe tokens OR the multi-token approach")
        else:
            print(f"   Use the multi-token approach - it's simple and effective!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()