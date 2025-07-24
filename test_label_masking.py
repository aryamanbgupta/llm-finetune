import numpy as np
import torch
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, log_loss
import warnings
warnings.filterwarnings('ignore')

# Setup tokenizer and special tokens
model_name = "Qwen/Qwen1.5-1.8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)

# Define outcome tokens
EXTRA_BASE = 40
OUTCOMES = ["0", "1", "2", "3", "4", "6", "WICKET"]
OUTCOME_INDICES = {o: i for i, o in enumerate(OUTCOMES)}
OUTCOME2TOK = {o: f"<|extra_{EXTRA_BASE+i}|>" for i, o in enumerate(OUTCOMES)}
TOK2OUTCOME = {v: k for k, v in OUTCOME2TOK.items()}

# Add special tokens
special_tokens_to_add = list(OUTCOME2TOK.values())
tokenizer.add_special_tokens({'additional_special_tokens': special_tokens_to_add})

def original_compute_metrics(logits, labels):
    """Original implementation that computes over full vocabulary"""
    predictions = np.argmax(logits, axis=-1)
    probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    
    true_labels_token_ids = []
    predicted_token_ids = []
    true_probs_for_loss = []
    
    for i in range(len(labels)):
        valid_label_indices = np.where(labels[i] != -100)[0]
        if len(valid_label_indices) > 0:
            label_idx = valid_label_indices[0]
            true_labels_token_ids.append(labels[i][label_idx])
            predicted_token_ids.append(predictions[i][label_idx])
            true_probs_for_loss.append(probabilities[i][label_idx])  # Full vocab probs!
    
    # Convert token IDs to outcome indices
    true_tokens = tokenizer.convert_ids_to_tokens(true_labels_token_ids)
    predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_token_ids)
    
    true_labels_indices = [OUTCOME_INDICES[TOK2OUTCOME[token]] for token in true_tokens]
    predicted_indices = [OUTCOME_INDICES.get(TOK2OUTCOME.get(token, ""), -1) for token in predicted_tokens]
    
    # Calculate metrics with wrong shape
    accuracy = accuracy_score(true_labels_indices, predicted_indices)
    
    # This will fail or give wrong results!
    try:
        cross_entropy = log_loss(true_labels_indices, true_probs_for_loss, labels=np.arange(len(OUTCOMES)))
    except Exception as e:
        cross_entropy = f"ERROR: {str(e)}"
    
    return {
        "accuracy": accuracy,
        "cross_entropy": cross_entropy,
        "prob_shape": np.array(true_probs_for_loss).shape if true_probs_for_loss else None
    }

def fixed_compute_metrics(logits, labels):
    """Fixed implementation that computes over 7 outcome tokens only"""
    outcome_token_ids = tokenizer.convert_tokens_to_ids(list(OUTCOME2TOK.values()))
    
    true_labels = []
    pred_probs = []
    pred_labels = []
    
    for i in range(len(labels)):
        valid_indices = np.where(labels[i] != -100)[0]
        if len(valid_indices) > 0:
            idx = valid_indices[0]
            
            # Get logits only for outcome tokens at the target position
            outcome_logits = logits[i, idx, outcome_token_ids]
            probs = torch.softmax(torch.tensor(outcome_logits), dim=-1).numpy()
            pred_probs.append(probs)
            pred_labels.append(np.argmax(probs))
            
            # Map true label to outcome index
            true_token_id = labels[i][idx]
            true_token = tokenizer.convert_ids_to_tokens([true_token_id])[0]
            true_outcome = TOK2OUTCOME.get(true_token, "0")
            true_labels.append(OUTCOME_INDICES[true_outcome])
    
    # Now compute metrics with correct shapes
    pred_probs = np.array(pred_probs)  # Shape: (n_samples, 7)
    accuracy = accuracy_score(true_labels, pred_labels)
    # Need to specify all possible labels for sklearn
    cross_entropy = log_loss(true_labels, pred_probs, labels=np.arange(len(OUTCOMES)))
    
    return {
        "accuracy": accuracy,
        "cross_entropy": cross_entropy,
        "prob_shape": pred_probs.shape
    }

# Create synthetic test data
print("="*80)
print("SYNTHETIC TEST: Creating fake model outputs")
print("="*80)

# Simulate model outputs for 3 examples
n_samples = 3
seq_len = 40
vocab_size = len(tokenizer)

# Get outcome token IDs
outcome_token_ids = tokenizer.convert_tokens_to_ids(list(OUTCOME2TOK.values()))
print(f"\nVocabulary size: {vocab_size}")
print(f"Outcome token IDs: {outcome_token_ids}")

# Create fake logits (batch_size, seq_len, vocab_size)
np.random.seed(42)
logits = np.random.randn(n_samples, seq_len, vocab_size).astype(np.float32)

# Make the model "predict" specific outcomes by boosting their logits
target_outcomes = ["0", "WICKET", "4"]  # What we want the model to predict
target_positions = [25, 30, 10]  # Where in the sequence the targets are

# Create labels (mostly -100 except at target positions)
labels = np.full((n_samples, seq_len), -100, dtype=np.int32)

for i, (outcome, pos) in enumerate(zip(target_outcomes, target_positions)):
    token_id = tokenizer.convert_tokens_to_ids(OUTCOME2TOK[outcome])
    labels[i, pos] = token_id
    # Boost the logit for this outcome to make it likely
    logits[i, pos, token_id] += 5.0

print(f"\nTest setup:")
for i in range(n_samples):
    pos = target_positions[i]
    outcome = target_outcomes[i]
    token_id = tokenizer.convert_tokens_to_ids(OUTCOME2TOK[outcome])
    print(f"  Example {i}: Target '{outcome}' (token_id={token_id}) at position {pos}")

# Run both versions
print("\n" + "="*80)
print("RUNNING ORIGINAL COMPUTE_METRICS")
print("="*80)
orig_results = original_compute_metrics(logits, labels)
print(f"Probability shape: {orig_results['prob_shape']} (should be (3, 7) but is (3, {vocab_size})!)")
print(f"Accuracy: {orig_results['accuracy']:.4f}")
print(f"Cross-entropy: {orig_results['cross_entropy']}")

print("\n" + "="*80)
print("RUNNING FIXED COMPUTE_METRICS")
print("="*80)
fixed_results = fixed_compute_metrics(logits, labels)
print(f"Probability shape: {fixed_results['prob_shape']} ✓")
print(f"Accuracy: {fixed_results['accuracy']:.4f}")
print(f"Cross-entropy: {fixed_results['cross_entropy']:.4f}")

# Detailed analysis of what's happening
print("\n" + "="*80)
print("DETAILED ANALYSIS: Why the original is broken")
print("="*80)

# Look at one example in detail
example_idx = 0
pos = target_positions[example_idx]
target_outcome = target_outcomes[example_idx]
target_token_id = tokenizer.convert_tokens_to_ids(OUTCOME2TOK[target_outcome])

# Get probabilities both ways
full_probs = torch.softmax(torch.tensor(logits[example_idx, pos, :]), dim=-1).numpy()
outcome_logits = logits[example_idx, pos, outcome_token_ids]
outcome_probs = torch.softmax(torch.tensor(outcome_logits), dim=-1).numpy()

print(f"\nExample {example_idx}: Target outcome is '{target_outcome}'")
print(f"\nOriginal method (WRONG):")
print(f"  - Computes softmax over {vocab_size} tokens")
print(f"  - Probability mass spread across entire vocabulary")
print(f"  - Target token probability: {full_probs[target_token_id]:.6f}")
print(f"  - But sklearn expects probabilities for indices 0-6, not token IDs!")
print(f"  - sklearn looks at index {OUTCOME_INDICES[target_outcome]}, which has probability: {full_probs[OUTCOME_INDICES[target_outcome]]:.6f}")
print(f"  - This is the probability of token ID {OUTCOME_INDICES[target_outcome]} ('{tokenizer.decode([OUTCOME_INDICES[target_outcome]])}'), NOT our outcome!")

print(f"\nFixed method (CORRECT):")
print(f"  - Computes softmax over only 7 outcome tokens")
print(f"  - Probability mass concentrated on actual outcomes")
print(f"  - Outcome probabilities: {dict(zip(OUTCOMES, outcome_probs))}")
print(f"  - Target outcome probability: {outcome_probs[OUTCOME_INDICES[target_outcome]]:.6f}")

# Show why metrics are meaningless with original method
print("\n" + "="*80)
print("WHY THE ORIGINAL METRICS ARE MEANINGLESS")
print("="*80)
print("The original method has sklearn compute log_loss using:")
print("  - y_true: [0, 6, 4] (correct outcome indices)")
print("  - y_pred: probability vectors of shape (151653,) instead of (7,)")
print("\nWhen sklearn calculates the loss for class 0, it looks at y_pred[0],")
print("which is the probability of token '<|endoftext|>', not outcome '0'!")
print("\nThis makes the cross-entropy calculation completely meaningless.")

# Test what happens with more realistic logits
print("\n" + "="*80)
print("REALISTIC TEST: What happens during actual training")
print("="*80)

# Create more realistic logits where model is uncertain
realistic_logits = np.random.randn(1, 50, vocab_size).astype(np.float32) * 0.1
realistic_labels = np.full((1, 50), -100, dtype=np.int32)

# Set target at position 30
target_pos = 30
target_outcome = "2"
target_token_id = tokenizer.convert_tokens_to_ids(OUTCOME2TOK[target_outcome])
realistic_labels[0, target_pos] = target_token_id

# Model is uncertain - similar logits for all outcomes
for oid in outcome_token_ids:
    realistic_logits[0, target_pos, oid] = np.random.randn() * 0.5

print(f"Model is uncertain about outcome at position {target_pos}")

# Compare methods
orig_metrics = original_compute_metrics(realistic_logits, realistic_labels)
fixed_metrics = fixed_compute_metrics(realistic_logits, realistic_labels)

print(f"\nOriginal cross-entropy: {orig_metrics['cross_entropy']}")
print(f"Fixed cross-entropy: {fixed_metrics['cross_entropy']:.4f}")
print("\nThe original gives misleading values because it's measuring")
print("the model's ability to predict random vocabulary tokens,")
print("not its ability to predict cricket outcomes!")