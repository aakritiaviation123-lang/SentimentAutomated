"""
NUMERICAL WALKTHROUGH: Sentiment Analysis Data Processing
Interactive examples showing how numbers flow through the pipeline
"""

import numpy as np
from scipy.special import softmax
import pandas as pd

print("=" * 80)
print("SENTIMENT ANALYSIS: NUMERICAL DATA PROCESSING")
print("=" * 80)

# ============================================================================
# PART 1: TOKENIZATION
# ============================================================================
print("\n\n### PART 1: TOKENIZATION (Text → Token IDs) ###\n")

# Simulated vocabulary (actual model has 50,265 tokens)
vocabulary = {
    "This": 1045,
    "phone": 2340,
    "is": 2003,
    "amazing": 6429,
    "[CLS]": 101,
    "[SEP]": 102,
    "[PAD]": 0
}

text = "This phone is amazing"
tokens = text.split()
print(f"Input text: '{text}'")
print(f"Tokens: {tokens}")

# Convert to IDs
token_ids = [vocabulary["[CLS]"]] + [vocabulary[t] for t in tokens] + [vocabulary["[SEP]"]]
print(f"Token IDs: {token_ids}")

# Pad to length 128
max_length = 128
padded_ids = token_ids + [0] * (max_length - len(token_ids))
print(f"After padding to {max_length} tokens: {padded_ids[:6]}...{padded_ids[-5:]}")
print(f"Shape: [{len(padded_ids)}]")

# ============================================================================
# PART 2: EMBEDDINGS
# ============================================================================
print("\n\n### PART 2: EMBEDDING LAYER (Token IDs → Dense Vectors) ###\n")

# Simulate embeddings (each token → 768-dim vector)
hidden_size = 768
num_tokens = len(padded_ids)

# Random embeddings (in reality, these are learned weights)
np.random.seed(42)
embeddings = np.random.randn(num_tokens, hidden_size) * 0.1

print(f"Embeddings shape: [{num_tokens}, {hidden_size}]")
print(f"First token ([CLS]) embedding (first 10 dims):")
print(f"  {embeddings[0, :10]}")
print(f"Second token (This) embedding (first 10 dims):")
print(f"  {embeddings[1, :10]}")

# ============================================================================
# PART 3: ATTENTION MECHANISM
# ============================================================================
print("\n\n### PART 3: ATTENTION (First Transformer Layer Example) ###\n")

# Simplified attention for first 5 tokens
seq_len = 5
attention_dim = 64  # 768 / 12 heads

# Simulate Query, Key, Value projections
Query = np.random.randn(seq_len, attention_dim) * 0.1
Key = np.random.randn(seq_len, attention_dim) * 0.1
Value = np.random.randn(seq_len, attention_dim) * 0.1

# Compute attention scores
scores = Query @ Key.T / np.sqrt(attention_dim)
print(f"Attention scores (before softmax):")
print(f"{scores.round(3)}\n")

# Apply softmax to get attention weights
attention_weights = softmax(scores, axis=1)
print(f"Attention weights (after softmax - rows sum to 1.0):")
print(f"{attention_weights.round(3)}")
print(f"Row sums: {attention_weights.sum(axis=1).round(4)}")

# Apply attention to values
attended = attention_weights @ Value
print(f"\nAttended values shape: [{seq_len}, {attention_dim}]")
print(f"Attended values (first token, first 5 dims): {attended[0, :5].round(4)}")

# ============================================================================
# PART 4: MODEL OUTPUT (LOGITS)
# ============================================================================
print("\n\n### PART 4: MODEL OUTPUT (Logits for 3 Classes) ###\n")

# Simulate 3 different texts with different logits
texts_examples = [
    {"text": "This phone is amazing!", "logits": [1.2, 0.8, 4.5]},
    {"text": "The weather is nice", "logits": [0.5, 3.1, 1.2]},
    {"text": "Terrible service!", "logits": [5.8, 0.2, 0.1]}
]

labels = ['Negative', 'Neutral', 'Positive']

for example in texts_examples:
    print(f"Text: '{example['text']}'")
    print(f"Raw logits: {example['logits']}")
    
    # ================================================================
    # PART 5: SOFTMAX CALCULATION
    # ================================================================
    logits = np.array(example['logits'])
    
    # Step 1: Exponentiate
    exp_logits = np.exp(logits)
    print(f"  Exponentials: {exp_logits.round(3)}")
    
    # Step 2: Sum
    sum_exp = np.sum(exp_logits)
    print(f"  Sum of exponentials: {sum_exp:.3f}")
    
    # Step 3: Normalize
    probs = exp_logits / sum_exp
    print(f"  Probabilities:")
    
    for i, (label, prob) in enumerate(zip(labels, probs)):
        percentage = prob * 100
        print(f"    {label:10s}: {prob:.4f} ({percentage:5.2f}%)")
    
    # Prediction
    pred_idx = np.argmax(probs)
    confidence = np.max(probs)
    print(f"  → Prediction: {labels[pred_idx]}, Confidence: {confidence:.4f}\n")

# ============================================================================
# PART 6: BATCH AGGREGATION (5000 texts)
# ============================================================================
print("\n\n### PART 6: AGGREGATION (5000 Predictions) ###\n")

# Simulate predictions for 5000 texts
np.random.seed(42)
num_predictions = 5000

# Generate random predictions with realistic distribution
predictions_dist = np.random.dirichlet([1, 1.2, 0.9])  # Realistic bias
predictions = np.random.choice(
    [0, 1, 2],
    size=num_predictions,
    p=predictions_dist
)
confidences = np.random.beta(8, 2, num_predictions)  # Biased toward high confidence

# Count sentiments
sentiment_counts = pd.Series(predictions).value_counts().sort_index()
sentiment_labels_dict = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

print(f"Sentiment Distribution ({num_predictions} predictions):\n")
total = len(predictions)
for idx, count in sentiment_counts.items():
    label = sentiment_labels_dict[idx]
    percentage = (count / total) * 100
    print(f"  {label:10s}: {count:5d} tweets ({percentage:5.2f}%)")

# Confidence statistics
print(f"\nConfidence Statistics:")
print(f"  Mean confidence:        {confidences.mean():.4f} ({confidences.mean()*100:.2f}%)")
print(f"  Median confidence:      {np.median(confidences):.4f}")
print(f"  Std deviation:          {confidences.std():.4f}")
print(f"  Min confidence:         {confidences.min():.4f}")
print(f"  Max confidence:         {confidences.max():.4f}")
print(f"  High-confidence (>0.9): {(confidences > 0.9).sum()} ({(confidences > 0.9).mean()*100:.1f}%)")

# ============================================================================
# PART 7: SENTIMENT SCORE
# ============================================================================
print("\n\n### PART 7: OVERALL SENTIMENT SCORE ###\n")

# Map to numeric scores
score_map = {0: -1, 1: 0, 2: 1}
scores = np.array([score_map[p] for p in predictions])

overall_score = scores.mean()
print(f"Score mapping:")
print(f"  Negative → -1")
print(f"  Neutral  →  0")
print(f"  Positive → +1")
print(f"\nCalculation:")
print(f"  Sum = (1,850 × 1) + (2,100 × 0) + (1,050 × -1)")
print(f"  Sum = {(predictions == 2).sum()} - {(predictions == 0).sum()} = {scores.sum()}")
print(f"  Overall Score = {scores.sum()} / {total} = {overall_score:.4f}")
print(f"\nInterpretation: Slight {('positive' if overall_score > 0 else 'negative')} bias")

# ============================================================================
# PART 8: WEIGHTED CONFIDENCE
# ============================================================================
print("\n\n### PART 8: WEIGHTED SENTIMENT WITH CONFIDENCE ###\n")

weighted_scores = scores * confidences
weighted_sentiment = weighted_scores.mean()

print(f"Weighted calculation (sentiment × confidence):")
print(f"  Without weighting: {overall_score:.4f}")
print(f"  With confidence weighting: {weighted_sentiment:.4f}")
print(f"  Difference: {(weighted_sentiment - overall_score):.4f}")

# ============================================================================
# PART 9: TIME COMPLEXITY
# ============================================================================
print("\n\n### PART 9: COMPUTATIONAL COMPLEXITY ###\n")

print(f"Model: RoBERTa-base-sentiment")
print(f"  Total parameters: 125,000,000")
print(f"  Attention heads: 12")
print(f"  Layers: 12")
print(f"  Hidden dimension: 768")
print(f"  Vocab size: 50,265")
print(f"\nInference time per text:")
print(f"  CPU: ~5-10 milliseconds")
print(f"  GPU: ~0.5-1 milliseconds")
print(f"\nFor 5,000 texts:")
print(f"  CPU: ~30-50 seconds")
print(f"  GPU: ~3-5 seconds")

print(f"\nMemory usage:")
print(f"  Model weights: ~500 MB")
print(f"  Per-sample overhead: ~1 MB")
print(f"  Total for 5000 in memory: ~5.5 GB")

# ============================================================================
# PART 10: UNCERTAINTY ANALYSIS
# ============================================================================
print("\n\n### PART 10: UNCERTAINTY & ERROR ANALYSIS ###\n")

# Generate cases with varying confidence
uncertain_examples = [
    ("Great product!", [0.05, 0.10, 0.85]),  # Clear positive
    ("It's okay I guess", [0.20, 0.60, 0.20]),  # Ambiguous (neutral)
    ("Not bad but could be better", [0.35, 0.40, 0.25]),  # Ambiguous (mixed)
]

for text, logits_raw in uncertain_examples:
    probs = softmax(np.array(logits_raw))
    pred = labels[np.argmax(probs)]
    conf = np.max(probs)
    
    print(f"Text: '{text}'")
    print(f"  Prediction: {pred}")
    print(f"  Confidence: {conf:.4f} ({conf*100:.2f}%)")
    print(f"  Distribution: {', '.join([f'{l}:{p:.2f}' for l, p in zip(labels, probs)])}")
    
    if conf < 0.60:
        print(f"  ⚠️  WARNING: Low confidence - prediction unreliable")
    print()

print("\n" + "=" * 80)
print("END OF NUMERICAL WALKTHROUGH")
print("=" * 80)
