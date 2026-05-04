# Deep Dive: Numerical Data Processing in Sentiment Analysis

## Architecture Overview

Your system follows this numerical pipeline:

```
Raw Text → Tokenization → Embedding → Neural Network → Logits → Softmax → Classification
```

---

## 1. TOKENIZATION: Text → Numbers

### What Happens
The tokenizer breaks text into tokens and maps them to integer IDs from a vocabulary of 50,265 words.

### Example
```
Text: "This phone is amazing"

Step 1 - Split into tokens:
["This", "phone", "is", "amazing"]

Step 2 - Map to vocabulary IDs:
Vocabulary contains:
  "This" → ID: 1045
  "phone" → ID: 2340
  "is" → ID: 2003
  "amazing" → ID: 6429

Result: [1045, 2340, 2003, 6429]

Step 3 - Add special tokens:
[CLS] token: 101 (marks start)
[SEP] token: 102 (marks end)

Final token sequence: [101, 1045, 2340, 2003, 6429, 102]
```

### Padding (Max Length = 128)
If text is shorter than 128 tokens, pad with zeros:
```
[101, 1045, 2340, 2003, 6429, 102, 0, 0, 0, ..., 0] (128 total)
```

**Mathematical dimension:** `[batch_size=1, sequence_length=128]` = numerical matrix

---

## 2. EMBEDDING LAYER: Convert to Dense Vectors

### Process
Each token ID is converted to a 768-dimensional vector:

```
Token ID 1045 → [0.234, -0.512, 0.891, ..., -0.123]  (768 dimensions)
Token ID 2340 → [-0.156, 0.423, -0.234, ..., 0.567]
Token ID 2003 → [0.789, -0.234, 0.456, ..., -0.891]
Token ID 6429 → [0.123, 0.567, -0.789, ..., 0.234]
```

**Result:** A matrix of shape `[128, 768]` representing the entire sequence

### Why 768?
- RoBERTa-base uses 768 hidden dimensions
- Each value learned through training on 1 billion+ Twitter posts
- Higher dimensions = more expressive representation

---

## 3. TRANSFORMER PROCESSING: Neural Network Inference

### Input
```
Embeddings matrix: [128, 768]  (128 token vectors, each 768-dim)
```

### What the Model Does (Simplified)

#### Layer 1-12: Self-Attention + Feed-Forward
Each layer performs:
```
Output = Attention(Query @ Key^T / √64) @ Value + FFN(x)

Where:
- Query, Key, Value are learned transformations of input
- √64 is scale factor (64 = 768/12 attention heads)
- FFN is a 2-layer neural network: ReLU(x @ W1 + b1) @ W2 + b2
```

**Numerical example of attention:**
```
Similarity matrix (Query @ Key^T):
         Token_1  Token_2  Token_3  Token_4
Token_1  [2.3,    1.2,     0.5,     -0.3]
Token_2  [1.1,    2.8,     1.4,     0.2]
Token_3  [0.4,    1.3,     2.1,     0.9]
Token_4  [-0.2,   0.3,     0.8,     1.9]

After softmax (normalizes to 0-1 probabilities):
         Token_1  Token_2  Token_3  Token_4
Token_1  [0.42,   0.31,    0.17,    0.10]
Token_2  [0.28,   0.48,    0.18,    0.06]
Token_3  [0.12,   0.26,    0.45,    0.17]
Token_4  [0.08,   0.14,    0.32,    0.46]

This shows which tokens "attend to" (pay attention to) other tokens.
```

### Output After 12 Layers
Final hidden state: `[128, 768]` matrix

---

## 4. CLASSIFICATION HEAD: Extract 3 Logits

### Process
Take the `[CLS]` token output (first position) and pass through linear layer:

```
Input: [CLS] vector = [0.234, -0.512, ..., 0.123]  (768 dims)

Linear transformation: X @ W + b
Where W is [768 × 3] weight matrix, b is [3] bias vector

Output logits: [1.2, 0.8, 4.5]  (3 values for 3 classes)
```

### Mathematical Formula
```
logit_negative = Σ(input_i × W_i,0) + b_0 = 1.2
logit_neutral  = Σ(input_i × W_i,1) + b_1 = 0.8
logit_positive = Σ(input_i × W_i,2) + b_2 = 4.5
```

---

## 5. SOFTMAX: Convert Logits to Probabilities

### The Formula
```
P(class_i) = e^(logit_i) / Σ(e^(logit_j) for all j)
```

### Numerical Example
```
Raw logits: [1.2, 0.8, 4.5]

Step 1 - Exponentiate:
e^1.2 = 3.32
e^0.8 = 2.23
e^4.5 = 90.02

Step 2 - Sum exponents:
Sum = 3.32 + 2.23 + 90.02 = 95.57

Step 3 - Normalize:
P(Negative) = 3.32 / 95.57 = 0.0347 (3.47%)
P(Neutral)  = 2.23 / 95.57 = 0.0233 (2.33%)
P(Positive) = 90.02 / 95.57 = 0.9420 (94.20%)

Result: [0.0347, 0.0233, 0.9420]
✓ These sum to 1.0 (probability distribution)
```

### Why Softmax?
- Guarantees probabilities between 0-1
- Emphasizes the highest logit (exponential effect)
- Creates a proper probability distribution

---

## 6. CONFIDENCE & PREDICTION

### Prediction Rule
```
predicted_class = argmax([0.0347, 0.0233, 0.9420])
                = index 2 = "Positive"

confidence = max([0.0347, 0.0233, 0.9420])
           = 0.9420
```

### Statistical Properties
```
For a batch of 5,000 texts:

confidence_values = [0.89, 0.92, 0.78, 0.91, ..., 0.85]

Statistics:
- Mean confidence = 0.874 (87.4% average certainty)
- Std deviation = 0.089 (8.9% variance)
- Min confidence = 0.523 (weakest prediction)
- Max confidence = 0.998 (strongest prediction)
- Texts with >90% confidence = 3,245 (64.9%)
```

---

## 7. AGGREGATION: From Individual Predictions to Summary

### Per-Sample Output
```
Row 1: Tweet = "Love this!" 
       → [Negative: 0.05, Neutral: 0.12, Positive: 0.83]
       → Prediction: Positive, Confidence: 0.83

Row 2: Tweet = "It's okay I guess"
       → [Negative: 0.15, Neutral: 0.72, Positive: 0.13]
       → Prediction: Neutral, Confidence: 0.72

Row 3: Tweet = "Worst experience ever"
       → [Negative: 0.91, Neutral: 0.07, Positive: 0.02]
       → Prediction: Negative, Confidence: 0.91
```

### Aggregated Statistics (5,000 rows)
```
Sentiment counts:
- Positive: 1,850 tweets
- Neutral:  2,100 tweets
- Negative: 1,050 tweets

Distribution percentages:
- Positive: 1,850 / 5,000 = 37.0%
- Neutral:  2,100 / 5,000 = 42.0%
- Negative: 1,050 / 5,000 = 21.0%

Confidence aggregations:
- Average confidence: 0.874
- Median confidence: 0.889
- High-confidence predictions (>0.9): 3,245 / 5,000 = 64.9%
```

### Numerical Score (Optional)
```
Score mapping:
Negative → -1
Neutral  → 0
Positive → +1

Sentiment Score = Σ(individual scores) / n
                = (1,850×1 + 2,100×0 + 1,050×-1) / 5,000
                = (1,850 + 0 - 1,050) / 5,000
                = 800 / 5,000
                = 0.16

Interpretation: Overall dataset has slight positive bias (0.16 on -1 to +1 scale)
```

---

## 8. YOUR CODE'S NUMERICAL PIPELINE

```python
# sentiment_limited.py execution flow:

1. Load CSV (5,000 rows sampled)
2. For each text:
   a) Tokenize → [128] token IDs
   b) Pass through model (12 layers) → [3] logits
   c) Apply softmax → [3] probabilities
   d) Extract prediction & confidence
3. Create DataFrame with 5,000 rows:
   [Original text] | [Sentiment] | [Confidence]
4. Calculate statistics:
   - sentiment_counts = value_counts()
   - avg_conf = mean(confidence)
   - high_conf_count = count(confidence > 0.9)
5. Generate visualizations with actual numerical values
```

---

## 9. COMPUTATIONAL COMPLEXITY

### Parameters Count
```
RoBERTa-base model:
- 12 transformer layers × (attention + FFN + normalization)
- Total parameters: ~125 million weights

Per forward pass:
- Matrix multiplications: ~2,000+ operations
- For 5,000 texts: ~10 million matrix operations
- Time on CPU: ~30 seconds
- Time on GPU: ~3-5 seconds
```

### Memory Requirements
```
Per text inference:
- Token embeddings: 128 × 768 × 4 bytes = 393 KB
- Attention matrices: 12 layers × 128 × 128 = 197 KB
- Total per sample: ~1 MB

Batch of 5,000:
- Model weights: ~500 MB
- Intermediate activations: ~5 GB (if keeping all in memory)
```

---

## 10. ERROR & UNCERTAINTY ANALYSIS

### Sources of Numerical Error
```
1. Tokenization ambiguity:
   "Don't" → ["Don", "'t"] vs ["Do", "n't"]
   
2. Context loss:
   Max 128 tokens = ~500 characters lost if text longer
   
3. Floating-point precision:
   softmax([1.2, 0.8, 4.5]) may vary slightly
   due to 32-bit vs 64-bit arithmetic
   
4. Model calibration:
   Confidence 0.95 ≠ 95% actual accuracy
   (overconfident on uncertain cases)
```

### Low-Confidence Cases
```
Example: "The movie was... interesting"

Logits: [0.2, 1.1, 0.8]
After softmax: [0.32, 0.41, 0.27]
Confidence: 0.41 (only 41% certain)

Interpretation: Text is genuinely ambiguous
(Neither clearly positive nor negative)
```

---

## Summary Table: Data Transformation

| Stage | Input | Operation | Output | Dimensions |
|-------|-------|-----------|--------|-----------|
| Raw | Text string | Tokenization | Token IDs | `[128]` |
| Embedding | Token IDs | Lookup table | Dense vectors | `[128, 768]` |
| Encoder | Embeddings | 12-layer Transformer | Hidden states | `[128, 768]` |
| Classification | [CLS] token | Linear + Softmax | Probabilities | `[3]` |
| Prediction | Probabilities | argmax + max | Class + Confidence | `[string, float]` |
| Aggregation | 5,000 predictions | Counting + Statistics | Report | Summary stats |

---

## Quick Reference: Key Numbers

```
Model: cardiffnlp/twitter-roberta-base-sentiment
- Training data: ~2.4 billion English Twitter posts
- Vocabulary size: 50,265 tokens
- Hidden dimension: 768
- Attention heads: 12
- Transformer layers: 12
- Total parameters: ~125 million
- Max input length: 128 tokens (~500 chars)

Output classes: 3
- [0] Negative
- [1] Neutral
- [2] Positive

Typical inference time:
- CPU: 5-10ms per text
- GPU: 0.5-1ms per text
```

