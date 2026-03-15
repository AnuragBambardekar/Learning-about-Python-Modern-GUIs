# Understanding the code

This document explains the **math behind the microGPT code**, directly mapping:

- Code blocks  
- Mathematical equations  
- Numerical examples  

The goal is to connect the **implementation to the mathematics** used in training a language model.

---

## 1. Training Objective

### Code

```python
loss_t = -probs[target_id].log()
```

### Mathematical Meaning

The model minimizes **negative log likelihood**:

$
L_t = -log P(x_{t+1} | x_1, x_2, ..., x_t)
$
Where:

- $x₁..xₜ$ = previous tokens
- $xₜ₊₁$ = correct next token

### Example

If the model predicts:

| Token | Probability |
|------|-------------|
| a | 0.1 |
| n | 0.8 |
| e | 0.1 |

Correct token = **n**

Loss:

$
L = -log(0.8) ≈ 0.22
$

If probability were 0.1:

$
L = -log(0.1) ≈ 2.30
$

The model is penalized strongly for confident mistakes.

---

## 2. Linear Layer

### Code

```python
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
```

### Mathematical Operation

Matrix multiplication:

$
y = W x
$

Where

- $x ∈ Rᵈ$
- $ W ∈ Rᵐˣᵈ$

Each output dimension:

$yᵢ = Σⱼ Wᵢⱼ xⱼ$

### Example

$x = [1,2]$

$W = \begin{bmatrix} 3 & 4 \\ 5 & 6 \end{bmatrix}$

Result:

$y₁ = 3(1) + 4(2) = 11$  
$y₂ = 5(1) + 6(2) = 17$

---

## 3. Softmax

### Code

```python
exps = [(val - max_val).exp() for val in logits]
total = sum(exps)
return [e / total for e in exps]
```

### Mathematical Formula

$
softmax(zᵢ) = e^(zᵢ) / Σⱼ e^(zⱼ)
$

Subtracting max value improves numerical stability.

### Example

Logits:

$[2, 4]$

Exponentials:

$[e², e⁴] = [7.39, 54.6]$

Probabilities:

$[0.12, 0.88]$

---

## 4. RMSNorm

### Code

```python
ms = sum(xi * xi for xi in x) / len(x)
scale = (ms + 1e-5) ** -0.5
return [xi * scale for xi in x]
```

### Mathematical Definition

Mean square:

$ms = (1/d) Σ xᵢ²$

Scale factor:

$scale = 1 / sqrt(ms + ε)$

Output:

$x̂ᵢ = xᵢ * scale$

This keeps vector magnitude stable.

---

## 5. Attention Mechanism

### Code

```python
attn_logits = [
    sum(q_h[j] * k_h[t][j] for j in range(head_dim)) 
    / head_dim**0.5
    for t in range(len(k_h))
]
```

### Mathematical Equation

$scoreₜ = (Q · Kₜ) / √d$

Where

- $Q$ = query vector
- $Kₜ$ = key vector
- $d$ = dimension

### Example

$Q = [1,2]$  
$K = [3,4]$

Dot product:

$1×3 + 2×4 = 11$

Scaled:

$11 / √2 ≈ 7.78$

---

## 6. Attention Weights

### Code

```python
attn_weights = softmax(attn_logits)
```

### Mathematical Equation

$αₜ = e^(scoreₜ) / Σ e^(score)$

These weights determine **how much each previous token matters**.

---

## 7. Weighted Sum of Values

### Code

```python
head_out = [
    sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
    for j in range(head_dim)
]
```

### Mathematical Equation

$output = Σ αₜ Vₜ$

Full attention formula:

$Attention(Q,K,V) = softmax(QKᵀ / √d) V$

---

# 8. ReLU Activation

### Code

```python
xi.relu()
```

### Mathematical Function

$ReLU(x) = max(0, x)$

Example:

$[-2,3] → [0,3]$

Adds non-linearity.

---

# 9. Total Sequence Loss

### Code

```python
loss = (1 / n) * sum(losses)
```

### Mathematical Formula

$L = (1/n) Σ -log P(xₜ₊₁ | x≤ₜ)$

This is **average cross-entropy loss** across tokens.

---

# 10. Backpropagation

### Code

```python
loss.backward()
```

### Mathematical Principle

Uses the **chain rule**:

$∂L/∂W = (∂L/∂z) × (∂z/∂W)$

Where

- W = parameters
- z = intermediate outputs

The `Value` class builds a **computation graph** so gradients can propagate backward.

---

# 11. Adam Optimizer

### Code

```python
m[i] = beta1 * m[i] + (1 - beta1) * p.grad
v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
```

### Mathematical Equations

First moment:

$mₜ = β₁ mₜ₋₁ + (1-β₁) gₜ$

Second moment:

$vₜ = β₂ vₜ₋₁ + (1-β₂) gₜ²$

Update:

$θ = θ − η (mₜ / (√vₜ + ε))$

Adam stabilizes training.

---

# 12. Inference (Generation)

### Code

```python
token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
```

### Mathematical Meaning

Sampling from the probability distribution:

$xₜ₊₁ ~ P(xₜ₊₁ | x≤ₜ)$

Temperature modifies probabilities:

$Pᵢ ∝ e^(zᵢ / T)$

Where

- $T < 1 → deterministic$
- $T > 1 → creative/random$

---

# Final Mathematical Summary

The model pipeline:

Input tokens  
→ Embedding  
→ Attention  
→ MLP  
→ Logits  
→ Softmax  

Training objective:

$minimize Σ -log P(xₜ₊₁ | x≤ₜ)$

This is optimized with **gradient descent**.
