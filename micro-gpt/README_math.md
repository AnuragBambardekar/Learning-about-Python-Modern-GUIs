# Looking at it Mathematically

## 1. What Problem Is GPT Solving?

GPT learns a **conditional probability distribution**:

$
P(x_{t+1} \mid x_1, x_2, \dots, x_t)
$

That is:

> Given all previous characters, what is the probability of the next one?

### Example Dataset Entry
```
Name: "anna"
```
Training creates pairs:

| Context (input) | Target (next char) |
|-----------------|--------------------|
| BOS             | a                  |
| BOS a           | n                  |
| BOS a n         | n                  |
| BOS a n n       | a                  |
| BOS a n n a     | BOS                |

---

The model’s job is to assign high probability to the correct next character.

## 2. From Characters to Numbers

Characters are mapped to integers:

a → 0 <br>
n → 1 <br>
BOS → 2 <br>

The model never sees letters — only numbers.

---

## 3. Logits and Probabilities

The model outputs **logits** (raw scores):

$
\text{logits} = [2.0,\ 4.0,\ 0.5]
$

These are converted into probabilities using **softmax**:

$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
$

Result:

| Character | Probability |
|---------|-------------|
| a       | 0.11        |
| n       | 0.81        |
| BOS     | 0.08        |


So the model says:

    “There is an 81% chance the next character is n.”

That’s the entire output of GPT.

---

## 4. The Loss Function (How Wrong Was the Model?)

GPT uses **negative log-likelihood**:

```
Why negative log likelihood:

Probability vs loss:

| Character | Probability |
|---------|-------------|
| 0.99    | ~0.01       |
| 0.9     | 0.10        |
| 0.5     | 0.69        |
| 0.1     | 2.30        |
| 0.01    | 4.60        |

As probability of the correct answer increases, loss decreases smoothly.

Negative log likelihood is simply a way to reward high probability for correct predictions and punish wrong confident predictions while keeping the math stable for training.

```

$
\text{Loss} = -\log P(\text{correct token})
$

If the correct next token is `n`:

$
\text{Loss} = -\log(0.81) \approx 0.21
$

- Small loss → confident & correct  
- Large loss → confident & wrong  

---

## 5. Embeddings

Each token is mapped to a vector.

Embeddings are one of the **most important ideas in language models**.  
They convert **discrete symbols (tokens)** into **continuous vectors** that a neural network can operate on.

Simply because - Computers do not understand words like humans do.

```
Intuition:

One simple method is **one-hot encoding**.

Example vocabulary:


[a, n, cat, dog]


One-hot representation:

| Token | Vector    |
|-------|-----------|
a       | [1,0,0,0] |
n       | [0,1,0,0] |
cat     | [0,0,1,0] |
dog     | [0,0,0,1] |

Problems:

- Very **high dimensional**
- No **semantic meaning**
- All tokens equally distant

distance(cat, dog) = distance(cat, a)
Which is clearly wrong.

Hence, instead of one-hot vectors, we assign **dense vectors**:
```

Example (2D for simplicity): <br>
$a → [1.0, 0.5]$ <br>
$n → [0.8, 1.2]$ <br>

These vectors are **learned during training**.

During training:

1. The model predicts the next token
2. The loss is computed
3. Gradients update the embedding values

Eventually the vectors evolve into something meaningful.

Each dimension captures **latent features** like:

- animate vs inanimate
- food vs object
- abstract vs physical

But these features are **not explicitly labeled**.

So a language model is really learning the **geometry of words in high-dimensional space**.

They emerge during training.

---

## 6. Attention: Core Mathematics

Attention answers:

> Which previous tokens matter most right now?

```
Intuition:

Humans do this naturally.

Example sentence:

The cat sat on the ___

To predict the next word, you mentally focus on:

cat
sat
on

You don't care much about "the" earlier in the sentence.

Attention is the mechanism that lets a model decide what to focus on.

```

### 6.1 Queries, Keys, Values

Each embedding `x` is transformed:

$
Q = x W_Q,\quad K = x W_K,\quad V = x W_V
$

These are just matrix multiplications.

```
Intuition:

Instead of thinking about matrices, think about searching in a database.

| Concept | Intuition                             |
|---------|---------------------------------------|
Query     | what the current token is looking for |
Key       | what each previous token offers       |
Value     | the information content of the token  |

So the process becomes:

current token asks: what information do I need?
previous tokens answer: here is what I contain

```

Example for current token (`n`):

    Q = [0.9, 1.1]


Previous tokens have keys:

    K_BOS = [0.2, 0.1]
    K_a   = [0.7, 0.6]
    K_n   = [1.0, 1.2]

These keys represent what each token is about in vector space.

---

### 6.2 Similarity Scores

Similarity is computed using dot products.

```
Dot product intuition:

If two vectors point in similar directions, the dot product is large.

If they point in different directions, it is small.

```

> So this calculation answers:
**How relevant is this previous token to what I need right now?**

Compute similarity between query and each key:

$
\text{score}_t = \frac{Q \cdot K_t}{\sqrt{d}}
$

Example:

| Token | Score |
|------|-------|
| BOS  | 0.21  |
| a    | 0.92  |
| n    | 1.56  |

```
Interpretation:

BOS  → barely relevant
a    → somewhat relevant
n    → very relevant

So the model realizes:

the previous n matters most.

```
---

### 6.3 Attention Weights

Apply softmax:

Softmax Converts Scores Into Attention

Raw scores are not probabilities.

| Token | Weight |
|-------|--------|
| BOS   | 0.13   |
| a     | 0.33   |
| n     | 0.54   |

```
Now they sum to 1.

Interpretation:

54% attention → previous n
33% attention → a
13% attention → BOS

This means:

The model mostly looks at the previous n.
```

Interpretation:

> The model attends most strongly to the previous `n`.

---

### 6.4 Context Vector

```
Values Contain the Information

Each token also has a value vector.

Think of values as the information each token carries.

Example:

V_BOS
V_a
V_n

These vectors encode semantic content learned during training.

```

**Final attention output:**

$
\text{Attention}(Q,K,V) = \sum_t \alpha_t V_t
$

```
Which means:

context =
0.13 * V_BOS
+ 0.33 * V_a
+ 0.54 * V_n

So the final vector is a weighted blend of previous tokens.

Important idea:

The more relevant a token is, the more its information contributes.

```

This creates a context vector that blends relevant past information.

That vector is what the model uses to predict the next character.

```
The context vector is basically:

"What information from the past is most relevant for predicting the next token."

It is like the model summarizing the history.

Example interpretation:

context =
mostly information about previous n
some information about a
little information about BOS
```

---

## 7. Multi-Head Attention

```
Before attention (in RNNs), models had a single compressed memory of the whole sentence.

Attention instead lets the model directly look back at every previous token and decide how important each one is.

So instead of remembering everything, the model can focus selectively.
```

Instead of one attention operation, GPT uses several **heads**:

- Each head learns a different relationship
- Outputs are concatenated and mixed

This increases expressive power.

```
Intuition:

Earlier we saw **attention** as a mechanism that lets a token look back at previous tokens and decide which ones matter.

But there is a limitation if we only use **one attention operation**.

A single attention head can only focus on **one kind of relationship at a time**.

Language, however, contains **many different relationships simultaneously**.

Examples:

The boy who ate the apple was hungry

Different relationships in this sentence:

- grammar: **boy → was**
- action: **boy → ate**
- object: **ate → apple**
- structure: **who → boy**

One attention head might capture **syntax**, another **semantic meaning**, another **long-rane dependencies**.

```

```
## How Multi-Head Attention Works

Instead of computing attention once, the model computes it **multiple times in parallel**.

Each head has its own parameters:


Head 1: WQ1 WK1 WV1
Head 2: WQ2 WK2 WV2
Head 3: WQ3 WK3 WV3
Head 4: WQ4 WK4 WV4


Each head produces its own **context vector**.

Example:


head1_output = [0.5, 0.2]
head2_output = [0.1, 0.9]
head3_output = [0.7, 0.3]
head4_output = [0.2, 0.4]


These outputs are then **concatenated**:


[0.5,0.2 | 0.1,0.9 | 0.7,0.3 | 0.2,0.4]


And mixed using another linear layer.

---

## Intuition

Think of multi-head attention like **multiple experts analyzing the same sentence**.

Each expert specializes in something different:

| Head | What it might learn |
|-----|---------------------|
Head 1 | grammar |
Head 2 | semantic meaning |
Head 3 | long-range dependencies |
Head 4 | punctuation / structure |

The final representation combines all of these viewpoints.


```

```
But each head only outputs a small vector.

So the question becomes:

How do we combine these different perspectives?

Instead of averaging them, we stack them together.

Concatenation:

[0.5, 0.2 | 0.1, 0.9 | 0.7, 0.3 | 0.2, 0.4]

Which becomes:

[0.5, 0.2, 0.1, 0.9, 0.7, 0.3, 0.2, 0.4]

Why?

Because concatenation preserves information from each head separately.

If we averaged instead:

average = [0.375, 0.45]

we would destroy the distinctions between heads.

Concatenation keeps:

head1 information
head2 information
head3 information
head4 information

all available.

```

Now we have a big vector, but the model still needs to turn it into something useful.

So we apply another linear transformation:

$
y=W_O . concat(head_1, head_2, ..., head_h)
$

This layer is sometimes called the output projection.

Then the linear layer learns:

How should these signals be combined?

Example transformation:

$W_O \quad learns: 0.8 * grammar + 0.4 * semantic $+ 0.2 * long-range

to produce a new representation.

Without the linear layer, heads would remain independent.

The linear layer allows interaction between heads.

---

## 8. Feed-Forward Network (MLP)

After attention produces a **context vector**, the model applies a small neural network.

$
\text{ReLU}(xW_1)W_2
$

This is called the **feed-forward layer** or **MLP block**.

This:
- Adds non-linearity
- Enables complex transformations

### Explanation:

What This Means Step-by-Step

- First transformation:

$h = xW₁$

This projects the vector into a **larger space**.

Example:

x = [0.4, 0.8]

$W₁$ expands dimension
$→ h = [0.2, 1.1, -0.5, 0.7]$


- Apply ReLU

ReLU(x) = max(0,x)


Example:

$[0.2, 1.1, -0.5, 0.7]
→
[0.2, 1.1, 0, 0.7]$


Negative values are removed.

This introduces **non-linearity**, which lets the model represent complex patterns.

- Second Transformation

Now compress back down:

$output = hW₂$

Example:

$[0.2, 1.1, 0, 0.7]
→ [0.6, 0.3]$


```
It does three conceptual things:

- Expand the representation
- Apply a nonlinear filter
- Compress it back

Think of it like thinking harder about the information attention gathered.

After attention, the model has a vector representing the current token and its context.



Example:
Input Vector
x = [0.4, 0.8]

In real GPT models this might be:

x ∈ R^768

This vector contains information about the sentence so far.

--

First Transformation (Expand Space)
This multiplies the vector by a matrix.

Example:

x = [0.4, 0.8]

W1 expands dimension
2 → 4

Result:

h = [0.2, 1.1, -0.5, 0.7]
Intuition

This creates many new combinations of features.
Example transformation:

feature1 = 0.4*0.3 + 0.8*0.1
feature2 = 0.4*0.9 + 0.8*0.8
feature3 = ...

So the network is generating new candidate patterns.
Think of this step as:

"Let me consider many different interpretations of this context."

--

ReLU (Nonlinear Filter)
Why do this?

Without ReLU, the entire network would just be linear transformations.

Multiple linear layers collapse to one linear layer.

Meaning:

A(Bx) = Cx

which gives the model very limited power.

ReLU fixes this by creating nonlinear behavior.

Intuition

ReLU acts like a feature detector.

Example:

pattern detected → positive value → keep it
pattern absent → negative value → remove it

So the network is essentially asking:

which patterns are relevant here?

--

Second Transformation (Compress)
Now we reduce the dimension again.
So the network recombines the activated features into a final vector.

```


```
Step Intuition

Attention answers:

> What information from the past is important?

The MLP answers:

> How should we **transform that information** to make a prediction?

So:

Attention gathers information from other tokens.

The MLP then transforms that information.

Think of it like:

attention = retrieve information
MLP       = think about it
```

---

## 9. Backpropagation (Learning via Calculus)

The training objective is:

$
\min_\theta \sum_t -\log P(x_{t+1} \mid x_{\le t})
$

Using the **chain rule**:

$
\frac{\partial \text{Loss}}{\partial W} =
\frac{\partial \text{Loss}}{\partial \text{logits}}
\cdot
\frac{\partial \text{logits}}{\partial W}
$

The `Value` class computes this automatically.

```
This is **negative log likelihood**.

But the model has millions or billions of parameters.

How do we know how to adjust them?

We compute **gradients**.

---

## The Key Idea: The Chain Rule

Suppose:

Loss → logits → hidden layer → weights


We want to know:
how much does changing W affect the loss?


Using calculus:

dLoss/dW =
(dLoss/dLogits) *
(dLogits/dW)

This is called the **chain rule**.

---

## Intuition

Backpropagation works like **blame assignment**.

If the model predicted the wrong word:


The cat drank the ___

Correct answer:
milk

But the model predicted:
table

Backpropagation asks:
which weights contributed to this mistake?

Then it adjusts those weights slightly.

```


---

## 10. Parameter Update (Adam Optimizer)

Once gradients are computed, parameters are updated.


$
W \leftarrow W - \eta \cdot \nabla_W \text{Loss}
$

Where

- **η** = learning rate
- **∇W Loss** = gradient

This moves parameters in the direction that **reduces loss**.

Adam improves this by:
- Using momentum
- Normalizing gradient magnitude

```
## Why Adam Is Used

Adam improves basic gradient descent by:

1. **Momentum**

   Smooths gradients across steps.

2. **Adaptive scaling**

   Adjusts learning rate per parameter.

Intuition:

if a parameter keeps getting large gradients
→ reduce step size

if gradients are small
→ increase step size


This makes training **more stable and faster**.
```

---

## 11. Inference (Text Generation)

After training, the model generates text.

This process is **autoregressive**.

Meaning:
next token depends on previous tokens


---

Generation Loop:
1. Start with BOS <br>
``` <BOS> ```

2. Predict probability distribution<br>
Model predicts probabilities:

|  |  |
|-----|-----|
hello | 0.6 |
hi    | 0.3 |
hey   | 0.1 |

3. Sample next token<br>
Example:
hello

Now input becomes:
```<BOS> hello```

4. Repeat<br>

Creativity is controlled via **temperature**:

$
P_i \propto \exp\left(\frac{\text{logit}_i}{T}\right)
$


```
Temperature (Controlling Creativity)

Logits are scaled:

P_i ∝ exp(logit_i / T)

Where T is temperature.

Effects:
| Temperature | Behavior |
|-----|------------------|
hello | deterministic    |
hi    | balanced         |
hey   | creative/random  |


Example:

Low temperature:
The cat sat on the mat

High temperature:
The cat danced across the moonlit refrigerator

```

---

## 12. Final Mental Model

GPT is:

- A probabilistic next-token predictor
- Built from linear algebra + calculus
- Optimized by minimizing surprise

Formally:

$
\text{GPT} = \arg\min \sum -\log P(x_{t+1} \mid x_{\le t})
$

Everything else exists to make this probability estimate better.

```
A GPT layer works like this:

tokens
↓
embeddings
↓
multi-head attention
↓
feed-forward network
↓
logits
↓
softmax
↓
next token


Training adjusts parameters so that correct tokens receive high probability.

Over time the model learns the statistical structure of language.
```

---

## Key Insight

GPT does not *understand* language.

It models **statistical structure** extremely well.

Understanding emerges as a side effect of scale.
