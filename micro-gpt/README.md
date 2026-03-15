# micro-gpt - GPT from scratch

~ https://x.com/karpathy/status/2021694437152157847

~ https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95


Karpathy's pure-Python GPT: autograd engine, transformer, tokenizer, training & inference — in ~200 lines of code with zero dependencies.


# The Big Picture
This code implements an entire GPT language model — the same fundamental architecture behind ChatGPT — in pure Python with zero external libraries. **No frameworks. No PyTorch. No NumPy. Just raw Python and math**. It includes an autograd engine, a transformer, a tokenizer, a training loop, and inference. All from scratch.

- Zero dependencies (pure python)
- ~200 lines of code
- GPT architecture

## What it does
It trains a tiny language model to:

    Predict the next character in a sequence.

Specifically, it learns from a dataset of names and then generates new names that look similar.

If it sees:

    A n n a


It learns patterns like:
- After “A”, often comes “n”
- After “nn”, often comes “a”
- Names often end after certain patterns

That’s it. That’s the entire intelligence.


# 1. Setup & Data
```py
if not os.path.exists('input.txt'):
    # download a file of names
docs = [l.strip() for l in open('input.txt').read().strip().split('\n') if l.strip()]
random.shuffle(docs)
```

- The program needs some text to learn from. Here it uses a list of names.
- It reads all the names from a file and shuffles them randomly.
- Imagine you want a kid to learn spelling by looking at a book of names — this is the book.

# 2. Tokenizer
```py
uchars = sorted(set(''.join(docs))) # unique characters
BOS = len(uchars) # special start token
vocab_size = len(uchars) + 1
```

- The model cannot understand letters directly. It needs numbers.
- Each unique character (like 'a', 'b', 'c') gets its own number.
- There’s a special number BOS (“Beginning of Sequence”) to mark the start of a name.

💡 Think: The model sees “Alice” as [BOS, 0, 11, 8, 2, 4, BOS] instead of letters.

# 3. The Value Class (Autograd)
- This part is fancy math for teaching the computer how to learn from mistakes.
- Every number in the model (weights, intermediate values) is wrapped in a Value object.
- `Value` remembers how it was calculated so that later it can figure out how to tweak itself to be better.
- `backward()` is like asking every part of the model: “how should I change to improve?” This is how learning happens.

💡 Think: Imagine each Lego block knows how it contributes to the final structure, so you can adjust blocks to fix a crooked tower.

# 4. Model Parameters (Knowledge Storage)
```py
state_dict = {'wte': ..., 'wpe': ..., 'lm_head': ...}
```

- These are the “knobs” the model can turn to learn. These are just matrices filled with random numbers at first.

They represent:
- `wte` = token embeddings → how each letter is represented.
- `wpe` = position embeddings → remembers where the letter is in the name. [“Ana” and “naa” use same letters but different order.]
- `lm_head` = final layer → predicts the next character.
- There are also attention and MLP layers.

💡 Think: This is like a blank sheet of Lego instructions — initially random, but the model will tune it by learning from examples.

# 5. Linear, Softmax, RMSNorm
- `linear(x, w)` → simple math to mix numbers.
- `softmax` → converts a list of numbers into probabilities (how likely -each next letter is).
- `rmsnorm` → keeps numbers at reasonable sizes so the model doesn’t get -“confused” by too big or small numbers.

💡 Think: These are tools the model uses to combine what it knows and make decisions.

# 6. The GPT Function (How the Model Thinks)
```py
def gpt(token_id, pos_id, keys, values):
    ...
    return logits
```

This is the brain of the model. Here’s what happens step by step:<br>

1. Look at the current letter (`token_id`) and where it is in the name (`pos_id`).

2. Embed them into numbers (`tok_emb` + `pos_emb`).

3. **Attention block:** Each letter asks: “Which previous letters are important for predicting the next letter?” This is called self-attention.

    For example:

    In “Anna”:
    - The last “a” might attend strongly to “nn”

    Attention works by:
    - Computing similarity scores
    - Turning them into probabilities
    - Mixing previous information weighted by importance

    This allows the model to look backward at the sequence.

4. **MLP block (tiny math brain):** Mixes information from attention, adds non-linearities, and improves predictions.

5. **Output:** produces `logits`, which are scores for each possible next letter.
    ```py
    logits = linear(x, lm_head)
    ```
    This produces a score for each possible next character.
    
    Then:
    ```py
    softmax(logits)
    ```
    Converts scores into probabilities.

    Example:
    ```
    a → 0.60
    b → 0.05
    c → 0.01
    ...
    ```
    The correct next letter should have high probability.

💡 Think: This function reads a partial name and guesses what letter should come next.


# 7. The Loss — Measuring Wrongness
```py
loss_t = -probs[target_id].log()
```

If the correct letter had probability:
- 0.9 → small loss (good)
- 0.1 → large loss (bad)

Loss measures:
    
    “How surprised was the model by the correct answer?”

Training tries to minimize this loss.


# 8. Backward Pass — Learning
```py
loss.backward()
```

This triggers the chain rule across the entire computation graph.

It calculates gradients:

    For each parameter: how much should it change?

This is pure calculus applied programmatically.


# 9. Optimizer (Adam)
After gradients are computed:
```py
p.data -= ...
```

Weights are updated.

Adam:
- Uses momentum
- Smooths updates
- Adapts learning rate per parameter

This makes training stable and efficient.

- Remembers past mistakes.
- Adjusts faster if a mistake is repeated.
- Helps the model learn efficiently without getting stuck.

💡 Think: Adam is like a tutor who remembers what letters the kid keeps messing up and focuses practice there.


# 10. Training Loop (Learning Phase)
```py
for step in range(num_steps):
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    ...
    loss.backward()
    # update weights with Adam
```

For each training step:
1. Pick one name.
2. Predict next character at each position.
3. Compute average loss.
4. Backpropagate.
5. Update weights.

Repeat 1000 times.

Over time:
- Loss decreases
- Predictions improve
- Patterns are internalized

💡 Think: Like a kid practicing spelling: look at “Alice”, guess each next letter, see mistakes, adjust memory, try again.


# 11. Inference / Generating Names
```py
temperature = 0.5
for sample_idx in range(20):
    ...
    token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
```

Then repeatedly:
1. Feed current token into model
2. Get probability distribution
3. Sample next token
4. Repeat

Until BOS appears again (end of name).

This generates new names like:
```
Anira
Joviel
Marionel
```

These were never in dataset — but follow learned patterns.

- `temperature` controls creativity:
    - Low → safer, more predictable names.
    - High → more unusual, random names.
- The model generates names letter by letter, using what it learned.

💡 Think: Kid now can spell names they never saw before by combining patterns they learned.

# Summary

This code demonstrates that GPT is fundamentally:
- Embeddings
- Attention
- MLP layers
- Backpropagation
- Optimizer

That’s it.

Everything else in large models is:
- Scale
- Efficiency
- Engineering

This file is the core algorithm in its most stripped-down form.