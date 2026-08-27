# ShopSense — Sequential Deep Learning (SASRec) Notes

This documents the SASRec addition on top of the item-item CF system
(separate from the main README — that one stays as-is). Covers what the
sequential part actually does, and the specific design problems it had to
solve to work correctly on this dataset.

## What the sequential part is doing

Item-item CF answers "what's generally similar to what this user has
interacted with, across all of history." It has no idea what order things
happened in, and no idea what the user is doing *right now* versus six
months ago.

SASRec answers a different question: "given the exact order of items in
this user's current session, what comes next?" It reads a session as a
sequence and uses self-attention (the same mechanism a language model uses
to predict the next word) to predict the next item, one position at a time.
Concretely, in this notebook:

1. **Sessions are built from `train_events`** (the same leakage-free split
   used for CF) — a burst of one visitor's activity with no gap longer than
   30 minutes counts as one session.
2. **The model** is a small transformer: item embeddings + positional
   embeddings, fed through a causal (left-to-right only) attention encoder,
   trained with one real "next item" and one random negative per position.
3. **It's used as a re-ranker, not a replacement.** Item-item CF still
   generates ~200 broad candidates. SASRec looks at the user's most recent
   session and reorders just those candidates based on sequence context. If
   there's no session to look at, it falls back to plain CF order — SASRec
   never removes a recommendation, only reshuffles one when it has
   something to say.

This is why it's evaluated as **"Item-Item CF + SASRec"** against
**"Item-Item CF alone"** in the comparison table, rather than as a
standalone model — the honest question is "does adding sequence awareness
improve on CF," not "is SASRec better than CF in isolation."

## Problems the design has to handle, and how the code handles them

These aren't hypothetical — each one, if skipped, produces either a silent
correctness bug or a crash. Documenting them because recognizing *why* each
piece of code exists is the part that's actually worth remembering.

### 1. Variable-length sessions can't feed a fixed-size model

**Problem:** a transformer needs a fixed sequence length, but real sessions
range from 2 items to over `MAX_SEQ_LEN`.

**Solution:** pad short sequences with a reserved index `0`, and truncate
long ones to the most recent `MAX_SEQ_LEN` items (`seq[-(max_len+1):]`).
Real items are shifted `+1` everywhere specifically so index `0` is free to
mean "no item here" — nothing else in the vocabulary can collide with it.

### 2. The model must not be able to "see the future" within a session

**Problem:** if the model could attend to positions *after* the one it's
predicting, it would trivially cheat — the causal-masking equivalent of the
train/test leakage bug from the CF side, just inside a single sequence
instead of across time.

**Solution:** the transformer encoder is causal (left-to-right only), so
position *t* only ever attends to positions ≤ *t*. Same underlying
principle as the leakage-free CF split — no information from later in the
sequence reaches earlier predictions.

### 3. Negative sampling can accidentally pick a real positive

**Problem:** the loss needs a "negative" item the user didn't actually go
to next, to contrast against the real one. Sampling a random item naively
can occasionally re-pick an item that genuinely appears elsewhere in that
same sequence, quietly corrupting the training signal.

**Solution:** the negative sampler excludes every item that appears
anywhere in that sequence (`target_set`) before accepting a candidate, so
the "negative" is guaranteed to actually be one the user didn't interact
with in that session.

### 4. Padding positions must not contribute to the loss

**Problem:** padded positions aren't real predictions — if the loss
function treats them like real ones, the model partly learns to predict
padding, which is meaningless and drags down real signal.

**Solution:** `mask = (target_seq != PAD)` zeroes out loss at every padded
position before the sum, so gradients only flow from positions that
correspond to a real "next item."

### 5. Immediate repeats add no sequential signal

**Problem:** a user viewing the same item three times in a row (page
refreshes, indecision) isn't three different sequential decisions — feeding
it as-is trains the model to just predict "same item again," which is a
degenerate, uninteresting pattern to learn.

**Solution:** consecutive duplicate items within a session are collapsed
before building sequences (`item != seq[i-1]`), so the model only sees
genuine transitions between different items.

### 6. Not every user has a usable session for re-ranking

**Problem:** at inference time, some visitors have thin or no session
history — a real re-ranker can't assume every user has enough context.

**Solution:** `sasrec_score_candidates()` returns `None` when there's no
session for that visitor, and `cf_sasrec_rerank()` falls back to plain CF
order in that case. The system degrades gracefully instead of failing or
returning nothing.

## What's still an open, honest question

Whether SASRec re-ranking should actually ship depends entirely on whether
**"Item-Item CF + SASRec" beats "Item-Item CF alone"** in your Step 12
comparison table, run against your real data. This notebook is built to
measure that honestly — both are evaluated through the identical
`evaluate()` harness — but which one wins isn't decided in the code, it's
decided by the numbers you get when you run it. If plain CF wins, that's a
legitimate finding: the fix in that case is more session data or a longer
training run, not just keeping SASRec anyway because it's the fancier
model.
