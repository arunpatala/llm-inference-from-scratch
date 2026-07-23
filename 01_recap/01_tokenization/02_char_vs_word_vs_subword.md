# Why tokenization exists, and how it compares to character- and word-level

A neural network only consumes fixed-size numerical vectors, not strings.
Something has to turn text into a sequence of integer IDs, each indexing a
row in an embedding table. That table has to be finite (fixed number of
rows), but language is open-ended: new words, typos, code, numbers, other
scripts. So the scheme needs three things at once: a bounded vocabulary, a
guarantee that any input string is representable, and sequences short enough
to be computationally affordable. Attention cost grows quadratically with
sequence length, KV cache grows linearly, so tokens-per-unit-of-content is a
real cost knob, not just a detail.

## The three approaches, compared

| | Character/byte-level | Word-level | Subword (BPE) |
|---|---|---|---|
| Vocab size | Tiny (~256 for bytes) | Huge (100k+ for decent coverage) | Tunable: you pick it (Qwen3: ~151k) |
| Sequence length | Longest: every character is a token | Shortest: one token per word | In between: common words are 1 token, rare ones split into a few |
| OOV handling | None needed: every string decomposes into its bytes, all in-vocab | Real problem: anything outside the fixed list becomes a single generic `<UNK>` | None needed (byte-level variant): an unseen word falls back to more, smaller known pieces, down to raw bytes worst case |

## Why character-level loses

No OOV problem, but a word like "tokenization" becomes about 12 tokens
instead of 2-3. That's 4-6x more attention computation and KV cache for the
same content, and each individual token carries almost no meaning on its
own, so the model has to spend depth and capacity just reconstructing words
from letters before it can do anything else with them.

## Why word-level loses

Short sequences, but the out-of-vocabulary handling is the real problem. Any
word not in the fixed vocab (a typo, a rare technical term, a name, "running"
if only "run" was included) collapses to the same `<UNK>` ID as every other
unknown word. The model can't distinguish them; the information about what
was actually written is gone. Morphological variants ("run" / "runs" /
"running" / "runner") each need their own vocab slot unless something
smarter is done, which is part of what pushes word-level vocabularies so
large to begin with. Even the rare words that do make it into vocab appear
too infrequently during training to get a well-trained embedding, which is
the same mechanism behind the glitch-token phenomenon (question 16 in
`01_questions.md`).

## Why subword/BPE wins, specifically byte-level BPE

Byte-level BPE is what GPT-2, GPT-4, Llama, and Qwen all use. It's built
bottom-up from individual bytes, iteratively merging the most frequent
adjacent pair up to a target vocab size. Common whole words end up merged
into a single token, since they're frequent enough to earn it; rare or
unseen words just don't get fully merged and fall back to smaller pieces,
worst case individual bytes. That gives word-level's short sequences for
the common case and character-level's "nothing is ever truly unknown"
guarantee for the uncommon case, at a vocabulary size chosen deliberately
rather than one dictated by how many words happen to exist.

This is the direct answer to question 11: the byte-level fallback isn't a
nice-to-have, it's what makes "no OOV, ever" a hard guarantee instead of a
best-effort one.
