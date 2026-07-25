# Tokenization — Future Directions (chapter subsection draft)

Condensed from Socratic sessions reasoning from the interview foundation
(`../01_questions.md`) toward the research frontier (`frontier_questions.md`).
Each direction is written the way the author reasoned to it, then grounded with
the verified survey. This is a *draft subsection* for the end of the tokenization
chapter, not raw notes.

The organizing frame the author arrived at: **the whole history of tokenization
is really one question — what is the model's atomic unit of computation? — and
the frontier is drifting from "a fixed, discrete token from a frozen table"
toward "a learned, latent representation."**

## 1. From fixed discrete tokens to learned latent units

Almost every hard problem in this chapter traces to one root decision: imposing
**hard, fixed boundaries** at all. Fertility (Q24), domain bad-fit (Q29), token
healing (Q23), glitch tokens (Q16), the multilingual tax (Q24), frozen-for-life
(Q10/Q29) — all are downstream of a vocabulary learned once, for compression, and
then frozen.

The deeper critique the author landed on: tokenization is a **separate phase
optimized for the wrong objective**. BPE minimizes token count (compression,
frequency) — a *proxy* that is disconnected from the model's actual goal (task
performance). The two phases never share a loss. This is not just intuition: it
is the empirically-confirmed headline of Meta's *Compute Optimal Tokenization*
(2605.01188) — the **BPE-optimal compression rate is not the model-optimal one**,
and optimal compression even *decreases* as compute grows.

The proposed escape is "best of both worlds": operate on **raw bytes** (no fixed
vocab, so no fertility, no domain bad-fit, no glitch tokens, and the byte-floor
robustness of Q11 for free) but **group bytes into variable-size chunks** so you
are not paying one forward pass per byte (avoiding the char-level inefficiency of
Q2). That target requires a new architecture, and two exist:

- **Byte Latent Transformer (BLT, 2412.09871)** places boundaries by a *signal*
  the model already has (Q5): next-byte **entropy**. Predictable runs (low
  entropy) get swallowed into big patches; surprising bytes (high entropy) get
  fine-grained attention. Compute flows to where the information is.
- **H-Net (2507.07955)** *learns* the boundaries end-to-end, directly closing the
  objective mismatch above — segmentation optimized for performance, not
  compression.

Architecturally these are **latent transformers**: a small local encoder maps
bytes → patch vectors, a big **latent transformer** does the heavy work over
patches (few, large units), and a small local decoder maps patch → bytes. The
"latent" is literal — the expensive computation happens in the compressed patch
space, with byte encode/decode as the on-ramp and off-ramp. At matched training
FLOPs, byte/latent models now *match or beat* BPE on quality and are far more
robust to character perturbation. **Quality is no longer the blocker.**

### The inference problem (why this book cares)

Inference speed is the blocker, and the author reasoned to exactly why: even
though the big model runs **once per patch** (amortized over ~4-6 bytes via the
cheap local decoder), you still emit bytes one at a time, so there are **more
autoregressive decoding steps** plus the memory-bandwidth cost of the local
decoder per byte. That residual is the whole target of 2026 work: **Fast-BLT**
(2605.08044) adds a block-wise **diffusion** objective to emit *multiple bytes
per step*, plus self-speculation, to attack the decoding-steps problem head-on.

### What it does to the serving stack we build in this book

If boundaries are decided **dynamically per input**, the fixed-token assumptions
under KV cache, prefix caching, and speculative decoding are all in question:

- **Decode loop (Q5)** becomes two-level: a patch-level loop wrapping a byte-level
  loop. The KV cache is over **patches, not tokens**.
- **Speculative decoding (Q19)** loses its "shared vocabulary" footing — there is
  no vocabulary; draft/target agreement has to be redefined over patches or
  bytes.
- **Prefix caching (Q28)** is the subtle one, and the author's analysis lands on
  a genuinely open frontier question. Prefix caching survives **if and only if
  patching is causal.** BLT's entropy model is causal (a boundary at position i
  depends only on bytes before i), so the same prefix deterministically produces
  the same patches → cache hits. And there is a non-obvious *upside*: causal
  patching would be **prefix-consistent**, which BPE is *not* (Q27's
  non-compositionality — `tokenize("New Ent")` is not a prefix of
  `tokenize("New Enterprise")`). So causal dynamic patching could be *more*
  cache-friendly than fixed BPE, fixing the very non-compositionality that forces
  token healing (Q23). The threat is **bidirectional/learned** boundary decisions
  that look at the whole input: adding text elsewhere could shift earlier
  boundaries and re-patch the prefix → cache miss. During generation patching is
  causal anyway; for prefill the choice matters. Nobody has shipped this at scale,
  so the real prefix-cache hit rate for dynamic tokenization is unknown — one of
  the field's open questions.

### The production reality check

Nobody ships tokenizer-free at the frontier yet. The clearest signal is Gemma 4
(April 2026) moving to a **larger** 262K BPE-with-byte-fallback vocab — the
opposite direction. So the honest framing for the chapter: byte/latent is a
credible successor that has won on quality and robustness, is losing on inference
speed, is being repaired with diffusion + speculation, and has an unresolved
serving story (especially prefix caching under dynamic boundaries).

### The broader thread (flagged for a follow-up research pass)

Byte-latent is one instance of a larger drift — moving the unit of computation
off the fixed discrete grid into a learned/latent space. Adjacent directions:
**Large Concept Models** (compute over sentence-level concept embeddings) and
**latent reasoning** (reason in continuous latent space between tokens). Common
thread: *what granularity — discrete or continuous, fixed or learned — should the
model actually think in?* [Unverified in this session's survey; needs a focused
research pass before it goes in the chapter with citations.]

---

## 2. The opposite bet — supersizing the vocabulary (and the fracture)

While the byte-latent camp tries to *remove* the tokenizer, a second camp goes
the other way: make the vocabulary much bigger. Production sits with this camp —
Gemma 4's 262K vocab, and the 4-8x growth in frontier vocab sizes 2023->2025.

The author reasoned in from Q17: since the embedding matrix is a *shrinking
fraction* of the model as it scales (26% at 0.6B, ~8% at 8B), a bigger model has
headroom to grow the vocab back. That is the **"larger models deserve larger
vocabularies"** scaling law (Tao et al., NeurIPS 2024, 2407.13623) — optimal
vocab grows *sublinearly* with non-vocab params; Llama-2-70B "should" have used
~216K instead of 32K; most models are under-vocabularized. And it is not just
affordable but beneficial (Q20): bigger vocab -> fewer tokens per text -> shorter
sequences -> cheaper attention/KV and more content per context.

### The fracture (an unresolved frontier debate)

A 2026 Meta paper, *Compute Optimal Tokenization* (2605.01188), points the
opposite way: as compute grows, the **optimal amount of compression decreases**,
and data should be measured in **bytes, not tokens**. Reconciliation (best
synthesis; the field has *not* settled this — the two results come from
overlapping Meta author groups):

- They measure different regimes. "Bigger vocab" varies model size along the
  compute-optimal frontier — a capacity-rich large model wants more compression
  (bigger vocab, shorter sequences). "Less compression" lives in the
  *over-trained* regime — a tiny model trained far past Chinchilla (exactly
  Qwen3-0.6B: ~596M params, trillions of tokens, for inference-cheap deployment).
  In that data-rich / capacity-poor regime, *less* compression can win: smaller,
  more-repeated units give an abundant-data model more, simpler examples, while
  high compression makes each token rarer and harder to train (the Q16
  undertrained-token pressure at the margin).
- So the rule is joint: **optimal vocab depends on model size AND how
  over-trained the model is.** Capacity-rich -> more compression; data-rich small
  -> less.

Two sharpeners, both connecting back to the book:

- **Bytes-not-tokens removes the Q20 confound.** "Tokens" is a tokenizer-
  dependent unit — the same data is a different token count under a different
  tokenizer — so scaling in bytes strips out a moving target, and under the byte
  lens model-optimal compression falls with compute.
- **"Vocab size" is not one number (Over-Tokenized Transformer, ICML 2025,
  2501.16975).** Decouple *input* vocab (cheap n-gram lookup, helps all sizes)
  from *output* vocab (expensive softmax, helps big models, hurts small ones).
  Scaling input vocab to ~12.8M n-gram entries let a 400M model match a 1B
  baseline. Once split, part of the contradiction dissolves: a small model can
  carry a huge *input* vocab and a modest *output* vocab.

Also in this camp: **SuperBPE** (2503.13423) merges subwords across whitespace
into "superword" tokens — a rare simultaneous win on token count (-33%),
inference compute (-27%), and task accuracy (+4.0% at 200K vocab).

Chapter takeaway: there is no settled optimal-vocab law. Safe framing —
*optimal vocab depends jointly on model size and over-training, input and output
vocabularies should probably be sized separately, and the "measure in bytes"
reframing (Q20) is quietly the deeper point.*

## 3. Tokenization x reasoning — the "strawberry" problem is a tokenizer defect

The canonical failure: models miscount the r's in "strawberry." Grounded on
Qwen3-0.6B: "strawberry" -> ['str', 'aw', 'berry'] (three opaque tokens, the 3
r's split 1+0+2 across them), and " strawberry" (leading space) -> a SINGLE token
['Ġstrawberry']. The model receives atomic vectors, not letters — to count r's it
would have to have memorized each token's letter content. There is no
compositional character access (Q27).

So this is a tokenization failure, not a reasoning failure: the counting is
trivial, the model is blind to the characters. When it succeeds it is recalling a
memorized spelling, not counting — hence the unreliability. The one-token
" strawberry" case is worst: zero visibility into any letter.

The fix closes the loop with §1: character/byte-level access. Byte-latent models
(BLT/H-Net) see the bytes, so counting is direct — which is exactly why the
survey found them dramatically more robust on character-level tasks (H-Net 42.8
vs 22.2 on perturbed HellaSwag). §1 dissolves the strawberry problem for free.

The satisfying connection to Q14: numbers have the identical problem. If "1234"
were one opaque token the model couldn't do digit-wise arithmetic, so per-digit
tokenization (Q14) is precisely the "give the model character-level access for
the one domain where it really matters" carve-out. Strawberry is the same problem
for letters — but you cannot afford to char-split every word (Q2 inefficiency),
so digits get the exception and every other character-level task pays the
"strawberry tax."

The digit story has its own live debate: L2R 3-digit chunking (Llama 3, GPT-4o)
destabilizes place value, and forcing right-to-left alignment gives large
systematic arithmetic gains ("Tokenization counts", Singh & Strouse, 2402.14903)
— yet from-scratch training finds per-digit beats chunking outright ("Number
Cookbook", 2411.03766). So R2L is a good patch for existing tokenizers; per-digit
may win in a redesign. Contested.

Frontier framing: a chunk of apparent "reasoning" failure is really a
tokenizer-layer defect ("Counting Ability of LLMs and Impact of Tokenization",
2410.19730; "Say Anything but This: When Tokenizer Betrays Reasoning", 2601.14658
— 8-artifact taxonomy, "phantom edits"). The chapter's deepest tension surfaces
here: tokenization exists FOR efficiency (don't process char-by-char, Q2), but
that same efficiency HIDES the character structure some tasks need. Digits get a
carve-out; byte-latent dissolves the whole tension but pays at inference (§1).

### 3a. The general class behind strawberry: sub-token blindness

Strawberry is one instance of a general failure class — **sub-token blindness**:
any task needing structure *below* the token level is hard because the model sees
opaque token vectors, not what's inside them. The catalog (all same root):
- Character-level: spelling, reversing a string (models jumble a whole-token
  string like "DefaultCellStyle" but succeed if it's pre-split into chars),
  double-letter detection, character-position queries, anagrams, syllable/rhyme,
  word length. Benchmarks: CharBench (2508.02591), SubTokenTest (2601.09089).
- Numeric: arithmetic (Q14), multi-operand lookahead (2502.19981), and the famous
  "9.11 > 9.9" comparison error — but that one is MIXED (partly tokenization,
  partly contextual: 9.11 reads as a date or software version), so the chapter
  must not over-attribute it to tokenization.
- Already in the interview: glitch tokens (Q16), multilingual tax (Q24),
  adversarial tokenization (Q26), token healing (Q23), whitespace (Q12).
The unifying point: these are one failure mode in many hats — the tokenizer
trades sub-token visibility for efficiency (Q2), so every task needing to look
inside a token suffers. Digits get a carve-out (Q14); byte-latent (§1) dissolves
the class; everything else pays the "strawberry tax."

### 3b. A multimodal instance (author's lived example): VLM OCR "correcting" rare names

Observed: a vision-language model doing OCR "corrects" a rare email/name to a
more common one; separating the input by character makes it read correctly.
Classification: MIXED, like 9.11 — but tokenization is a genuine co-cause, and
the character-split fix *working* is the proof (if it were purely visual, pixel-
identical splitting couldn't help).
- Dominant root: VLM language-prior-over-vision imbalance. Documented — the LM is
  much larger/stronger than the vision encoder; "attention aggregates the visual
  evidence, the FFN at critical layers injects language priors that override it."
  The vision encoder sees the rare name; the LM overwrites it with a common one.
  (Seeing is Believing?, 2506.20168; Reading or Guessing?, 2605.27750; FADE,
  2606.29431; When Language Overwrites Vision, 2605.08245.)
- Tokenization co-cause + lever: the rare name is a low-prior token sequence,
  exactly what the FFN prior overrides; character-level prompting is a documented
  fix ("explicitly specifying character details helps identify all characters";
  "token embeddings encode character-level info, especially in larger models").
  Splitting strips the whole-word frequency prior so the correctly-perceived
  pixels get through — the Q14 carve-out applied to OCR. (Broken Tokens?,
  2506.19004.)
- Same frequency-prior/sub-token family as strawberry + Q14, in multimodal form.

## 4. (next direction — to be added)
