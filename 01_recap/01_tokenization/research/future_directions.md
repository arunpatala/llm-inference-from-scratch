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

## 4. The tokenizer as an economic + safety attack surface

The fourth security thread (after Q6 injection, Q16 glitch tokens, Q26 GCG), and
the newest: 2026 work reframes the tokenizer as an *economic* and *safety*
surface, not just a quality knob.

### Safety: the canonical-vs-fragmented gap (a token-boundary jailbreak)

Mechanism, built from the interview: refusal/safety training is done on TEXT,
tokenized the CANONICAL (greedy BPE) way, so refusal is keyed to canonical token
patterns. "Breaking Safety at the Token Boundary" (2607.01239) audited alignment
data and found ZERO fragmented harmful prompts — safety only ever saw the
canonical slice of token-space. Force a harmful word to fragment into a
non-canonical tokenization (Q27: one string, many valid token sequences) and the
refusal trigger never fires — a fragmentation optimizer flips first-token refusal
on 80-100% of refused HarmBench prompts.

Why it works is a vicious asymmetry between two things we established:
- Capability is robust to re-tokenization ("Broken Tokens", 2506.19004: up to
  93.4% retained under non-canonical segmentation; marginalizing over
  tokenizations can even beat canonical decoding, 2506.06446). The base model,
  trained on all of text, still "reads" the fragmented word.
- Safety is brittle to it — trained only on the canonical slice.
So fragmentation preserves meaning but destroys refusal. In our terms it's the
Q12 (extra-space) / Q23 (token-healing) off-distribution-boundary logic,
weaponized: capability trained on all of token-space, safety on a sliver, attack
the gap.

Why the obvious fix fails: SFT-on-fragments does NOT generalize (combinatorially
many fragmentations, Q27), so it's whack-a-mole. The real fix is canonicalization
before the safety check (or safety at the byte/normalized layer) — but that
COLLIDES with the robustness finding (models benefit from non-canonical
flexibility). Safety wants one canonical form; robustness wants many. Unresolved.

### Economic: per-token billing fraud

"Token Inflation" (2605.30040): per-token pricing + hidden reasoning tokens +
tokenization ambiguity lets a dishonest provider over-report usage undetectably
(~1,469% average inflation; ~50% "from tokenization ambiguity alone" below
detection thresholds). It is Q20 turned adversarial — everything is denominated
in tokens, but the token is an ambiguous, provider-controlled unit, so it becomes
a TRUST problem. Fix directions: verifiable billing (TEE attestation /
cryptographic proofs). Related position: "Stop Taking Tokenizers for Granted"
(2601.13260) argues the tokenizer is a supply-chain surface needing
pre-deployment audits.

Takeaway: adversarial tokenization is not just "re-optimize a suffix" (Q26) — it
is a STRUCTURAL gap between where safety lives (canonical token-space) and where
capability lives (all of token-space), plus an ambiguity that can be monetized.

## 5. Token granularity as a runtime resource-allocation knob (the inference-native one)

We treated granularity as fixed at tokenizer-training time. The newest, most
inference-relevant idea: make it a RUNTIME knob — spend fine tokens where meaning
is dense, coarse/merged tokens where it is redundant — jointly optimized against
KV-cache memory (Q30: KV scales with token count).

How fine-vs-coarse is controlled (4 mechanisms, by where in the stack):
- Semantic density (SemToken, 2508.15190): a lightweight encoder scores
  meaning-per-token; low density -> merge into coarse super-tokens, high -> keep
  fine. 2.4x fewer tokens, 1.9x speedup.
- Predictability / entropy (BLT-style, from S1): confident -> coarsen, surprised
  -> fine.
- Attention importance (KV-side): rarely-attended spans coarsened; SeKV
  (2606.31145) stores coarse semantic spans and reconstructs token detail on
  demand ("zoom in"), -53% GPU memory at 128K; sub-token routing (2604.21335)
  goes below the token, keeping selected groups of each value vector.
- Learned budget allocation: an optimizer distributes a fixed token budget across
  regions (AdapTok ILP for video, 2505.17011).

Common theme: a cheap signal (density / entropy / attention / learned predictor)
decides per-region whether to spend tokens finely or coarsen, applied at the
input, in the KV cache, or in the model's patching.

Where: coarsen low-information/low-attention parts (boilerplate, repeated context,
already-summarized history, whitespace); keep fine the high-information parts
(recent generation frontier, the actual question, heavily-queried spans). What it
buys (Q30): fewer KV entries for coarse parts -> less KV memory -> longer contexts
fit, cheaper serving; biggest win at small KV budgets / long context.

Also here: Incremental BPE (ICML 2026 Spotlight, 2605.30813) — a streaming
tokenizer emitting tokens as boundaries are determined, O(n log^2 t), ~3x over HF
tokenizers — tokenizer SPEED as an inference bottleneck, straight at Q9/Q30's
input-side cost.

Framing: granularity stops being a fixed tokenizer decision and becomes a runtime
resource-allocation problem against KV memory — the most inference-native idea in
the survey, and directly relevant to this book's later paged-attention / long-
context modules.

## 6. Beyond text: cross-domain "fixed segmentation is wrong", and the continuous frontier

Zooming out past language, multiple scientific domains independently reached the
same verdict in 2025-2026: fixed BPE-style segmentation is the wrong prior.
- DNA: DNAChunker (2601.03019) learns mutation-resilient adaptive segmentation
  and argues all fixed DNA tokenizations are wrong; Evo 2 (40B) goes the other
  way — single-nucleotide/byte-level, no learned tokenizer at all.
- Protein: GeoBPE (2511.11758) does geometric byte-pair encoding of 3D backbones,
  >10x bits-per-residue reduction with interpretable, function-aligned tokens.
- Time series: WaveToken wavelet-decomposes then quantizes; a frequency/pattern
  vocab beats per-sample tokenization.
- Code: TokDrift (2510.14972) — semantically identical code tokenizes differently
  under formatting/renaming because BPE boundaries ignore grammar, shifting
  behavior across 9 code LLMs (a domain instance of Q27 non-compositionality +
  Q29 bad-fit).

The standout, and the deepest idea: The Geometric Alignment Tax (2604.04155).
For CONTINUOUS domains, discrete tokenization forces continuous structure through
a "categorical bottleneck", and — counterintuitively — FINER quantization makes
the geometry WORSE; a continuous prediction head cuts distortion up to 8.5x. So
for continuous data the problem is not the vocab size, it is discreteness itself.

BitTokens (2510.06824) is the number-domain version and the direct sequel to Q14:
instead of splitting a number into per-digit tokens, encode the whole number as
ONE token via its IEEE-754 float bits — a structural encoding of value, not
frequency — and small LLMs then learn near-perfect arithmetic. Q14 gave the model
character-level access to digits; BitTokens gives it the value's bit structure
directly. (Yet frontier models keep shipping inconsistent BPE digit splits — the
gap between "solved in isolation" and "shipped".)

The generalizing frame for the chapter: BPE assumes the world is discrete symbols
where FREQUENCY equals meaning. That is roughly true for text, but false for
continuous scientific data (geometry, signals) and for structured domains
(code grammar, DNA codons, numeric place-value) whose real units the
frequency-merge heuristic ignores. So the tokenization question generalizes to:
"what is the right atomic unit for THIS domain's structure — and should it be
discrete at all?" For continuous domains the frontier answer is increasingly
"a continuous prediction head, no vocabulary."

## 7. Soft tokens: the input is a vector sequence, not a token sequence

The insight (author's connection, from the image-tokenization interview): a
transformer does not eat tokens, it eats a sequence of VECTORS. Where each vector
comes from is a choice — discrete tokenization (text -> ID -> embedding-table
lookup) is only ONE way to fill the sequence. Other paths produce the input
vectors DIRECTLY with an encoder, skipping the discrete vocabulary entirely:
- text -> token-ID lookup (classic tokenization),
- image -> ViT patch encoder (the vision "vectorizer", Q1 / image-tokenization Q1),
- long context -> a text encoder that compresses it (soft context compression),
- a retrieved doc / memory -> an encoder (e.g. xRAG compresses a doc to one vector).

Soft context compression is the direct TEXT-analog of the image vectorizer.
Encoder-decoder "Latent Context LMs" (2606.09659): a 0.6B encoder maps a long
token sequence to a SHORTER sequence of continuous latent embeddings (ratios 1:4,
1:8, 1:16); a 4B decoder consumes them. Continuous embeddings, not discrete IDs;
the goal is long-context / KV-cache efficiency (KV grows with context length). The
family: gist tokens (distill a prompt into a few vectors, ~26x), AutoCompressor
(summary vectors as soft prompts), ICAE (memory slots), 500xCompressor (compress
into KV values). Umbrella (Prompt Compression survey 2410.12388; Long-Context
survey 2503.17407): HARD compression prunes/summarizes into fewer *text* tokens
(still discrete, e.g. LLMLingua); SOFT compression maps into fewer *continuous
vectors* — the vectorizer family.

The mechanism is one we already met: these soft/gist/latent vectors occupy token
SLOTS in the decoder's sequence but are continuous — exactly the Q6 image-
placeholder splice (image embeddings hijack placeholder slots). The
encoder->latent->decoder shape is the S1 byte-latent architecture pointed at
CONTEXT instead of bytes; the motive is the S5 KV-compression lever; the "skip the
vocabulary, produce vectors directly" move is the Q1 image vectorizer.

Unifying frame this unlocks (a strong candidate thesis for the whole chapter):
tokenization (text -> ID -> vector) is just one way to produce the input-vector
sequence. Vision vectorizer, soft context compression, and byte-latent are the
SAME idea — an encoder produces the input vectors directly — applied to image,
text-context, and bytes respectively. Discrete tokens are one source of input
vectors, not the only one. Open question: how far can a model run on mostly
soft/continuous inputs (retrieved memories, compressed context, images) with
discrete tokens as just the human-facing I/O layer?

---

## Closing frame (for the subsection's conclusion)

All seven directions are one question in different clothes: the interview treated
the tokenizer as a fixed, discrete, frequency-learned, compression-optimized,
once-and-frozen table — and every frontier move relaxes one of those adjectives.
Learned-latent (S1) relaxes "fixed/discrete"; vocab scaling (S2) relaxes "the
right size is settled"; reasoning (S3) exposes the cost of "discrete hides
sub-token structure"; the attack surface (S4) exposes "canonical = trusted";
runtime granularity (S5) relaxes "once and frozen"; cross-domain (S6) relaxes
"frequency equals meaning, and the unit is discrete at all"; and soft tokens (S7)
relaxes the deepest assumption of all — that the model's input is a *token*
sequence rather than a *vector* sequence. The unifying thesis: tokenization is
the choice of the model's atomic unit of computation, discrete tokens are only
one way to produce the input-vector sequence, and the field is renegotiating
every property of that choice at once.
