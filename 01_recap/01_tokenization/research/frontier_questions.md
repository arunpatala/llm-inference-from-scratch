# Tokenization — research-level / frontier questions (mid-2026)

Not chapter-interview questions (those are in `../01_questions.md`). These are
open, forward-looking, sometimes-contested questions about where tokenization is
*going* — for the author's own learning in research mode. Grounded by a research
subagent's survey (URLs below). Verification caveats from that survey are kept:
`[verified]` = primary source fetched this session; `[unverified]` = a specific
number not confirmed against a primary source.

The one-line state of the field: **quality is no longer the blocker for
tokenizer-free models — inference speed is — and the debate has shifted from
"tokenizer vs none" to "fixed vs learned dynamic boundaries." Meanwhile the
serving bottleneck migrated *inside* the model, to the large-vocab LM head.**

---

## A. Will BPE survive? Tokenizer-free / byte-level / latent architectures

Current state: byte/latent models have reached **quality parity at scale**, so
the fight is now entirely about inference throughput.
- **BLT (Byte Latent Transformer, Meta)** — dynamically sized byte patches whose
  boundaries come from a small entropy model over next-byte uncertainty (more
  compute where bytes are hard to predict). First FLOP-controlled byte study to
  8B/4T bytes; matches Llama 3 at equal training FLOPs using up to ~50% fewer
  *inference* FLOPs; BLT-Entropy beat Llama 3 on a downstream average (61.1 vs
  60.0) and excels at char-level/robustness. (2412.09871)
- **H-Net (Cartesia)** — end-to-end *learned* chunking (similarity routing)
  replacing tokenize->model->detokenize. 2-stage H-Net beats a FLOP-matched BPE
  transformer (bits-per-byte 0.715 vs 0.730; 58.2% vs 55.5% downstream); a 760M
  2-stage matches a 1.3B BPE model; far more robust to char perturbation (42.8 vs
  22.2 on perturbed HellaSwag); ~4x data-efficiency on DNA. (2507.07955)
- Lineage: MEGABYTE (2023, did *not* match tokenized models at controlled
  compute), SpaceByte (NeurIPS 2024), MambaByte, and **EvaByte** (6.5B, 1.5T
  bytes, multibyte prediction, ~5x less data, up to 2x faster decode).
- **The 2026 move is fixing inference speed:** Fast-BLT [verified] adds a
  block-wise *diffusion* objective (BLT-D) + self-speculation, claiming >50%
  lower memory-bandwidth cost on generation (2605.08044); MTPC speeds byte models
  while provably preserving the verifier's outputs (2511.11346).
- **Production reality check:** nobody ships tokenizer-free at the frontier.
  Gemma 4 (April 2026) uses a **262,144-entry BPE-with-byte-fallback** vocab — a
  *larger conventional* tokenizer, the opposite direction from byte-level.

1. Do BLT/H-Net actually beat BPE on the compute-vs-quality *Pareto frontier* at
   frontier scale yet, or only at matched-FLOP research scale? (Quality: yes;
   throughput: not yet — that's the whole 2026 fight.)
2. **The book's real question:** if a model dynamically chunks bytes instead of
   using a fixed vocab, what happens to everything the serving stack assumes is
   fixed — KV-cache accounting, prefix caching (exact token match), speculative
   decoding (shared vocab), context budgeting? Does tokenizer-free *break the
   stack we're building*? (See sharp-question 1-2 below — this is genuinely open.)
3. Does byte-latent keep the no-OOV byte-floor guarantee trivially *and* dissolve
   fertility (no fixed vocab to be a bad fit), or just move the cost into patch
   length? (Caveat: "UTF-8 Plumbing", 2511.05578, argues byte-level models
   *unavoidably* can emit ill-formed UTF-8 — a new failure mode.)

## B. Vocabulary scaling — is 151k already too small? (consensus fractured)

- **"Scaling Laws with Vocabulary" (Sea AI Lab, NeurIPS 2024)** — optimal vocab
  grows with compute but *sub-linearly* with non-vocab params; Llama-2-70B
  "should" have used ~216K vs its 32K; most LLMs are *under*-vocabularized. A 3B
  model gained ARC-C 29.1->32.0 at fixed FLOPs going 32K->43K. (2407.13623)
- **Over-Tokenized Transformer (ByteDance, ICML 2025)** — *decouple input from
  output vocab*: scaling **input** vocab (multi-gram tokens) helps all sizes;
  scaling **output** softmax helps large models but hurts small ones. A ~12.8M
  n-gram input vocab let a 400M model match a 1B baseline (~2.5x efficiency).
  (2501.16975)
- Frontier vocab grew 4-8x 2023->2025: GPT-4o o200k ~200K, Llama 3 128,256,
  Gemma 256K, Qwen2.5 151,936, Gemma 4 262,144.
- **Contested counter-current:** Meta's **"Compute Optimal Tokenization"**
  [verified], 988 models 50M-7B, finds params scale with data in **bytes, not
  tokens**, and — against "bigger vocab is better" — **optimal compression
  *decreases* as compute grows** (BPE-optimal rate != model-optimal rate).
  (2605.01188)
- **SuperBPE** — merges subwords *across* whitespace into "superword" tokens:
  -33% tokens, -27% inference compute, +4.0% avg task gain at 200K vocab.
  (2503.13423)

4. Do "larger models deserve larger vocabularies" and "Compute Optimal
   Tokenization" reconcile into one law parameterized by tokens-past-Chinchilla,
   or are they measuring different regimes (compute-optimal vs over-trained
   frontier)? (Unresolved — notably between overlapping Meta author groups.)
5. Given the Q17 embedding/LM-head cost, at what model size does growing vocab
   stop paying off — and does input/output decoupling change the answer (sparse
   n-gram input lookup is not dense-FLOP)?
6. What does SuperBPE-style cross-word merging do to the digit-splitting /
   arithmetic trade-off (Q14) and to fertility fairness (Q24)?

## C. Tokenizer transplantation — does "frozen for life" still hold?

Trajectory: heuristic init (ReTok mean-of-subwords, WECHSEL, FOCUS) -> learned
hypernetwork (**ZeTT**, predicts embeddings for an arbitrary tokenizer but costs
hundreds of GPU-hours) -> **training-free geometric methods that now match/beat
the hypernetwork with zero gradient steps**. The 2025 headline is **OMP
(Orthogonal Matching Pursuit) transplantation** (Arcee/mergekit): reconstruct
each donor embedding as a k-sparse combo of shared anchor tokens, transfer the
coefficients — reportedly beats zero/mean-init, WECHSEL, FOCUS, and ZeTT.
(2506.06607, ZeTT 2405.07883, ReTok 2410.04335, Trans-Tokenization 2408.04303)

7. If you can graft a *better* tokenizer onto a pretrained model cheaply (OMP +
   a short "healing" phase of billions, not trillions, of tokens), does that
   **kill the "frozen for life" constraint** (Q10/Q24/Q29) this book leans on?
   (Increasingly, yes — worth hedging that claim in the chapter.)
8. Could a multi-tenant server **hot-swap a language-optimized tokenizer per
   request** via training-free transplant, to beat the multilingual tax? (Open —
   sharp-question 7.)

## D. Tokenization x reasoning (and a book-worthy correction)

**Correction for the chapter:** Llama 3 does *not* use single-digit tokenization
— that was Llama 1/2. Llama 3 (like GPT-3.5/4o, Gemini) chunks digit runs
left-to-right into groups of <=3 digits.
- **"Tokenization counts" (Singh & Strouse)** — L2R digit chunking destabilizes
  place value (a carry shifts chunk boundaries); forcing R2L alignment yields
  large *systematic* arithmetic gains (e.g. GPT-3.5 7-9 digit addition
  50.3%->97%; some overall-accuracy pairs [unverified]). R2L 3-digit has since
  reached production. (2402.14903)
- **Genuinely contested:** the **"Number Cookbook"**, training from scratch,
  finds *per-digit* beats 3-digit chunking for accuracy *and* length
  generalization, and larger vocab doesn't help numeracy. So R2L is a good
  *patch* for existing tokenizers; per-digit may win in a *redesign*.
  (2411.03766)
- **"Counting Ability of LLMs and Impact of Tokenization"** — the "strawberry"
  failures are largely a tokenization artifact (BPE hides char boundaries,
  undermining CoT counting), not pure reasoning failure. (2410.19730)
- **"Say Anything but This: When Tokenizer Betrays Reasoning"** [verified] —
  11,000+ replacement trials, "phantom edits", an 8-artifact taxonomy; argues
  part of apparent reasoning deficiency originates in the tokenizer layer.
  (2601.14658)

9. Is tokenization a *cause* of reasoning failures or a correlate? (2025-2026
   evidence increasingly says partial cause — Say Anything but This.)
10. Does per-digit vs R2L-3-digit have a clear winner in a ground-up redesign,
    and what does that imply for code/tool-use models that must emit exact
    numeric strings? (Open — the two papers disagree by regime.)
11. Does the char-level reasoning benefit survive at frontier scale, or is
    "strawberry" a small-model artifact that vanishes once the model memorizes
    BPE-hidden structure? (Open.)

## E. Inference-efficiency frontier — the bottleneck moved inside the model

Structural shift: as EAGLE-3/Medusa/MTP shrink the drafter's transformer body to
~1 layer, the **O(V·d) LM head over a 128-256K vocab becomes the drafter's
dominant cost** (~49% of drafting compute at 128K, ~62% with softmax). Two
response families:
- **Vocab-space truncation:** FR-Spec (frequency-ranked draft, ~75% head cut,
  1.12x over EAGLE-2, ships in MiniCPM4 — 2502.14856); VocabTrim (Qualcomm,
  training-free drop-in, ~16% memory-bound speedup — 2506.22694). Trend is
  **dynamic > static:** **DynaSpec** [verified] routes each context to coarse
  token clusters, recovering 98.4% of full-vocab accepted length vs 93.6% for
  fixed shortlists, up to 2.23x throughput on rare-token workloads (2510.13847).
- **Head restructuring:** SlimSpec low-rank-factorizes the drafter head, keeping
  the *full* vocab (no acceptance ceiling), claiming 4-5x head-latency reduction
  [author-reported] (2605.10453).
- **Detokenization is an emerging CPU-side tax** distinct from the LM head:
  vLLM's incremental detokenizer has documented failure modes (special tokens
  split across streaming chunks — our Q21; O(n^2) parsing on multi-token
  speculative deltas). CPU tokenization can reportedly saturate 40-60% of an H100
  host at high concurrency [vendor claim, unverified].
- **Native MTP heads are themselves large-vocab LM heads** (DeepSeek-V3 MTP:
  2nd-token acceptance 85-90%, ~1.8x TPS; SGLang up to ~60% higher throughput),
  so the FR-Spec/SlimSpec head-cost problem applies directly to MTP.

12. Is the endgame *dynamic sparse heads* (DynaSpec routing) or *structurally
    factorized heads* (SlimSpec low-rank) — and can they compose without eroding
    the exactness guarantee? (Open — sharp-question 3.)
13. What is the **detokenization-side scaling law**: as vocab and speculative-
    delta sizes grow, when does CPU-side streaming detokenization (our Q9/Q30
    output-side bottleneck) overtake the GPU? (Open — genuinely under-studied.)

## F. Multilingual fairness / tokenizer tax — real, quantified, NOT closed

- **"The Token Tax"** (Sept 2025) — on AfriMMLU across 16 African languages,
  fertility (tokens/word) *mechanistically predicts accuracy*, not just cost:
  higher fertility -> lower accuracy. (2509.05486)
- **"The Tokenizer Tax Across 24 European Languages"** (2026) — first controlled
  parallel-text isolation: ~2.5x fertility spread (English ~1.23 vs Greek/Maltese
  ~3.1 tokens/word), and crucially **vocab size alone doesn't predict efficiency
  — the proportion of script-specific merges matters more** (Llama 4's 200K got
  the *lowest* Ukrainian fertility; Qwen3's 151K the *worst*). (2605.24718)
- Mitigations are largely *post-hoc tokenizer swaps*: script-aware replacements
  (BrahmicTokenizer for Indic), ZeTT-style transfer, and H-Net++ attacking the
  tax architecturally (Persian +12% compression). The metric itself is contested
  ("Beyond Fertility"/STRR, 2510.09947).

14. Has any *frontier* tokenizer closed the tax, or is it always a post-hoc swap?
    (Not closed; and vocab size isn't the lever — script-specific merges are.)
15. Is fertility even the right fairness metric? (Contested — no agreed metric.)

## G. Security — the tokenizer as a first-class attack/supply-chain surface

- Glitch/undertrained-token detection matured: Fishing for Magikarp (EMNLP
  2024) -> GlitchMiner (gradient-based, AAAI 2026, 2410.15052) -> GlitchProber.
- **New 2025 attack class — adversarial tokenization:** the same string has many
  non-canonical tokenizations. **"Adversarial Tokenization" (ACL 2025)** shows
  re-tokenizing a harmful prompt *without changing any characters* preserves
  semantic signal to the model while evading safety alignment trained on
  canonical tokenizations. (2025.acl-long.1012)
- **TokenBreak (June 2025)** — insert chars so the *tokenizer* splits
  differently, fooling classifier guardrails while the LLM still reads the intent
  — and finds **BPE/WordPiece are susceptible while Unigram is notably more
  robust** (a concrete defense recommendation). (2506.07948)
- "Stop Taking Tokenizers for Granted" (2601.13260) frames the tokenizer as a
  supply-chain surface and calls for embedding audits / token pruning as a
  pre-deployment gate. (Fourth security thread for the chapter, after Q6/Q16/Q26.)

16. Should guardrails move to the byte/normalized-string layer entirely, once
    non-canonical tokenization can carry semantic payload past canonical-
    tokenization-trained classifiers? (Open — the Unigram-over-BPE finding is a
    partial answer.)

---

## Sharp research-level open questions (the subagent's list, lightly edited)

1. If segmentation is a *learned, input-dependent runtime computation*
   (BLT/H-Net), what's the right serving abstraction — and how do dynamic patch
   boundaries interact with paged KV cache, prefix caching, and continuous
   batching when different requests patch differently?
2. Can entropy/learned patching be made *deterministic and cacheable* across
   requests, or does input-dependent chunking fundamentally break prompt-prefix
   reuse in a shared-cache serving stack?
3. When the LM head over a 256K vocab is ~half the drafter cost, is the endgame
   dynamic sparse heads (DynaSpec) or factorized heads (SlimSpec) — and can they
   compose without eroding the exactness guarantee?
4. Do "Compute Optimal Tokenization" and "larger vocab" reconcile into one law
   by training-token-past-Chinchilla, or are they different regimes?
5. If input/output vocab should scale asymmetrically (Over-Tokenized), what's the
   optimal input-vocab size for a fixed serving memory budget (sparse lookup, not
   dense FLOP)?
6. Does the char-level reasoning benefit survive at frontier scale, or is
   "strawberry" a small-model artifact?
7. Can a model be made tokenizer-agnostic at inference (OMP-style transplant)
   well enough to hot-swap a per-request, per-language tokenizer in a multi-tenant
   server?
8. Is R2L digit tokenization a stopgap that per-digit or byte-level makes
   obsolete — and what does that mean for models that must emit exact numbers?
9. What's the correct guardrail architecture once non-canonical tokenization
   carries semantic payload past canonical-trained classifiers?
10. As diffusion byte decoding (BLT-D) matures, does the draft-verify framing of
    speculative decoding survive, or is it replaced by parallel block generation
    with a different acceptance calculus?
11. How should glitch-token pruning be standardized as a release gate without a
    costly re-embedding pass on frontier-size vocabularies?
12. For MTP/EAGLE drafters that are themselves large-vocab heads, is there a
    unified theory of "how much vocabulary a draft step actually needs" as a
    function of context entropy?
13. Does measuring pretraining data in *bytes not tokens* change how we budget
    and compare multilingual corpora — and does it silently penalize or reward
    high-fertility languages in the data mix?
14. Can learned dynamic chunking be *transplanted* onto a BPE-pretrained model,
    or does it require pretraining from bytes (gating adoption)?
15. What's the detokenization-side scaling law — when does CPU-side streaming
    detokenization, not the GPU, become the serving bottleneck?

## The five genuinely contested debates (unresolved as of mid-2026)

- Per-digit vs R2L-3-digit for arithmetic (frontier-patch evidence vs
  from-scratch evidence disagree).
- Does vocab keep growing or start shrinking? ("larger vocab" vs "compute-optimal
  compression decreases with compute" — overlapping Meta authors, unresolved).
- Will any frontier lab ship tokenizer-free? (Parity on paper; production keeps
  choosing bigger BPE.)
- Is fertility the right multilingual-fairness metric? (STRR et al. dispute it.)
- Are byte-level robustness wins worth their new failure modes (ill-formed UTF-8,
  slower decode)?

## Key sources

Byte/latent: 2412.09871 (BLT) · 2605.08044 (Fast-BLT [verified]) · 2507.07955
(H-Net) · cartesia.ai/blog/hierarchical-modeling · 2508.05628 (H-Net++) ·
huggingface.co/EvaByte/EvaByte · 2404.14408 (SpaceByte) · 2511.11346 (MTPC) ·
2511.05578 (UTF-8 ill-formed) · blog.google/.../gemma-4/
Vocab scaling: 2407.13623 (Scaling Laws w/ Vocabulary) · 2501.16975
(Over-Tokenized) · 2605.01188 (Compute Optimal Tokenization [verified]) ·
2503.13423 (SuperBPE)
Transfer: 2405.07883 (ZeTT) · 2506.06607 (OMP training-free) · 2410.04335
(ReTok) · 2408.04303 (Trans-Tokenization) · 2505.09738 (TokenAdapt)
Reasoning: 2402.14903 (Tokenization counts) ·
beren.io/2024-07-07-Right-to-Left-Integer-Tokenization · 2411.03766 (Number
Cookbook) · 2410.19730 (Counting Ability) · 2601.14658 (Say Anything but This
[verified])
Serving: 2502.14856 (FR-Spec) · 2506.22694 (VocabTrim) · 2510.13847 (DynaSpec
[verified]) · 2605.10453 (SlimSpec) · lmsys.org/blog/2025-07-17-mtp
Multilingual: 2509.05486 (Token Tax) · 2605.24718 (24 European Languages) ·
2406.11214 (GPT-4o bias) · 2510.09947 (Beyond Fertility/STRR)
Security: 2024.emnlp-main.649 (Fishing for Magikarp) · 2410.15052 (GlitchMiner) ·
2025.acl-long.1012 (Adversarial Tokenization) · 2506.07948 (TokenBreak) ·
2601.13260 (Stop Taking Tokenizers for Granted)

Verification: the four most load-bearing 2026 papers were fetched and confirmed
this session (Fast-BLT 2605.08044, Compute Optimal Tokenization 2605.01188, Say
Anything but This 2601.14658, DynaSpec 2510.13847). Items flagged [unverified]:
exact Singh & Strouse overall-accuracy pairs, SlimSpec author-reported figures,
vendor detokenization/glitch percentages, EvaByte's multibyte window, Adversarial
Tokenization ASR. Leads seen but not deep-fetched (cite as pointers only):
FLEXITOKENS, MrT5, TokAlign, BrahmicTokenizer.
