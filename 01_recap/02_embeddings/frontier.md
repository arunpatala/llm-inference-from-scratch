# Embeddings — frontier / future directions (2024-2026)

Research survey for the chapter's depth/future-directions section. Verification
flags kept: [V] fetched this session, [Vsub] subagent-fetched, [L] listing-only
(provisional numbers), [?] 2026-dated/single-source (double-check before print).
Every claim ties back to a chapter thread where possible.

The one-line state: embeddings split hard into two frontiers — the RETRIEVAL
embedding (models/compression/multi-vector, where a proven single-vector ceiling
is the big news) and the GENERATIVE input table (untied-at-scale + LM-head cost),
and both are being squeezed for inference cost.

## 0. The standout result: single-vector retrieval has a HARD ceiling

"On the Theoretical Limitations of Embedding-Based Retrieval" (Weller, Boratko,
Naim, Lee, DeepMind; arXiv 2508.21038, ICLR 2026) [V]. A single-vector embedding
of dimension d can only ever return a BOUNDED number of top-k document subsets
(~d) — a rank/sign-rank argument that holds even with free, test-set-optimized
embeddings. Their LIMIT benchmark makes SOTA embedders fail trivially simple
queries (<20% recall@100) while BM25 and multi-vector don't. Directly bounds Q7's
bi-encoder: dense single-vector retrieval CANNOT be scaled away — the fix is
multi-vector or hybrid/lexical. Arguably the most important embeddings result of
the cycle for a retrieval chapter.

## 0a. Synthesis frame: the retrieval Pareto surface (author's synthesis)

The organizing frame for the whole retrieval-embeddings story (unifies S0 LIMIT,
S1 compression, S4 multi-vector). Retrieval embedding design is not "pick the best
method" — it is navigating a Pareto SURFACE (quality vs cost), tunable at test
time, with orthogonal COMPOSABLE knobs:
- # vectors per doc -> expressiveness vs storage/compute (MetaEmbed tunable
  multi-vector).
- dims per vector -> quality vs vector size (Matryoshka/MRL truncation).
- bits per number -> quality vs storage/speed (binary/int8, RaBitQ).
They stack: MRL-truncate the dims, then binary-quantize the bits, then choose the
number of vectors -> ~96x compression at ~99% quality in production (Vespa/Azure do
MRL x binary). Each knob is a Pareto trade-off; composed, a whole surface to pick
from for a latency/storage/recall budget.

Where LIMIT fits (and becomes useful, not just a downer): LIMIT bounds the
single-vector (m=1) SLICE of this surface — no amount of dims or bits buys past
it, because the bound is on m=1's rank. So the m=1 frontier CAPS OUT; adding
vectors (m>1) opens a HIGHER frontier above it. LIMIT draws the wall on the cheap
end of the surface; multi-vector is how you climb to a new one.

MUVERA is the clever move: instead of picking a point on the frontier, it tries to
BEND the frontier itself — multi-vector quality at single-vector search cost
(approximately). Frontier-shifting, not frontier-sliding.

One-line thesis: retrieval embedding design = navigating a Pareto surface
(vectors x dims x bits) for a quality/cost budget — single-vector has a proven
ceiling (LIMIT), multi-vector opens a higher frontier, and methods like MUVERA
shift the frontier rather than slide along it. Echoes the tokenization S5
"granularity as a runtime knob against KV" — the same tunable-resource-allocation
theme, now for retrieval. Ties to open question 11 (do the knobs compose cleanly,
or does compression compound with the multi-vector approximation error?).

## 1. Compressed / efficient embeddings (the "compressed embeddings" ask)

- Matryoshka Representation Learning (MRL, Kusupati et al., arXiv 2205.13147,
  NeurIPS 2022) [V]: nested coarse-to-fine dims; ONE embedding truncatable to many
  dims at ~no accuracy loss, no extra inference cost. Now first-class in OpenAI
  text-embedding-3 (`dimensions` param), Gemini, Nomic, Qwen3, EmbeddingGemma.
- Quantization: int8 = 4x smaller / ~99.3% retention; binary = 32x smaller /
  ~92.5% raw (~96% with float rescore); CPU search 3.7-24.8x faster. Pipeline =
  binary Hamming shortlist -> int8 rescore (HuggingFace embedding-quantization
  blog [Vsub]).
- RaBitQ (Gao & Long, arXiv 2405.12497, SIGMOD 2024) [Vsub] displaced product
  quantization: D-dim -> D-bit code with a PROVEN error bound. Shipped in Milvus
  2.6 (IVF_RABITQ: 32x memory + ~3x QPS at 94.7% vs 95.2% recall).
- What production vector DBs use in 2025-26: scalar + BINARY quantization is
  default-tier (Qdrant, pgvector bit/halfvec 0.7, Milvus, Weaviate); the new move
  is rotation-then-quantize (Weaviate Rotational Quantization 1.32/1.33, Qdrant
  TurboQuant 1.18). MRL-truncate THEN binary-quantize (orthogonal levers) ->
  ~96x compression at ~99% quality (Vespa, Azure AI Search) [L].
- Chapter link: this is the concrete "compressed embeddings" story — MRL (dims) x
  quantization (bits) are orthogonal compression axes stacked in production.

## 2. Multimodal embeddings (the shared-space claim, Q14)

- SigLIP 2 (Tschannen et al., DeepMind, arXiv 2502.14786, Feb 2025) [V]: the
  default open CLIP-successor; keeps SigLIP's per-pair sigmoid loss (no global
  softmax) + caption pretraining + self-distillation + native aspect ratio.
- THE MODALITY GAP is real and contested: dual-encoder CLIP/SigLIP put image and
  text in TWO separate cones ("Mind the Gap", arXiv 2203.02053); contrastive loss
  preserves rather than closes it. MLLM-based embedders route images through the
  LLM's TOKEN space to bridge it — E5-V (arXiv 2407.12580) [Vsub] trains on text
  pairs only and claims to bridge — but no clean quantified head-to-head on
  residual gap exists [?]. So the chapter's "image and text share one space" (Q14)
  needs a hedge: geometrically close, but a persistent gap in dual-encoders.
- 2025 production multimodal: jina-embeddings-v4 (arXiv 2506.18902, 3.8B, emits
  BOTH single- and multi-vector, LoRA task adapters) [Vsub]; Cohere Embed 4
  (128k context / ~200-page docs, MRL) [L]; voyage-multimodal-3 (interleaved
  text+image in ONE tower, not two CLIP towers) [L]; Qwen3-VL-Embedding (arXiv
  2601.04720, 2B/8B, 77.8 MMEB-V2) [Vsub, ?2026].
- Correction: EmbeddingGemma is TEXT-ONLY (308M on-device), NOT multimodal — file
  under efficient/on-device.

## 3. LLM-based embedding models (decoder-as-embedder, Q6)

- E5-mistral ("Improving Text Embeddings with LLMs", Wang et al., arXiv
  2401.00368, ACL 2024) [Vsub]: the template — decoder LLM (Mistral-7B) +
  last-token pooling + instruction-prefixed queries + ~500k SYNTHETIC pairs, SOTA
  in <1k steps.
- LLM2Vec (BehnamGhader et al., arXiv 2404.05961, COLM 2024) [Vsub]: retrofits
  bidirectionality into a decoder unsupervised (enable bidirectional attention ->
  masked-next-token -> SimCSE contrastive). The causal mask is not fundamental for
  embedding.
- NV-Embed (NVIDIA, arXiv 2405.17428, ICLR 2025) [V]: a learned latent-attention
  pooling layer (beats last-token AND mean) + removing the causal mask during
  contrastive training; v2 hit MTEB-English #1 (72.31).
- Qwen3-Embedding/Reranker (arXiv 2506.05176) [V]: current top open family;
  0.6B/4B/8B, MRL dims, dual-encoder embedder + cross-encoder reranker,
  self-bootstrapped data. 8B = 70.58 MTEB-Multilingual #1 (Jun 2025), beating
  Google/OpenAI APIs.
- What changed by making the embedder a big decoder: inherits pretrained
  world/code/multilingual knowledge -> strong zero-shot; instruction conditioning
  makes one model task-general; trains on its OWN synthetic data; causality and
  pooling become design levers (last-token vs latent-attention). The cost:
  3072-4096-d vectors + high per-embedding latency -> which is PRECISELY why MRL
  truncation and 300M on-device models (EmbeddingGemma, Sep 2025) are headline
  features. Chapter link: sharpens Q6 (last-token pooling justified) + the
  serving-cost angle (Q7b).
- MTEB caveat: MMTEB (arXiv 2502.13595, 500+ tasks) is current [L]; known problems
  = corpus contamination, saturation (hundreds of models within tiny margins),
  train-split overfitting. Treat leaderboard deltas skeptically.

## 4. Late interaction / multi-vector (ColBERT lineage) — got cheap

- ColBERTv2 (arXiv 2112.01488) residual compression 6-10x; PLAID (arXiv
  2205.09707) centroid pruning up to 7x/45x faster (GPU/CPU) [L]. Single-vector =
  cheap/lossy; multi-vector = higher quality but O(tokens) storage.
- ColPali (Faysse et al., arXiv 2407.01449, 2024) [V]: multi-vector embeddings
  DIRECTLY from document-PAGE IMAGES (128-d per patch), MaxSim late interaction,
  no OCR; introduced ViDoRe. ColQwen2 swaps in Qwen2-VL for native-resolution
  patches. (Ties image-tokenization patch vectors to retrieval.)
- MUVERA (Google, arXiv 2405.19504, NeurIPS 2024) [V]: Fixed Dimensional Encodings
  compress a SET of multi-vectors into ONE vector whose inner product approximates
  Chamfer/MaxSim -> reuse off-the-shelf MIPS; ~10% better recall at 90% lower
  latency. The "make multi-vector as cheap as single-vector" result — and a
  partial answer to the LIMIT ceiling.
- MetaEmbed (Meta, arXiv 2509.18095, Sep 2025) [L]: "Matryoshka multi-vector" —
  learnable Meta Tokens choose HOW MANY vectors at test time (between CLIP's 1 and
  ColBERT's hundreds).

## 5. Input-embedding table / vocab-projection (generative side, Q8/Q9)

- Weight tying biases embeddings toward the OUTPUT space (Lopardo et al., arXiv
  2603.26663, Mar 2026) [Vsub]: the strongest recent E-vs-U result — a tied matrix
  aligns with the UNEMBEDDING U of comparable untied models, not with untied input
  E (output gradients dominate early training; fixable by rescaling input
  gradients). Notes every 2025-26 frontier model (DeepSeek-V3, Qwen3-32B, Gemma-3,
  OLMo-2, Llama-3/4) ships UNTIED; past ~1B params tying saves <10% of params.
  Grounds the chapter's E-vs-U orthogonality nugget with a mechanism, and confirms
  the Q17/Q25 tie-at-small / untie-at-scale rule.
- Cutting the LM-head/softmax cost (Q9, connects to spec decoding): the LM-head
  can be >60% of draft-step latency at large vocab. FR-Spec (arXiv 2502.14856,
  ACL 2025) [V] restricts the DRAFT head to a frequency-ranked (Zipf) subset —
  ~75% less head compute, distribution-preserving, ~1.12x over EAGLE-2. VocabTrim
  (Qualcomm, arXiv 2506.22694) [V] training-free, +16% edge speedup.
- Structured/factorized embeddings: TensorGPT (Tensor-Train, ~38x on GPT-2
  embedding, arXiv 2307.00526); Kronecker byte-level embeddings (arXiv 2605.29459)
  [?] and CARVQ residual VQ [?] newer/less-verified.

## 6. Embedding geometry / interpretability

- Platonic Representation Hypothesis (Huh, Cheung, Wang, Isola, arXiv 2405.07987,
  ICML 2024) [V]: models — even across vision and language — CONVERGE to a shared
  representation as they scale. Contested by "Convergence Without Understanding"
  (arXiv 2605.23315, argues shared input constraints not shared reasoning) [?].
- Semantic structure in LLM embeddings (Kozlowski et al., arXiv 2508.10003, 2025)
  [Vsub]: antonym-pair projections match human ratings and collapse to a ~3-d
  subspace; steering one semantic direction BLEEDS into correlated ones (entangled
  features) — an inference-time steering caveat.
- Anisotropy / narrow cone: established (inflated cosine sim); fixes BERT-flow,
  BERT-whitening, ZCA/Soft-ZCA, Stable Anisotropic Regularization (ICLR 2024).
  Tie-in: untied models are LESS anisotropic than tied — sharpens the Q12
  anisotropy caveat.

## 7. Soft / latent / context-compression (inference lever; Future-Directions S7)

- Lineage: gist tokens -> ICAE -> AutoCompressor -> xRAG (compress a retrieved doc
  to ONE token, arXiv 2405.13792, NeurIPS 2024) -> 500xCompressor (carries
  compression in per-layer KV, arXiv 2408.03094) [L].
- The frontier moved from soft-prompt tokens to compressing the KV CACHE itself as
  the "context embedding": KV-Distill, PALU (low-rank KV, ICLR 2025), KVZip
  (query-agnostic), ClusterKV [L]. Frame: soft-prompt compression and KV-cache
  compression are CONVERGING into one learned-latent-context lever (the tokenization
  Future-Directions S7 "input is a vector sequence" thesis, at the context level).

## 8. Security (a genuinely new axis)

- Embeddings are NOT private: vec2text ("Text Embeddings Reveal (Almost) As Much
  As Text", arXiv 2310.06816) recovers 92% of 32-token inputs [V]; ZSInvert
  (arXiv 2504.00147) inverts ANY embedder zero-shot, no per-model training [V]; a
  2025 reproduction (arXiv 2507.07700) shows Gaussian noise/quantization mitigates.
- vec2vec / universal geometry (Jha, Zhang, Shmatikov, Morris, arXiv 2505.12540,
  NeurIPS 2025) [V]: translates embeddings between DIFFERENT models' spaces with NO
  paired data (cycle-consistency/adversarial) — empirical support for Platonic AND
  a serious vector-DB security implication (a stored vector DB is neither opaque
  nor model-locked).

## What changed recently (2025-2026)
- Quantization went PQ -> RaBitQ/rotation; binary+MRL stacking is standard in prod
  vector DBs.
- The embedder became a big decoder LLM (Qwen3-Embedding, gemini-embedding-001,
  NV-Embed) with instruction conditioning + synthetic data — AND the field pushed
  the opposite way (300M on-device, MRL truncation) because big decoders are
  expensive.
- Multimodal embedders unified text/image/doc-image/video (jina-v4, Embed 4,
  Qwen3-VL-Embedding); several emit both single- and multi-vector.
- Multi-vector got cheap (MUVERA FDE; MetaEmbed tunable Matryoshka multi-vector).
- A HARD theoretical ceiling on single-vector retrieval was proven (LIMIT).
- Untied embeddings became universal at scale, with a mechanistic E-vs-U account.
- LM-head cost-cutting entered speculative decoding (FR-Spec, VocabTrim).

## Contested / surprising
- Does text+image truly share one space? Geometrically yes, but dual-encoders keep
  a persistent modality gap; MLLM-embedders claim to bridge, no clean head-to-head.
- Platonic convergence vs "convergence without understanding".
- Single-vector retrieval has a mathematical ceiling (LIMIT) — can't be scaled away.
- Embeddings leak inputs and are cross-model translatable without pairs (ZSInvert,
  vec2vec) — a vector DB is not a safe/opaque store.
- Tying embeddings may quietly hurt at scale (shapes the matrix for prediction).
- Isotropy/whitening isn't an unqualified good (entangled steering; anisotropy
  correlates with tying).

## Open questions (seed future directions)
1. Given LIMIT's dimension ceiling, what retrieval architecture scales past it —
   multi-vector, sparse/lexical hybrid, or generative retrieval?
2. Can MUVERA-style FDE keep multi-vector quality at binary/int8 precision?
3. Per-query adaptive MRL truncation dimension (vs a global choice)?
4. What's the actual residual modality gap of MLLM-embedders vs dual-encoders,
   measured the same way?
5. Do decoder-LLM embedders inherit BERT's anisotropy, and does latent-attention
   pooling change the geometry?
6. Can the LM-head cost be cut for the TARGET model (not just drafters) with exact
   sampling preserved?
7. Optimal PARTIAL/low-rank tying — the <10% param saving without the
   representational cost?
8. Does universal geometry (vec2vec) enable a "translate once, reuse everywhere"
   interop layer — and how to defend a vector DB against it?
9. The inference-optimal split between soft-prompt/gist and KV-cache compression,
   unified into one latent-context objective?
10. Provable inversion defenses without retrieval-quality loss?
11. Storage/quality frontier for doc-image late interaction with pruning +
    quantization + FDE combined?
12. How much MTEB progress survives contamination-controlled held-out eval?
13. Are the ~3-d semantic subspaces stable across models (a Platonic signature),
    usable for cheap controllable steering?
14. Can instruction-conditioned embeddings be cached/amortized (no fresh full-LLM
    forward per query)?
15. A compression scheme simultaneously Matryoshka-truncatable, binary-quantizable,
    AND inversion-resistant — or are these in tension?

## Verification note
[V]/[Vsub] items had abstracts fetched; [L] provisional (numbers from listings);
[?] 2026-dated/single-source, confirm before print. Notable [?]: Qwen3-VL-Embedding
2601.04720, Gemini Embedding 2 2605.27295, Leviathan decoupling 2601.22040,
Kronecker embeddings 2605.29459, "Convergence Without Understanding" 2605.23315.
Correction folded in: EmbeddingGemma is text-only.
