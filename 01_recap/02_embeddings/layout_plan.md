# Embeddings (recap chapter) — high-level layout / subsection plan

The second recap chapter, added via the tokenization/embeddings split
(`LOGS/20260723_1216_tokenization_embeddings_split.md`). Study-Note tier (chapter
+ one demo, lighter than a Core Module). Inference-focused, grounded on Qwen3-0.6B.
It picks up the tokenization chapter's cliffhanger ("these token IDs mean
nothing") and pays it off, then bridges to the position/RoPE and attention recap
chapters.

Organizing seam: **tokenization decides WHICH symbols exist; embeddings decide
what each one MEANS — as a position in a learned geometric space.** And the deep
inference hook: the embedding table is cheap to look up but expensive to use as
the output LM head.

## Subsections

1. **The payoff of the cliffhanger: IDs are meaningless; embeddings give meaning.**
   A token ID is a row number (seat number) — no math works on it. The embedding
   table turns each ID into a vector the model can compute with. This is where
   tokenization hands off.

2. **The embedding table = a lookup.** A `vocab_size x hidden` matrix; token id ->
   its row = its vector. Grounded: Qwen3-0.6B is 151936 x 1024. Mechanically a
   gather (O(1) per token) — cheap. This is the literal first layer of the model.

3. **What the vectors represent: geometry = similarity (distributional
   hypothesis).** "You shall know a word by the company it keeps" (Firth 1957).
   Similar-context tokens land near each other; directions carry relationships
   (king - man + woman ~ queen). word2vec/GloVe as the classic standalone example
   that made this famous.

4. **How the numbers are learned.** word2vec (predict context <-> predict word),
   GloVe (global co-occurrence). In an LLM the embedding table is not standalone —
   it's the first layer, learned jointly with everything else by next-token
   prediction. Not handwritten.

5. **Static vs contextual (the key distinction).** The raw embedding row is STATIC
   (one vector per token, like word2vec — "bank" starts as one blurry vector). The
   transformer CONTEXTUALIZES it: attention mixes neighbors so "river bank" and
   "money bank" move apart by the time they're deep in the model. So the embedding
   is the STARTING representation; meaning is refined by the model. (Subword note:
   the token, not the word, gets a row — the tokenization link.)

6. **Dimensionality.** hidden_size = 1024 for Qwen3-0.6B: the vector lives in
   1024-dim space. Bigger models -> bigger dim -> more capacity, more cost. What a
   dimension "is" (loosely) and why you can't picture it.

7. **The other end: the LM head / unembedding.** Final hidden state -> logits over
   the whole vocab, logit_i = <hidden, row_i>. This is where generation's
   "distribution over the next token" comes from (the tokenization Q5 payoff:
   logits over 151936, softmax, greedy). The unembedding matrix.

8. **Tied embeddings (the callback to tokenization Q17/Q25).** Input embedding =
   output LM head, same matrix used twice, for small models (Qwen3-0.6B ties;
   155.6M = ~26% of the model). Big models untie. New grounded nugget: in UNTIED
   models E and U end up largely ORTHOGONAL — they do different jobs (E = a good
   starting representation for contextualization; U = accurate next-token
   discrimination), so untying lets them specialize; tying forces them equal
   (a regularizer). The direct payoff of tokenization's forward-hook.

9. **Inference angle: lookup is cheap, the LM head is expensive.** The input
   lookup is a gather; the LM head is the 155.6M matmul + softmax over 151936
   EVERY step — the dominant cost, especially in a compressed drafter (ties to
   FR-Spec / VocabTrim / the tokenization Q19 spec-decoding thread). The vocab-size
   cost the tokenization chapter deferred lives HERE.

10. **Multimodal / soft embeddings splice into the SAME space.** From the image-
    tokenization chapter: the projector maps image patch vectors into the LLM's
    embedding space; compressed-context vectors (soft tokens) do the same. So the
    embedding space is the shared destination — the "input is a vector sequence,
    discrete tokens are one source" frame (Future-Directions S7).

11. **Bridge forward.** The embedding says WHAT a token means, not WHERE it sits;
    position is added separately (RoPE — the next recap chapter). And attention is
    what does the contextualization (also upcoming). End on that hand-off.

## SUPERSET SCOPE — the notes cover all branches (versions get derived)

The 11 subsections above are Part A (the LLM's INPUT embedding table). To be a
superset, the notes also cover the other meanings/branches of "embeddings". Any
deliverable version (blog / subsection / full chapter) selects a slice; the notes
hold everything.

### Part A — Input embeddings (the LLM's first layer) = subsections 1-11 above.

### Part B — Embedding MODELS (retrieval / semantic search / RAG) [the big miss]
A *different* use of "embeddings" from Part A. Cover:
- Input embedding (Part A) = per-TOKEN, one vector per token, then contextualized.
  Embedding model = one vector per whole SENTENCE/DOCUMENT, for similarity/search.
- How they're built: a repurposed encoder (BERT-style) or a decoder LLM with a
  pooling head, fine-tuned with CONTRASTIVE learning (similar texts -> high cosine,
  dissimilar -> low). Pooling strategies: mean / CLS / last-token.
- Use cases: RAG retrieval, semantic search, clustering, reranking, dedup.
- Bi-encoder (embed query + doc separately, cosine) vs cross-encoder (joint,
  reranking) — the retrieval-vs-rerank split.
- Grounding: Qwen ships dedicated Qwen3-Embedding models (note the tie_word_
  embeddings=false observation from the tokenization Q25 came from a Qwen3-
  Embedding config).
- Inference angle: embedding-model serving is a batch-encode workload (encode many
  texts -> vectors -> vector DB); prefill-heavy, no autoregressive decode; the
  "cost" is the forward pass + the vector index, not KV cache. Different serving
  profile from a generative LLM.

### Part C — Positional embeddings (the "where") [defer deep dive to RoPE chapter]
Token embedding = WHAT a token is; positional embedding = WHERE it sits. Brief
lineage: sinusoidal (original Transformer 2017) -> learned absolute (BERT/GPT-2)
-> RoPE (rotary, modern; Qwen3). Full treatment is the RoPE recap chapter; here
only the what-vs-where distinction and that position is ADDED separately.

### Part D — Adjacent topics (for completeness / cross-reference)
- Analogy arithmetic (king - man + woman ~ queen): the famous demonstration that
  DIRECTIONS in embedding space carry relationships (gender, plural, tense). Show
  it on real vectors in the demo.
- Count-based / LSA era (pre-word2vec): co-occurrence matrices + SVD/PMI were the
  first distributional dense vectors (1990s-2000s), before neural word2vec. A
  one-paragraph "prehistory" note between one-hot and word2vec.
- Similarity metric: cosine similarity (direction, magnitude-invariant) vs raw dot
  product; why cosine for retrieval. Embedding anisotropy / the "narrow cone"
  problem (LLM embeddings cluster in a cone; cosine can be misleading) — a mention.
- Segment / token-type embeddings (BERT's [CLS]/[SEP] + segment IDs) — encoder-era,
  a mention (decoder-only LLMs drop them).
- Multimodal embeddings: image/audio vectors projected into the SAME embedding
  space (the image-tokenization "vectorizer" + soft-tokens S7 frame) — already in
  subsection 10; cross-link.

## The one demo (Study-Note tier)
Load Qwen3-0.6B's real embedding table and show: (a) cosine similarity of related
tokens (` cat`/` dog` closer than ` cat`/` Tuesday`); (b) that it's tied
(input == output matrix); (c) the LM-head cost (shape 151936 x 1024). All
CPU/MPS, no CUDA.

## Scope notes
Recap tier = focused, not exhaustive (tokenization was over-built; keep this
tighter). NOT covering: training embeddings from scratch, embedding-model /
retrieval (sentence embeddings) beyond a mention, the full geometry/interpretability
literature. Position (RoPE), RMSNorm, attention are their OWN recap chapters.

## Sources (to cite when written)
- Distributional hypothesis (Firth): standard; GeeksforGeeks / CS447 lecture notes.
- word2vec (Mikolov 2013, arXiv:1301.3781), GloVe (Pennington 2014).
- Static vs contextual: zilliz AI-FAQ; "From Tokens to Words: On the Inner Lexicon
  of LLMs" (arXiv:2410.05864).
- Tied embeddings / E-vs-U orthogonality: Press & Wolf 2017 (tying);
  "weight tying when and why" (medium, vishal09vns); representational-capacity /
  unembedding-geometry notes (arXiv:2606.02765).
- Grounded: Qwen3-0.6B real embedding table (to inspect this session).
