# Embeddings (recap) — interview questions

Answer inline under each question, same as the tokenization files. The embeddings
history is the OTHER HALF of the tokenization history: tokenization covered which
units exist (word -> subword -> byte); this covers how a unit becomes a vector
(one-hot -> static learned -> contextual). Researched reference:
`embeddings_history.md`, `layout_plan.md`.

## Set A — the history (one-hot -> static -> contextual)

1. Before learned embeddings, a word was handed to a model as a one-hot vector:
   1 at the word's index, 0 elsewhere. (a) What's its dimension and why is that a
   practical problem? (b) What does one-hot fail to capture about meaning
   (compare cat/dog/Tuesday)? (c) How does an embedding MATRIX turn a one-hot into
   a dense vector?

   (a) Dimension = vocab size (e.g. 50k), so it's huge and sparse — one 50k-long
   vector per word. (b) The fatal flaw (author): one-hot gives no similarity —
   any two different words are ORTHOGONAL (dot product 0), so cat.dog = 0 and
   cat.Tuesday = 0; every pair is equally far. It encodes identity but zero
   similarity, so the model can't share structure — what it learns about cat
   can't transfer to dog. It must learn every word independently (data-
   inefficient, no generalization). (c) The embedding matrix E is
   vocab_size x hidden. one-hot(word) @ E = the ROW of E at the word's index =
   the word's dense embedding — multiplying by a one-hot just SELECTS a row. So
   "one-hot x E" and "look up the word's row" are the same operation; in practice
   you skip the matmul and index directly (a gather, layout S2). This is why E is
   a "lookup table", and it's literally the 151936x1024 E from tokenization Q17.
   One-hot has no geometry (all orthogonal); learned E has geometry (similar =
   close) — the whole leap (layout S3).

2. So how do you LEARN an E where cat and dog end up close, without hand-labeling
   which words are similar? What's the core idea/signal (the distributional
   hypothesis), and how did word2vec/GloVe turn it into an algorithm?

   The distributional hypothesis: a word is known by the company it keeps (Firth
   1957). Author's mechanism: cat and dog are INTERCHANGEABLE in the same slots
   ("I have a pet ___"), so a network trained to fill that slot must give them
   similar vectors (they have to make the same prediction); the vector used to
   predict becomes the embedding. No labels — the text's own word<->context pairs
   are the self-supervised signal.
   Sharpenings: word2vec has two directions — CBOW predicts the target FROM its
   context (the "predict the blank" framing), skip-gram predicts the CONTEXT from
   the target word; either way the learned per-word weight vector IS its
   embedding. GloVe reaches the same place via GLOBAL co-occurrence counts
   (factorize how often words co-occur) instead of window-prediction. Same
   distributional idea, two routes. Punchline: similarity falls out of
   interchangeability — same company -> pushed together -> cat/dog cluster,
   Tuesday drifts off. Geometry from raw text, zero labels.

3. word2vec/GloVe give ONE fixed vector per word (static). What's the problem
   with a single vector for a word like "bank" (river vs money), and what did
   contextual embeddings (ELMo, then BERT/GPT) change?

   Static problem: a single fixed vector for "bank" is forced to be a BLURRY
   AVERAGE of all senses — a compromise between river-bank and money-bank that's
   not really either. Polysemy collapses into one muddled superposition; static
   embeddings can't separate senses (one slot per word).
   Contextual (author): the vector is computed FROM the sentence, so the two
   banks arrive at different vectors — ELMo (biLSTM LM, 2018) then BERT/GPT
   (transformers).
   Crucial modern nuance (layout S5): a modern LLM is BOTH, in sequence. The input
   embedding table is STILL static (E[bank] = one vector regardless of context);
   then the transformer layers CONTEXTUALIZE it (attention mixes in neighbors), so
   by the time bank is deep in the model, river-bank and money-bank have moved
   apart. Static input embedding -> transformer contextualizes -> context-
   dependent representation. The static embedding is the starting point; the model
   refines it into the contextual one.

4. FastText added subword (character n-gram) information to word embeddings. What
   problem did that solve that word2vec/GloVe couldn't (hint: the OOV / rare-word
   issue, and it connects to why LLMs tokenize into subwords)?

   Author got OOV (unseen word decomposes into known subword pieces -> still gets
   a vector). Correction of a conflation: subwords do NOT help with different
   SENSES (bank river vs money) — that's polysemy, solved by CONTEXTUAL embeddings
   (Q3). Subwords work on the SURFACE form. FastText/subwords fix three
   surface-level problems: (1) OOV; (2) rare words share pieces with common words
   -> training signal instead of a starved vector (exactly the Q16 undertrained-
   token point); (3) morphology — run/running/runner share the "run" subword, so
   their vectors are related instead of independent.
   The two axes of the chapter, kept apart: subword (FastText -> BPE) handles the
   SURFACE (OOV/rare/morphology — words that share pieces); contextual (ELMo/BERT)
   handles MEANING IN CONTEXT (polysemy — same word, different senses). Punchline:
   FastText was the embedding-side precursor to BPE subword tokenization — the same
   "represent by subword pieces" insight LLMs adopted at the tokenizer level.

5. In a modern LLM the embedding table isn't learned by word2vec/GloVe at all —
   it's trained end-to-end from scratch (random init, joint next-token
   prediction). Why did the field abandon separate word-vector pretraining, and
   what does "end-to-end" buy?

   Three reasons (author got the first two):
   - Learn for the TASK, not a proxy (author): word2vec optimizes a proxy
     ("predict nearby words"); end-to-end optimizes the real objective (next-token
     prediction), so the embedding is shaped by what the model actually needs. No
     two-stage disconnect (the same insight as the learned-tokenization §1
     reasoning).
   - Subword (author): word2vec gives WORD vectors, but LLMs tokenize into
     subwords, so you need a table over subword tokens — the jointly-learned table
     gives that for free; a bolt-on word table doesn't fit the vocab.
   - Contextual >> static (Q3): the transformer contextualizes anyway, so you
     don't need a separately-good static embedding, just a good STARTING point for
     the model's own contextualization — which end-to-end learns. (Small models
     also TIE it with the LM head — one matrix, learned jointly, used both ways.)
   So: jointly-optimized-for-the-real-task + subword + gets-contextualized beats
   separately-pretrained-for-a-proxy + word-level + static on every axis. The
   embedding table is now just the model's first layer.

## Set B — embedding MODELS (retrieval / semantic search / RAG)

A DIFFERENT use of "embeddings" from Set A: not the LLM's per-token input table,
but a model that maps a whole sentence/document to ONE vector for similarity
search. Grounded: Qwen3-Embedding is the SAME Qwen3 base (0.6B-8B) repurposed.

6. How do you collapse a sequence of per-token vectors into a SINGLE vector for a
   whole text, and how do you train it so similar texts land close?

   Pooling (author: mean or last-token): mean pooling (average all token vectors),
   last-token pooling (final token's hidden state), or CLS pooling (BERT-style
   [CLS] vector). Decoder LLMs (Qwen3-Embedding) use LAST-TOKEN (EOS) pooling
   specifically because attention is causal — only the last token has attended to
   the whole sequence, so its final hidden state has seen everything (mean/CLS
   came from bidirectional BERT). Qwen3-Embedding uses the final <EOS> hidden
   state.
   Training (author: like word embedding models, similar high / opposite low):
   contrastive learning — pull POSITIVES together (paraphrases, query<->relevant-
   doc), push NEGATIVES apart (unrelated), using in-batch negatives + mined hard
   negatives. The word2vec link (author): same similarity-geometry idea as Q2 but
   at the SENTENCE level with explicit PAIR supervision (contrastive) instead of
   word2vec's context-window self-supervision. word2vec: "same context -> close";
   embedding model: "same meaning -> close".
   Recipe: run text through the LLM -> pool to one vector (last-token for decoders)
   -> contrastively fine-tune so similar texts are close. That's how the same
   Qwen3 base becomes Qwen3-Embedding.

   Where positives come from (author's insight — co-occurring text as positives):
   co-occurrence gives positive pairs for FREE, self-supervised — adjacent
   passages, two spans of the same doc, title<->body, question<->answer,
   query<->clicked-doc, hyperlinks. This is how Contriever/E5/GTE pretrain (with
   in-batch negatives). It is the SENTENCE-level analog of word2vec: word2vec used
   window co-occurrence for word similarity; embedding models use passage
   co-occurrence for sentence similarity — the distributional hypothesis one level
   up ("know a passage by the company it keeps"), connecting Q2<->Q6. Nuance:
   co-occurrence positives are noisy, so the usual pipeline is two-stage:
   (1) self-supervised contrastive pretraining on co-occurrence positives (cheap,
   huge), then (2) supervised contrastive fine-tuning on labeled pairs (MS MARCO,
   NLI) + hard-negative mining. Qwen3-Embedding uses this multi-stage recipe.

7. Bi-encoder vs cross-encoder, and how serving an embedding model differs from a
   generative LLM.

   (a) Bi-encoder (author): embeds documents and queries INDEPENDENTLY, so docs
   are embedded once offline into a vector DB; at query time embed just the query
   (one forward pass) + fast approximate-nearest-neighbor (ANN) search over
   precomputed vectors -> scales to millions, no per-query recompute. Cross-encoder
   (author): runs query+doc TOGETHER (no separable embedding), so nothing can be
   precomputed — one forward pass per (query,doc) pair, O(num_docs) per query;
   accurate (full token interaction) but can't scale. Pipeline: two-stage —
   bi-encoder retrieves top-k cheaply, cross-encoder reranks the k. Qwen ships both
   (Qwen3-Embedding bi + Qwen3-Reranker cross).

   (b) Serving difference (the capstone): an embedding model SKIPS almost
   everything we built for generation. No autoregressive decode loop — a single
   forward pass then pool to one vector. No growing KV cache, no decode phase, no
   stop tokens, no streaming detok. All prefill, no decode -> compute-bound and
   batch-friendly (encode huge batches in parallel) vs the memory-bandwidth-bound
   decode of a generative LLM. The bottleneck moves: generative LLM = decode loop
   + KV memory; embedding SYSTEM = forward-pass throughput for batch-encoding the
   corpus + the ANN index (vector DB, FAISS/HNSW), a different system from the
   model server. The decode/KV/streaming complexity vanishes; the new complexity
   is the vector index.

## Set C — Part A inference subsections (LM head, tied embeddings, cost)

8. After the transformer, you have a final hidden vector (1024-d). How do you get
   from it back to "which token comes next"? And in a tied model, what IS the LM
   head relative to E, and what does logit_i mean?

   Mechanism (author): a hidden x vocab matmul -> logits -> softmax -> probs ->
   sample. Correction of "it's the inverse of E": NOT the inverse — E is
   151936x1024, not square, so no inverse exists. The LM head reuses E as its
   TRANSPOSE (E^T, 1024x151936) so dims line up: hidden(1x1024) @ E^T = logits.
   Transpose = same weights flipped, not an inverse.
   What logit_i computes (author): logit_i = hidden . E[i] = dot product of the
   hidden vector with token i's embedding row. Geometrically a SIMILARITY —
   "which token's embedding does my hidden state most point toward?"; highest dot
   wins. A similarity search over the embedding rows, not an undo of the embedding.
   Elegant symmetry: input E maps ID->vector; output E^T scores the hidden state
   against every token's embedding. Same matrix both directions — literally the Q1
   "geometry = similarity" idea at the output: generation = find the token whose
   embedding is closest to where the hidden state landed.

9. The input embedding and the LM head use the SAME matrix E (tied), yet one is
   cheap and one expensive. Which is which and why?

   Author: token id -> embedding is one op (gather a single row of E, ~zero
   compute, cheap); hidden -> tokens has to compute similarity with EVERY
   embedding (all 151936 rows). Numbers: the output is a 1024x151936 matmul
   (~155.6M multiply-adds EVERY step) + softmax over 151936 — often the single
   most expensive op per token. So the vocab-size cost the tokenization chapter
   deferred lives HERE (the LM head), not the input lookup: the same 155.6M matrix
   (26% of Qwen3-0.6B, Q17) is cheap to READ (one row) but expensive to USE as
   output (all rows, every step) — input = "look up one", output = "score against
   all". This is exactly the FR-Spec thread (tokenization Q19): once a spec-decode
   drafter is ~1 layer, the LM head over a 128-256K vocab is ~half the drafter
   cost, so FR-Spec/VocabTrim/DynaSpec trim the DRAFT's output vocab to cut this
   matmul+softmax. Re-derives why the LM head is the drafter bottleneck.

## Set D — positional (hint), adjacent topics, multimodal cross-link

10. Positional embeddings (Part C — HINT ONLY, deep dive deferred to the RoPE
    recap chapter, per the author's scoping call). Token embedding = WHAT a token
    is; positional = WHERE it sits. Why position is needed: attention is
    permutation-invariant (a bag of tokens, no inherent order), so without a
    positional signal "dog bites man" and "man bites dog" look identical to the
    model. So a positional signal is ADDED to the token embeddings. Lineage
    (deferred to RoPE chapter): sinusoidal (Transformer 2017) -> learned absolute
    (BERT/GPT-2) -> RoPE rotary (Qwen3). Here: one-paragraph hint + hand off.

11. Count-based / LSA prehistory (Part D). Before neural word2vec, the first
    distributional DENSE vectors came from count-based methods: build a word x
    context co-occurrence matrix, weight it (PMI), and factorize with SVD (LSA/LSI,
    1990s-2000s). word2vec (2013) reached similar geometry with a neural predictor.
    A one-paragraph bridge between one-hot and word2vec — same distributional idea,
    counts-then-SVD instead of predict-then-learn.

12. Cosine vs dot + anisotropy (Part D). Cosine similarity = direction only
    (magnitude-invariant, normalizes out length); dot product = direction AND
    magnitude. Retrieval uses cosine (or normalized dot) to compare MEANING, not
    text length. Caveat: LLM/contextual embeddings are ANISOTROPIC — they cluster
    in a narrow cone rather than spreading over the sphere, so raw cosine can
    inflate similarities (everything looks somewhat alike); mitigations like
    whitening/normalization exist. A mention.

13. Segment / token-type embeddings (Part D). BERT added SEGMENT embeddings
    (sentence A vs B) plus [CLS]/[SEP] for its paired-input / next-sentence tasks —
    a third embedding summed with token + position. Decoder-only LLMs (GPT/Qwen)
    DROP them (single stream, no paired-sentence task). Encoder-era trivia; a
    mention, and it ties to the tokenization post-processor ([CLS]/[SEP]).

14. Multimodal cross-link (Part D / subsection 10). Image/audio patch vectors are
    projected (by the projector/connector) into the SAME embedding space as text
    token embeddings (image-tokenization Q1/Q6 — the "vision vectorizer"; soft
    context vectors do the same). So the embedding space is the shared destination
    for text, image, and compressed-context — the Future-Directions S7 frame: "the
    model's input is a vector sequence; discrete tokens are just one source of
    those vectors." Cross-link to 07_image_tokenization and future_directions S7;
    no new content, just the pointer.
