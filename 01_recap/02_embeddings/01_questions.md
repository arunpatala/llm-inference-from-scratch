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
