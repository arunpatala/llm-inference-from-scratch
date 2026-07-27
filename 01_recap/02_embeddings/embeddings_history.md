# How embeddings are learned: GloVe vs end-to-end (historical line)

Answers a common confusion for the "how the numbers are learned" subsection
(layout §4). Grounded via search.

## Direct answer: modern LLM token embeddings are trained END-TO-END from scratch

Randomly initialized, learned jointly with the whole model by next-token
prediction. NOT produced by word2vec/GloVe. You *could* initialize the embedding
table with pretrained word vectors, but it is NOT standard practice. So it's not
"GloVe-type", and not a "mix" for a from-scratch LLM. The "mix" appears only in
adaptation (below).

## Historical line

1. One-hot / count-based (pre-2013): words as sparse one-hot vectors or
   co-occurrence counts (LSA). No learned dense embeddings; high-dim, sparse, no
   built-in similarity.
2. Static learned word embeddings (2013-2017) — the "GloVe-type" era: word2vec
   (Mikolov 2013), GloVe (Pennington 2014), FastText (2016). A dedicated algorithm
   learns ONE dense vector per word from a corpus, SEPARATELY, and you plug those
   fixed vectors in as input features / initialization for a downstream RNN/CNN.
   Static: one vector per word, context-independent.
3. The contextual shift (2018) — the turning point: ELMo (biLSTM LM) made
   embeddings contextual (vector depends on the sentence); then transformer-based
   BERT and GPT-1 trained the embedding table END-TO-END as part of the model.
   Contextual >> static, so the standalone word2vec/GloVe pipeline was superseded.
4. Modern LLMs (GPT-2/3 2019-2020 -> Llama/Qwen 2023+): the SUBWORD embedding
   table is randomly initialized and trained end-to-end from scratch by
   next-token prediction — no word2vec/GloVe. It's the model's first layer; the
   transformer contextualizes; small models tie it with the LM head.

## Where a "mix" / informed init DOES appear (connects to the tokenization chapter)

- Model adaptation (not from-scratch): extending a pretrained LLM to a new
  language/vocabulary -> new tokens get informed init (mean of their subtoken
  embeddings, or transplantation like ZeTT/OMP — the tokenizer-transplantation
  Future Direction), then continued training. Warm-start, not GloVe.
- Rare-token quality mechanism (a strong Q16 callback): only embedding rows for
  tokens PRESENT in a batch get gradient updates, so rare tokens accumulate far
  fewer updates -> lower-quality embeddings — exactly the glitch/undertrained-token
  phenomenon (tokenization Q16). And it's WHY subwords help: BPE decomposes rare
  words into FREQUENT subwords, so even uncommon vocabulary gets adequate training
  signal. The "how learned" subsection should point back at Q16.

## Sources
- Token initialization methods (emergentmind); "From Tokens To Vectors" embedding-
  layer training (ml-digest); dailydoseofds tokenization+embeddings.
- word2vec (Mikolov 2013, arXiv:1301.3781), GloVe (Pennington 2014), FastText
  (Bojanowski 2016/2017).
- Contextual shift: ELMo (Peters 2018), BERT (Devlin 2018), GPT-1 (Radford 2018).
- Informed new-token init / adaptation: Token Distillation (arXiv:2505.20133),
  Learned Embedding Propagation (arXiv:2412.21140), mean-of-subtokens baselines.
