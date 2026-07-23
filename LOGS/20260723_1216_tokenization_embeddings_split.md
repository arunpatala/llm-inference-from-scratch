# Decision: tokenization and embeddings are two separate recap chapters

Raised mid-interview: does the embeddings/word-vectors material belong before,
after, together with, or interleaved into the tokenization chapter? Decided
via AskUserQuestion.

## Decision — Option A

Two separate Study-Note recap chapters, in pipeline order:

1. **Tokenization** — text → integer IDs. Ends deliberately on a cliffhanger:
   "you now have integers that mean nothing (seat numbers); making them *mean*
   something is the next chapter." Keeps the existing 60-question interview
   scaffold in `01_recap/01_tokenization/` unchanged.
2. **Embeddings / word vectors** (NEW recap chapter, not in the original recap
   topic list) — integer IDs → learned vectors. Distributional hypothesis,
   word2vec analogy arithmetic (`king - man + woman ≈ queen`), geometry =
   similarity, why the raw row is static but the transformer contextualizes it,
   dimensionality (Qwen3: 1024).

## Why (reasoning that drove it)

- Tokenization is discrete/combinatorial; embeddings are continuous/geometric.
  Different *kind* of topic → clean "one objective per chapter" seam
  (Diátaxis rule in `craft.md` §1, §6).
- Pipeline order (text → tokenize → embed) means the tokenizer chapter's
  dead-end ending motivates the embeddings chapter for free
  (motivate-before-mechanism, `craft.md` §4). Difficulty ramps discrete→continuous.
- Rejected a combined chapter: mixing the two is exactly the muddiness the
  style system warns reads as generic; and it would need a second interview
  anyway, so it saves nothing.
- Rejected embeddings-first: backwards dependency (explaining rows before what
  decides the rows).

## The one connecting thread (intentional, not a leak)

**Vocab size ↔ embedding-table cost.** Vocab size is a tokenizer decision, but
its main cost is the embedding table (Qwen3-0.6B: 151936×1024 = ~155.6M params
= ~26% of the model, tied). Introduce vocab-size-as-cost as a forward hook at
the end of the tokenization chapter; pay it off in the embeddings chapter as a
callback. This thread is what makes the two chapters feel authored together
rather than assembled.

## TODO created by this decision

- [ ] Build an embeddings interview scaffold (`01_recap/02_embeddings/01_questions.md`)
      parallel to the tokenization one, with sources — questions on word2vec/GloVe,
      distributional hypothesis, cosine similarity, analogy arithmetic,
      static-vs-contextual, dimensionality choice, tied embeddings (callback),
      and one or two coding exercises (real Qwen3 embedding cosine-sim demo —
      runs on the Mac, no CUDA). Do this AFTER the tokenization interview is done.
- [ ] When drafting the tokenization chapter, end it on the "IDs mean nothing"
      cliffhanger + the vocab-size forward hook.
