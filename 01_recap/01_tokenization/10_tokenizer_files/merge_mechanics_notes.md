# How merge rules are used, and how encoding stays fast (chapter notes)

Arose from inspecting the real tokenizer.json (vocab 151,643 + merges 151,387).
Grounded on Qwen3-0.6B; the traces below were replayed from the actual merge list
and verified to MATCH tok.encode.

## How merge rules are used (the encoding algorithm)

Real trace, "hugging" (replaying Qwen's merges; rule # = position in the merge
list = learned order = priority):

    start: ['h','u','g','g','i','n','g']            (individual characters)
     step 1: merge (i,n)     rule #2      -> h u g g [in] g
     step 2: merge (in,g)    rule #31     -> h u g g [ing]
     step 3: merge (u,g)     rule #512    -> h [ug] g ing
     step 4: merge (ug,g)    rule #2340   -> h [ugg] ing
     step 5: merge (ugg,ing) rule #35012  -> h [ugging]
    final: ['h','ugging']  ids=[71, 35268]   (matches the tokenizer)

The algorithm:
1. Break the word into characters/bytes (base units).
2. Look at ALL adjacent pairs; find the one whose merge rule has the LOWEST rule
   number (learned earliest). The rule number is the priority.
3. Merge that pair into one piece.
4. Repeat until no adjacent pair has a rule; then stop.
5. Look up the final pieces' IDs in the vocab.

Three things the trace shows:
- Rank = priority, not left-to-right: (i,n)=rule #2 merges before (u,g)=rule #512
  even though u,g sits earlier positionally. You scan the whole word each step and
  apply the earliest-learned applicable merge. (Rule #2 = "in" was the 3rd merge
  Qwen ever learned, after the two space-merges — it's that common.)
- Rank IS the frozen learned order, which is why encoding is deterministic and
  reproducible (Q27; the 05_bpe/merge_order_matters exercise on real data).
- It stops when no rule applies, which is why rare words fragment: "hugging" ->
  ['h','ugging'] because there's no (h, ugging) rule; "lowest" -> ONE token
  (89898). Common word -> one token; rarer -> pieces (the Q2/Q24 frequency story,
  visible in the merges).

So vocab + merges are the two halves: merges = the rank-ordered RECIPE for
combining bytes into tokens (the engine); vocab = the final LOOKUP from pieces to
IDs. tokenizer.json needs both.

## How it stays fast — you do NOT scan all 151k rules

- Merges are a HASH MAP {(a,b) -> rank}, so checking a pair is O(1), not a scan.
  Per step you look only at the current word's adjacent pairs (at most L-1, L =
  word length ~5-15 chars), do L hash lookups, take the min-rank pair, merge.
  You never touch the other ~151k rules.
- Steps <= L-1 (each merge shortens the word by 1). So the reference algorithm is
  ~O(L^2) PER WORD, and L is tiny. Cost is bounded by WORD length, not vocab/rule
  count.
- Pre-tokenization (Q22) matters for speed too: the regex pre-split chops text
  into short chunks first, so BPE only runs inside a word-sized window — never
  across the document, never cross-word merges.

Production speedups (fast Rust tokenizers, tiktoken):
- Word-level CACHING (the big one): Zipf's law means a few words dominate, so
  memoize word -> tokens; most of a document hits the cache. tiktoken + HF fast
  tokenizers both do this.
- Priority queue/heap instead of re-scanning each step: pop the best merge, update
  only the two affected neighbors.
- Rust + parallel chunk processing (the is_fast backend).
- This is why we measured ~1M tokens/sec on CPU in Q9 (0.58 ms / 601-token
  prompt) — hash lookups + caching + Rust, not a rule march.

Cost-profile contrast: ENCODING (apply learned merges) is cheap (O(L^2)/short
word, cached). TRAINING (find the merges by counting pairs across a corpus,
repeatedly) is the expensive part — the "hours vs seconds" issue (05_bpe Q8),
solved with priority queues + incremental counts. Frontier: Incremental BPE (ICML
2026, streaming) and BlockBPE (parallel) push encoding speed further, since at
high throughput tokenizer speed is the input-side bottleneck (Q9/Q30).
