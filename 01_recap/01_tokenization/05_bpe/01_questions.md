# BPE — questions

Answer inline under each question, same as `../01_questions.md`.
Skip or write "not sure" where there's no real take yet.

1. BPE wasn't invented for NLP at all. It started as a general-purpose data
   compression algorithm (Gage, 1994), and Sennrich et al. repurposed it for
   translation/tokenization in 2016. What property of a compression
   algorithm makes it a good fit for building a subword vocabulary?

   _(answer here)_

2. Walk through the actual training algorithm as you understand it: what's
   the starting vocabulary, what gets counted, what gets merged, and when
   does training stop?

   _(answer here)_

3. "Byte-pair encoding" merges pairs of what, exactly, at each step? Is it
   always literally a pair of raw bytes, or does that change between the
   original compression algorithm, character-level BPE, and byte-level BPE?

   _(answer here)_

4. Training happens once, over a large corpus, and produces an ordered list
   of merge rules. Encoding a brand-new piece of text at inference time
   doesn't recount frequencies at all, it just replays that same frozen
   list of merges, in the order they were learned. Why does it have to be
   that specific order, rather than any order that produces a valid result?

   _(answer here)_

5. Question 22 (in `../01_questions.md`) already covered that
   GPT-2/GPT-4-style tokenizers pre-split text by category (letters,
   digits, punctuation, whitespace) with a regex before BPE runs.
   Mechanically, where does that pre-split sit relative to the merge loop?
   Does BPE run once per whole document, or once per pre-split chunk?

   _(answer here)_

6. If a byte-level BPE tokenizer starts with a base vocabulary of 256 raw
   bytes and ends with a vocabulary of size V, how many actual merge
   operations did training perform to get there? Do the arithmetic for
   Qwen3's tokenizer using its real `vocab_size` from
   `../03_algorithms_and_qwen3.md`.

   _(answer here)_

7. A tokenizer's efficiency is often measured in bytes-per-token: how many
   raw UTF-8 bytes, on average, get compressed into a single token. What
   does a higher bytes-per-token number actually buy you at inference time,
   concretely?

   _(answer here)_

8. Naive BPE training implementations can take hours on a large corpus;
   well-engineered ones (priority queues, incremental count updates instead
   of full rescans) can do the same job in seconds. Since you'll never actually
   train your own tokenizer for this project, why does this efficiency fact
   matter at all to an inference engineer?

   _(answer here)_

9. A real research finding: choosing BPE merges by anything close to
   random, rather than strictly by "most frequent pair," barely changes
   downstream model quality. Does that surprise you? What does it suggest
   about how much the specific greedy frequency heuristic actually matters,
   versus just having *some* consistent subword vocabulary?

   _(answer here)_

10. Once vocab size is fixed and training is done, the merge list is frozen
    for the model's entire lifetime; no merge is ever added or removed
    later. What would have to be true for a deployed model to need a
    genuinely new tokenizer, not just a bigger one?

    _(answer here)_

## Sources (for citation when the chapter is written)

- [Sennrich, Haddow, Birch (2016) — Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/) — Q1
- [Hugging Face LLM course — Byte-Pair Encoding tokenization](https://huggingface.co/learn/llm-course/en/chapter6/5) — Q2, Q4
- [Sebastian Raschka — Implementing a BPE tokenizer from scratch](https://sebastianraschka.com/blog/2025/bpe-from-scratch.html) — Q2, Q3
- [Karpathy: Let's build the GPT tokenizer](https://simonwillison.net/2024/Feb/20/lets-build-the-gpt-tokenizer/) — Q3, Q5
- [From Hours to Seconds: Optimising BPE Tokeniser Training](https://medium.com/@logan_16888/from-hours-to-seconds-optimising-bpe-tokeniser-training-f4234300d03e) — Q8
- ["What changes when you randomly choose BPE merge operations? Not much"](https://arxiv.org/pdf/2305.03029) — Q9
