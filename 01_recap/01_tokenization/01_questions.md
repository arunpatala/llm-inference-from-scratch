# Tokenization — questions

Answer inline under each question, replacing the placeholder. Skip or write
"not sure" where there's no real take yet — a thin answer is a signal to ask
a follow-up later, not something to fill in generically.

## Set A — building the chapter

1. In your own words: what is a token, and why does the model need numbers
   instead of raw text?

   _(answer here)_

2. Why subword tokenization (BPE-style) instead of whole-word or
   character-level — what problem does splitting into pieces like "token" +
   "ization" actually solve?

   _(answer here)_

3. Qwen3's vocab is ~151k entries — bigger vocab is a trade-off, not a free
   win. What's the trade-off as you understand it?

   _(answer here)_

4. We're using the Instruct model, so a raw prompt gets wrapped in a chat
   template before tokenization. What do you think that template actually
   contains, and why can't we just tokenize the user's raw text directly?

   _(answer here)_

5. Walk through the actual pipeline: raw prompt string → what happens →
   tensor of token IDs the model consumes. What are the steps as you
   understand them?

   _(answer here)_

6. What special tokens do you expect matter here (BOS/EOS/pad/end-of-turn),
   and why does the model need explicit markers for where a turn ends?

   _(answer here)_

7. Have you personally hit a bug or surprise caused by tokenization
   specifically — a weird split, whitespace handling, padding side, anything
   like that? What happened?

   _(answer here)_

8. On the way out: tokens get sampled one at a time during generation. What
   has to happen to turn each new token ID back into the text the user
   actually sees, streaming?

   _(answer here)_

9. Does tokenization run on CPU or GPU, and does its cost matter at all next
   to the model's forward pass, or is it noise?

   _(answer here)_

10. One sentence for a colleague: why does tokenization deserve its own
    section in an inference book instead of a two-line footnote?

    _(answer here)_

## Set B — grounded in documented tokenizer behavior

11. Byte-level BPE guarantees there's no true "unknown token" — even a
    character never seen in training still encodes via raw bytes. Why does
    that guarantee matter for a production inference engine, versus a
    tokenizer that could just fail on unseen input?

    _(answer here)_

12. `" hello"` and `"hello"` (leading space or not) tokenize to different IDs
    entirely. Why does that happen mechanically, and has it ever bitten you
    or surprised you?

    _(answer here)_

13. Case matters too — `"BPE"` and `"bpe"` are different tokens. Obvious once
    you know how BPE merges work, or does it still feel like a gotcha?

    _(answer here)_

14. Many modern tokenizers, including Qwen's, deliberately split numbers into
    individual digits instead of merging `"12345"` into one token. Why would
    a tokenizer designer choose that on purpose?

    _(answer here)_

15. Special tokens like `<|im_start|>`/`<|im_end|>` aren't produced by the BPE
    merge algorithm at all — they're injected as reserved vocab entries that
    bypass merging entirely. Why do they need to be handled separately?

    _(answer here)_

16. "Glitch tokens" (the SolidGoldMagikarp phenomenon) are real vocab entries
    that make models output garbage when they appear, caused by a mismatch
    between the tokenizer's training corpus and the model's training corpus.
    Does that risk apply to a small model like Qwen3-0.6B, in your view?

    _(answer here)_

17. Vocab size isn't free — it directly sets the size of two of the model's
    biggest weight matrices (embedding table and LM head). For Qwen3-0.6B's
    ~151k-token vocab, what do you think that costs in raw parameters, and
    does that change how you think about vocab size as a design choice?

    _(answer here)_

18. When batching multiple prompts of different lengths (Module 2's
    territory), padding side — left vs. right — actually affects
    correctness, not just style. Do you already know why, or want to reason
    through it together here?

    _(answer here)_

19. Speculative decoding (Module 6) has a hard requirement: draft and target
    models must share the exact same tokenizer and vocabulary, because the
    algorithm compares probability distributions over identical token IDs —
    mismatch collapses the acceptance rate toward zero. Does that settle how
    you'd pick a draft model for Qwen3-0.6B, or is it still open?

    _(answer here)_

20. Different model families use completely different tokenizers, so "1000
    tokens" means a different amount of actual text depending on the model.
    Why does that matter specifically for an inference engine — not just as
    a token-counting curiosity?

    _(answer here)_

## Set C — harder

21. When a multibyte UTF-8 character (an emoji, a non-Latin script character)
    is split across two token boundaries, streaming generation token-by-token
    can emit a broken partial byte sequence — visible as a "�" replacement
    character mid-stream. This is a real bug that's shipped in more than one
    serving engine. What does a correct streaming decoder need to do
    differently from decoding each new token in isolation?

    _(answer here)_

22. GPT-2's tokenizer (and GPT-4's) doesn't run BPE merges on the raw byte
    stream directly — it first splits text into categories (letters, digits,
    punctuation, whitespace) with a regex, and only merges within a category.
    What goes wrong if you skip that pre-split and let BPE merge freely
    across category boundaries?

    _(answer here)_

23. "Token healing" exists because a prompt can end at a boundary the model
    never actually saw ending there in training — e.g. a prompt ending in
    "New Ent" gets tokenized as-is, even though training data would usually
    continue that into one longer token like "Enterprise". What's the actual
    failure mode this causes at generation time, and how would you fix it?

    _(answer here)_

24. Tokenizer "fertility" (tokens produced per word) varies enormously by
    language — English measures around 1.2-1.4 tokens/word, some languages
    measure 10-16 tokens/word for equivalent content. Context length and KV
    cache budget are both measured in tokens, not words or characters. What
    does that actually mean for a non-English user hitting the same
    prompt-length limit on the same inference engine?

    _(answer here)_

25. Qwen3's own model family makes an explicit, size-dependent choice: the
    0.6B/1.7B/4B models tie the input embedding and output LM-head weights
    (one matrix, used twice), while the 8B+ models use two separate
    matrices. *(Verify this against Qwen3-0.6B-Instruct's actual config.json
    before it goes in the chapter — this is a secondary-source claim.)* Why
    would tying make sense at 0.6B specifically but not at 8B?

    _(answer here)_

26. Gradient-based jailbreak attacks (GCG and its variants) search for
    adversarial token sequences using gradients through the model's own
    vocabulary — and the resulting attack is tied to that specific
    tokenizer, so a suffix that works against one model's tokenizer often
    doesn't transfer to a model with a different vocab. What does that imply
    about tokenization as part of a model's actual attack surface, rather
    than just an encoding detail?

    _(answer here)_

27. Given a fixed tokenizer (fixed merge rules), is BPE encoding
    deterministic — does the exact same input string always produce the
    exact same token IDs? If yes, where does the "boundary" problem in Q23
    (token healing) actually come from, if not from encoding itself being
    ambiguous?

    _(answer here)_

28. Prefix caching (a future chapter) reuses KV cache blocks for requests
    sharing an identical token prefix — reuse is exact-match at the token
    level, not the text level. What's a realistic way a system-prompt
    template could silently break prefix-cache hit rate, purely through how
    it tokenizes?

    _(answer here)_

29. A tokenizer's merge vocabulary is learned once, from one training corpus,
    then frozen for the model's entire life. What does it concretely mean
    for a tokenizer to be a bad fit for a domain — e.g. feeding a lot of
    source code or chemistry notation through a tokenizer trained mostly on
    English prose?

    _(answer here)_

30. Is tokenization ever actually the bottleneck in a serving engine, or is
    it always negligible next to the GPU forward pass? Under what realistic
    condition — request volume, generation length, tokenizer implementation
    — could it stop being negligible?

    _(answer here)_

## Sources (for citation when the chapter is written)

- [Karpathy: Let's build the GPT tokenizer](https://simonwillison.net/2024/Feb/20/lets-build-the-gpt-tokenizer/) — Q11, Q12, Q13, Q22
- [SolidGoldMagikarp and other glitch tokens](https://www.kith.org/words/2023/12/10/solidgoldmagikarp-and-other-glitch-tokens/) — Q16
- [Fishing for Magikarp: detecting under-trained tokens](https://arxiv.org/abs/2405.05417) — Q16
- [vLLM issue #7252 — draft/target vocab mismatch](https://github.com/vllm-project/vllm/issues/7252) — Q19
- [Speculative decoding in production: hidden traps](https://tianpan.co/blog/2026-04-17-speculative-decoding-production-hidden-traps) — Q19
- [UTF-8 Plumbing: Byte-level Tokenizers Unavoidably Enable LLMs to Generate Ill-formed UTF-8](https://openreview.net/forum?id=8ExXncFpf6) — Q21
- [llama.cpp issue #8691 — tokenizer not working on partial UTF-8 bytes](https://github.com/ggml-org/llama.cpp/issues/8691) — Q21
- [Guidance docs: token healing](https://guidance.readthedocs.io/en/latest/example_notebooks/tutorials/token_healing.html) — Q23, Q27
- [The Art of Prompt Design: Prompt Boundaries and Token Healing](https://medium.com/data-science/the-art-of-prompt-design-prompt-boundaries-and-token-healing-3b2448b0be38) — Q23, Q27
- [Language Model Tokenizers Introduce Unfairness Between Languages](https://arxiv.org/pdf/2305.15425) — Q24
- [The Tokenizer Tax Across 24 European Languages](https://arxiv.org/html/2605.24718) — Q24
- [Weight tying in language models — when and why LLMs share embeddings](https://medium.com/@vishal09vns/weight-tying-in-language-models-when-and-why-llms-share-embeddings-7f5f5376c625) — Q25
- [Qwen/Qwen3-Embedding-8B discussion — tie_word_embeddings: false](https://huggingface.co/Qwen/Qwen3-Embedding-8B/discussions/17) — Q25
- [What Is the GCG Attack? — FutureAGI Guide](https://futureagi.com/glossary/gcg-attack/) — Q26
