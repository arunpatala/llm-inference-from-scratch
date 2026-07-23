# Tokenization: out of scope

Real, documented topics this chapter doesn't go deep on. Worth knowing they
exist; none are needed to build Module 00.

- **Reasoning/thinking tokens.** Qwen3 toggles thinking mode with `<think>`/
  `</think>` tags and a configurable `thinking_budget`; reasoning content
  counts against the token budget like anything else. See
  [Qwen3's thinking-budget docs](https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/thinking_budget.md).
- **Tool-calling tokens.** A separate mechanism from thinking tokens: dedicated
  `<tool_call>`/`<tool_response>` tags, sometimes with explicit
  decision tokens so the model's first generated token commits to "call a
  function" vs. "answer directly." See
  [Hugging Face: messages and special tokens](https://huggingface.co/learn/agents-course/en/unit1/messages-and-special-tokens).
- **Multimodal placeholder tokens in a text-only model.** Qwen3-0.6B's
  vocab reserves `<|vision_start|>`, `<|image_pad|>`, `<|video_pad|>` despite
  being text-only, almost certainly because the vocab is shared across the
  whole Qwen3 family.
- **Vision/audio tokenization as a different mechanism entirely.** VQ-VAE-style
  codebooks quantize continuous image features into discrete IDs. It's the
  same underlying idea, discretize the world into vocab entries, but a
  completely different mechanism from BPE merges. See the VQ-VAE tokenizer
  literature,
  e.g. [End-to-End Vision Tokenizer Tuning](https://arxiv.org/pdf/2505.10562).
- **Tokenizer-free / byte-latent architectures.** Active research:
  [Meta's Byte Latent Transformer](https://arxiv.org/html/2412.09871v1),
  MEGABYTE, ByT5, and CANINE all dynamically patch raw bytes instead of
  using a fixed BPE vocab, reportedly matching BPE-based models at scale.
  This is
  character-level tokenization's sequence-length problem (file
  `02_char_vs_word_vs_subword.md`) solved by adaptive patch boundaries
  instead of static merges.
- **Unicode normalization / homoglyph attacks.** A sharper security angle
  than question 26's GCG attack, and needs no gradient access at all: a
  Cyrillic "і" swapped for a Latin "i" produces a different token sequence
  that reads as identical text to a human but bypasses token-level safety
  filters. See
  [confusable-LLM attack vectors](https://paultendo.github.io/posts/confusable-llm-attack-vectors/).
- **Tokenizer/model version mismatch.** Loading a checkpoint with a
  tokenizer that doesn't exactly match what it was trained with produces
  silently degraded or garbage output, no error thrown. Boring, and easy to
  hit precisely because it's boring.
- **Tokenizer startup cost.** Building the merge-rule structure for a
  ~152k-entry vocab isn't free; a fast Rust-backed tokenizer versus a pure
  Python one affects a server's cold-start latency specifically, distinct
  from question 30's steady-state throughput question.
