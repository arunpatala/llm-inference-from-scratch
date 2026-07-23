# Tokenization algorithms, and what Qwen3-0.6B actually uses

## Common algorithms

- **BPE (byte-pair encoding)**: the classic approach. Iteratively merge the
  most frequent adjacent pair in the training corpus until reaching a target
  vocab size.
- **Byte-level BPE (BBPE)**: BPE run over raw bytes (256 base symbols)
  instead of Unicode characters. What GPT-2, GPT-4, and Qwen use. Guarantees
  zero out-of-vocabulary tokens by construction, since every possible byte is
  already in the base vocabulary.
- **WordPiece**: BERT's tokenizer. Same iterative-merge idea as BPE, but the
  merge criterion maximizes training-data likelihood rather than raw
  frequency.
- **Unigram language model**: the opposite direction from BPE. Start from a
  large candidate vocabulary and prune it down, keeping whichever tokens
  maximize a probabilistic objective. Used by SentencePiece's "unigram" mode,
  and by T5, ALBERT, XLNet.
- **SentencePiece**: not an algorithm itself, a library implementing BPE or
  Unigram directly over raw text. Treats whitespace as an explicit symbol
  (`▁`) so tokenization stays fully reversible and language-agnostic without
  separate pre-tokenization rules. Used by Llama 1/2 and T5.

## What Qwen3-0.6B actually uses

Pulled from the real `config.json` and `tokenizer_config.json` on
[Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), not a secondary
source:

- Tokenizer class: `Qwen2Tokenizer`. Qwen3 didn't train a new tokenizer at
  all, it reuses Qwen2's entirely.
- Algorithm: byte-level BPE, built on OpenAI's `tiktoken` (same lineage as
  GPT-3.5/4's tokenizer), with the vocabulary specifically extended for
  multilingual and Chinese coverage.
- `vocab_size`: **151936** exactly.
- No `unk_token` defined. Consistent with byte-level BPE's "no OOV, ever"
  guarantee (see `02_char_vs_word_vs_subword.md`): there's nothing for an
  unknown-token slot to do.
- `eos_token`: `<|im_end|>`, `pad_token`: `<|endoftext|>`. `bos_token` is
  `null`: Qwen3 doesn't prepend a beginning-of-sequence token at all, despite
  `bos_token_id: 151643` existing in the model config. The ID is reserved
  but unused by the tokenizer's default behavior.
- `tie_word_embeddings: true`. This confirms question 25's claim for this
  exact model, not just the secondary source that first raised it.

Other architecture fields from the same `config.json`, useful later for
Module 00: `hidden_size` 1024, `num_attention_heads` 16,
`num_key_value_heads` 8 (GQA, 2 query heads per KV head), `head_dim` 128,
`num_hidden_layers` 28, `rope_theta` 1000000, `max_position_embeddings`
40960.

## The tied-embedding number, worked out

The tied matrix is `vocab_size × hidden_size` = `151936 × 1024` = **~155.6M
parameters**. Qwen3-0.6B has roughly 600M parameters total, so this single
matrix accounts for about **26% of the entire model**. Tied, it's paid for
once and used twice: as the input embedding and as the output LM head.

Untied, at this same vocab size, the model would need a second ~155.6M
matrix for the LM head alone, pushing total size up by roughly a quarter for
zero architectural change otherwise. That's the concrete answer to "why does
tying make sense at 0.6B but not at 8B" (question 25): the embedding table's
cost is fixed by vocab size, not model size, so it's a shrinking fraction of
the total as the model grows. Untying stops being expensive relative to
everything else.
