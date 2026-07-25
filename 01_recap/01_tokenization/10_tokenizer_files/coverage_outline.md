# Real tokenizer files: loading, saving, formats — coverage outline

What a "Tokenizer Files" chapter can cover. Grounded on the actual files
Qwen3-0.6B writes (inspected via save_pretrained), plus the general format
landscape (HF fast, HF legacy/slow, SentencePiece, tiktoken). Sources at bottom.

## 0. The concrete anchor — what Qwen3-0.6B actually saves

`tok.save_pretrained(dir)` writes exactly THREE files (verified):
- `tokenizer.json` (11,422,650 bytes ~ 11 MB) — the self-contained "fast" (Rust
  `tokenizers`) serialization. Holds the ENTIRE pipeline + vocab.
- `tokenizer_config.json` (694 bytes) — the HF transformers-level config (which
  class, special tokens, limits, flags).
- `chat_template.jinja` (4,168 bytes) — the Jinja2 chat template (newer HF saves
  it as its own file; older versions inlined it in tokenizer_config.json).

Notably ABSENT (vs older tokenizers): no vocab.json, no merges.txt, no
special_tokens_map.json, no added_tokens.json, no .model. Modern fast tokenizers
fold all of that into tokenizer.json. The file layout alone tells you the
tokenizer's generation/type.

## 1. Inside tokenizer.json — the pipeline as JSON (the big file)

Top-level keys (verified): version, truncation, padding, added_tokens,
normalizer, pre_tokenizer, post_processor, decoder, model. These map 1:1 onto the
standard pipeline (normalize -> pre-tokenize -> model -> post-process -> decode):
- `normalizer`: {"type": "NFC"} — grounds coverage-gap-1: Qwen DOES normalize.
- `pre_tokenizer`: {"type": "Sequence", ...} — the regex/byte-level pre-split
  (Q22).
- `model`: {"type": "BPE", vocab, merges, byte_fallback, ignore_merges,
  continuing_subword_prefix, end_of_word_suffix, fuse_unk, dropout} — the actual
  BPE table (the ~11 MB is almost all vocab + merges).
- `post_processor`, `decoder` — how tokens are reassembled/mapped back.
- `added_tokens`: 26 entries, each {id, content, special, single_word, lstrip,
  rstrip, normalized} — the reserved special tokens (Q15) with behavior flags.

## 2. The legacy / "slow" layout (GPT-2 style, or saving a slow tokenizer)

Splits what tokenizer.json unifies:
- `vocab.json` — token string -> integer ID map (= model.vocab).
- `merges.txt` — the ordered BPE merge rules, one per line, in learned order
  (= model.merges; the output of BPE training, 05_bpe).
- `special_tokens_map.json` — special-token ROLES (bos/eos/pad/unk/...) -> strings.
- `added_tokens.json` — tokens added beyond the base vocab.
- `tokenizer_config.json` — the config (same role as in the fast layout).
So: fast = one JSON; slow = vocab.json + merges.txt + the map/added/config files.

## 3. SentencePiece layout (Llama 1/2, T5, some Gemma)

- `tokenizer.model` — a BINARY protobuf (google/sentencepiece_model.proto)
  holding the learned pieces + scores + config. Not human-readable. In HF it's
  usually wrapped with tokenizer_config.json + special_tokens_map.json. Trade-off:
  compact/fast to load, but opaque (can't grep/diff), and needs the protobuf
  schema to inspect (vocab-expansion pain).

## 4. tiktoken layout (OpenAI GPT-3.5/4)

- `.tiktoken` file — the "mergeable ranks": byte-sequence -> rank/ID (the rank IS
  the merge order). Compact binary-ish. Convertible to vocab.json/merges.txt.
  Rust-backed, very fast; no normalizer/pipeline objects (byte-level + regex only).

## 5. Fast vs slow (the architectural distinction, coverage-gap-8)

- Fast: Rust `tokenizers` backend, self-contained tokenizer.json, gives offset
  mapping (gap 2) + high throughput (Q9/Q30). `is_fast == True`.
- Slow: pure-Python, vocab.json + merges.txt, no offsets, slower.
- `AutoTokenizer` prefers fast; `use_fast=False` forces slow.

## 6. Loading & saving mechanics

- `AutoTokenizer.from_pretrained(name_or_path)` — resolves files from the HF Hub
  (cached under ~/.cache/huggingface) or a local dir; picks the tokenizer class
  from tokenizer_config.json's `tokenizer_class`.
- `tok.save_pretrained(dir)` — writes the files; which files depends on fast vs
  slow and whether a legacy format exists.
- `trust_remote_code` / custom tokenizer classes — a security/trust surface.

## 7. Key config fields and what they mean (tokenizer_config.json)

Verified keys: tokenizer_class, model_max_length, bos_token, eos_token, pad_token,
unk_token, add_prefix_space, clean_up_tokenization_spaces, split_special_tokens,
errors, extra_special_tokens, backend. Connect to earlier threads:
- `errors="replace"` -> the byte-decode replacement-char behavior (Q21, invalid
  bytes -> the replacement char).
- `byte_fallback` (in model) -> the no-true-OOV guarantee (Q11).
- special-token fields -> Q6/Q15.
- `model_max_length` -> the context-window/truncation budget (gap 3, Q20).
- `add_prefix_space` -> the leading-space behavior (Q12).

## 8. Practical / inference concerns

- Version/format mismatch: tokenizer files must match the model checkpoint they
  were trained with, or output silently degrades (04_out_of_scope: tokenizer/
  model version mismatch). No error thrown.
- Adding tokens: `add_tokens` grows added_tokens + vocab; the MODEL's embedding
  table AND LM head must be resized to match (Q17 tie-embeddings) or loading/
  generation breaks.
- The chat_template.jinja ships AS PART OF the tokenizer and is executable code
  run per request (chat-template Q1/Q5 attack surface) — loading a tokenizer
  brings its template with it.
- Format conversion: SentencePiece .model -> HF tokenizer.json; tiktoken ->
  vocab/merges — needed for cross-ecosystem use.

## Sources
- HF issue #4777 (purpose of merges.txt / special_tokens_map / added_tokens):
  https://github.com/huggingface/transformers/issues/4777
- HF fast tokenizers: https://huggingface.co/docs/transformers/fast_tokenizers
- HF tokenizers Models API: https://huggingface.co/docs/tokenizers/en/api/models
- SentencePiece model proto:
  https://github.com/google/sentencepiece/blob/master/src/sentencepiece_model.proto
- tiktoken: https://github.com/openai/tiktoken · export to vocab/merges issue #60:
  https://github.com/openai/tiktoken/issues/60
- tiktoken vs SentencePiece:
  https://discuss.huggingface.co/t/what-is-the-difference-between-tiktoken-and-sentencepice-implements-about-bpe/86079
- Grounded: real Qwen3-0.6B save_pretrained output (this session).
