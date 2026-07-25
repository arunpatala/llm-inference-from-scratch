# Tokenizer files (loading, saving, formats) — questions

Answer inline under each question, same as the other question files. Skip or
write "not sure" where there's no real take yet — a thin answer is a signal to
ask a follow-up, not to fill it in generically. Grounded facts and the format
landscape are in `coverage_outline.md`.

## Set A — what's actually on disk

1. When you save Qwen3-0.6B's tokenizer you get exactly three files:
   `tokenizer.json` (~11 MB), `tokenizer_config.json` (~700 bytes), and
   `chat_template.jinja` (~4 KB). What does each one hold, and why is one of them
   thousands of times bigger than the others?

   _(answer here)_

2. `tokenizer.json` is described as holding the whole tokenizer "pipeline." Given
   the standard pipeline — normalize -> pre-tokenize -> model -> post-process ->
   decode — what sections would you expect inside the file, and which pipeline
   stage does each map to? (Qwen's actually has: normalizer, pre_tokenizer,
   model, post_processor, decoder, added_tokens.)

   _(answer here)_

3. Older GPT-2-style tokenizers don't save a single `tokenizer.json` — they save
   `vocab.json` + `merges.txt` (plus a couple of small JSONs). What does each of
   those two files contain, and what's the relationship between them and the
   `model` section inside a modern `tokenizer.json`?

   _(answer here)_

4. `tokenizer_config.json` is ~700 bytes; `tokenizer.json` is ~11 MB. What's the
   division of labor between them — what kind of information lives in the tiny
   config vs the huge file, and why does that split make sense?

   _(answer here)_

## Set B — formats and the fast/slow split

5. SentencePiece models (Llama 1/2, T5) ship a single binary `tokenizer.model`
   (a protobuf) instead of JSON. What's the trade-off of a binary-protobuf format
   versus HF's readable JSON for a tokenizer file — think about loading speed,
   inspectability, and debugging.

   _(answer here)_

6. tiktoken (GPT-3.5/4) uses `.tiktoken` files storing "mergeable ranks" instead
   of `vocab.json` + `merges.txt`. What do you think "mergeable ranks" means, and
   why might OpenAI use a different format from Hugging Face?

   _(answer here)_

7. What is the "fast vs slow" tokenizer distinction, and how does it show up in
   the SAVED FILES (which files each produces) and in what the tokenizer can DO
   (a capability the fast one has that the slow one doesn't)?

   _(answer here)_

## Set C — fields, consistency, and gotchas

8. The `added_tokens` section lists 26 special tokens, each with flags like
   `special`, `single_word`, `lstrip`, `rstrip`, `normalized`. Why does a special
   token need these flags — pick one (e.g. `lstrip`/`rstrip` or `normalized`) and
   say what would go wrong without it.

   _(answer here)_

9. Loading a tokenizer for a model checkpoint requires them to MATCH. What,
   concretely, has to match between the tokenizer files and the model weights,
   and what's the failure mode when they don't (does it error, or something
   worse)?

   _(answer here)_

10. The `chat_template.jinja` is saved AS PART OF the tokenizer, not the model.
    Why does the tokenizer own the chat template, and what does "the tokenizer
    ships executable code that runs on every request" imply for loading a
    tokenizer from an untrusted source?

    _(answer here)_

11. If you add new tokens to a tokenizer (`add_tokens`) and save it, what changes
    in the files — and what else, in the MODEL, must change to stay consistent?
    (Hint: recall the tied embedding table and LM head from the vocab-size
    discussion.)

    _(answer here)_

12. `tokenizer_config.json` includes `errors="replace"` and the model has
    `byte_fallback`. Connect each of these config fields to a guarantee or
    behavior you already reasoned about earlier in the chapter (the no-true-OOV
    guarantee, and the partial-UTF-8 / invalid-byte handling).

    _(answer here)_

## Sources (for citation when the chapter is written)

- [HF issue #4777 — purpose of merges.txt / special_tokens_map / added_tokens](https://github.com/huggingface/transformers/issues/4777)
- [HF docs: Fast tokenizers](https://huggingface.co/docs/transformers/fast_tokenizers)
- [HF Tokenizers: Models API](https://huggingface.co/docs/tokenizers/en/api/models)
- [SentencePiece model.proto](https://github.com/google/sentencepiece/blob/master/src/sentencepiece_model.proto)
- [tiktoken](https://github.com/openai/tiktoken) · [export to vocab/merges (issue #60)](https://github.com/openai/tiktoken/issues/60)
- [tiktoken vs SentencePiece (HF forum)](https://discuss.huggingface.co/t/what-is-the-difference-between-tiktoken-and-sentencepice-implements-about-bpe/86079)
- [Converting SentencePiece tokenizer.model to HF tokenizer.json](https://www.oreateai.com/blog/technical-practice-converting-sentencepieces-tokenizermodel-to-huggingface-format-tokenizerjson/c812e81744186ec6c68b7f7af09e47b9)
- Grounded: real Qwen3-0.6B `save_pretrained` output (inspected this session).
