# Tokenization chapter — status & remaining TODO

## Research: DONE (for an inference book's scope)

The tokenization chapter's research/interview phase is complete for what an
inference book needs. What exists:
- Main interview Q1-30 (`01_questions.md`) — author's voice, grounded on Qwen3.
- Image/multimodal interview Q1-10 (`07_image_tokenization/`).
- Tokenizer-files sub-module (`10_tokenizer_files/`) — the 3 files, vocab+merges
  with a verified merge trace + efficiency, pipeline-as-JSON, files-vs-code, the
  26 Qwen3 special tokens, the chat template (tools + thinking + reasoning-vs-
  prefix-cache).
- Notes: `02_char_vs_word_vs_subword`, `02a_bytes_vs_chars_vs_tokens`,
  `03_algorithms_and_qwen3`, `04_out_of_scope`, `09_pipeline_stages_and_template_
  extensions`, plus the 10_tokenizer_files notes.
- Demos (5), verified exercises (3), 05_bpe + 06_chat_template sub-modules.
- Future Directions: 7 sections (`research/future_directions.md`).
- Research reference: 3 survey rounds, ~90 cited papers (`research/frontier_
  questions.md`).
- History: dense + readable (`history/`).
- Coverage gap analysis vs standard curricula (`COVERAGE_GAPS.md`).

## Remaining TODO

1. **BPE inference-time logic — consolidate.** The encoding algorithm (replay
   merges by rank, stop-when-no-rule, hash-map/O(L^2)/caching efficiency) is
   captured in `10_tokenizer_files/merge_mechanics_notes.md` and the
   `05_bpe/encode_with_merges` exercise. For the final chapter, consolidate this
   into one clean "how BPE encodes at inference" section. (Training logic — count
   pairs, merge most frequent — lives in `05_bpe/train_toy_bpe`; include only as
   background, since this is an inference book.)

2. **Final chapter creation.** Turn the notes/interview/research into an actual
   deliverable. The version to produce is a choice — see the `chapter-versions`
   skill. Likely target for this book: the inference-focused complete chapter
   (+ optional exercises, the Core-Module tier per the project plan).

## Not doing (out of scope for an inference book)
Tokenizer *training* deep-dive, WordPiece/Unigram build-it-yourself, corpus
datasheets — background only. See `04_out_of_scope.md` and `COVERAGE_GAPS.md`
(intentionally-skipped list).
