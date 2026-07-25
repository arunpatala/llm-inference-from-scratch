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

   Roles (author): tokenizer.json = the token-str<->id data; tokenizer_config
   = special tokens + settings; chat_template.jinja = the Jinja messages->string
   template. All correct.

   Why 11 MB — it holds TWO big data structures, not one (author had the vocab,
   the merges were taught). Grounded on the real file:
   - vocab: 151,643 token->id entries (~4.7 MB), e.g. ('!',0), ('"',1).
   - merges: 151,387 ORDERED merge rules (~4.5 MB), e.g. [Ġ,Ġ], [ĠĠ,ĠĠ], [i,n].
   Together ~9.2 MB of the 11; the rest is the small pipeline objects.
   Why both are needed: vocab = the ingredients (which tokens exist + IDs, used
   for the final string<->id lookup); merges = the recipe (HOW to build tokens
   from bytes — apply in learned order until none apply, exactly Q27 determinism
   and the 05_bpe/encode_with_merges exercise). The example merges are the same
   kind of list train_toy_bpe produced: Qwen's first merges are "two spaces",
   "four spaces" (code indentation), then "in". (Mild redundancy: the vocab is
   derivable from merges but stored explicitly for fast lookup.)

   Division by kind: tokenizer.json = the tokenizer's DATA (vocab + merges +
   pipeline = what the tokenizer IS); tokenizer_config.json = metadata/settings
   (which class, which token is eos/pad/bos, model_max_length, flags = how
   transformers USES it). The big file is the tokenizer; the small file
   configures it.

2. `tokenizer.json` is described as holding the whole tokenizer "pipeline." Given
   the standard pipeline — normalize -> pre-tokenize -> model -> post-process ->
   decode — what sections would you expect inside the file, and which pipeline
   stage does each map to? (Qwen's actually has: normalizer, pre_tokenizer,
   model, post_processor, decoder, added_tokens.)

   [Taught — author didn't know; grounded on the real tokenizer.json sections.]

   Mapping (section -> stage -> what it does in Qwen):
   - normalizer -> NORMALIZE: {type: NFC} Unicode normalization (gap 1).
   - pre_tokenizer -> PRE-TOKENIZE: a Sequence of Split (the regex) + ByteLevel.
     The Split regex is the Q22 category pre-split — its pattern has \p{L}+ (letter
     runs), separate punctuation/whitespace groups, and \p{N} which matches a
     SINGLE digit (that's Q14 digit-splitting, right in the regex). ByteLevel then
     maps raw bytes to the visible Ġ-alphabet (Q12 space-as-Ġ).
   - model -> MODEL: type BPE, apply vocab + merges within each chunk (Q1 trace).
   - post_processor -> POST-PROCESS: assemble final sequence / add structural
     special tokens / fix offsets. Qwen's is minimal (ByteLevel bookkeeping, no
     BOS since bos is null); BERT's is where [CLS]...[SEP] get wrapped (Q6/Q15).
   - decoder -> DECODE (reverse, IDs -> text): ByteLevel decoder undoes the byte
     mapping (Ġ -> space, bytes -> UTF-8, with the Q21 replacement-char handling).
   - added_tokens -> not a stage but the reserved special-token registry (the 26
     tokens), matched BEFORE BPE and bypassing merges (Q15).

   Two things the grounded file shows: ByteLevel appears THREE times (pre_tokenizer
   in, post_processor, decoder out) — the byte<->Ġ mapping applied on the way in
   and undone on the way out, the whole round-trip. And the pre_tokenizer regex is
   Q22 + Q14 made literal.

3. Older GPT-2-style tokenizers don't save a single `tokenizer.json` — they save
   `vocab.json` + `merges.txt` (plus a couple of small JSONs). What does each of
   those two files contain, and what's the relationship between them and the
   `model` section inside a modern `tokenizer.json`?

   Direct consequence of Q1: the legacy format SPLITS the two big data structures
   into two separate files. `vocab.json` = the token-string -> ID map (=
   model.vocab). `merges.txt` = the ordered BPE merge rules, one per line in
   learned order (= model.merges). The modern tokenizer.json UNIFIES those two
   into its `model` section AND adds the declarative pipeline objects (normalizer,
   pre_tokenizer, post_processor, decoder) that the legacy slow format kept
   implicit in Python code. So: legacy = vocab.json + merges.txt + small JSONs
   (special_tokens_map, added_tokens) + tokenizer_config; modern fast = all of it
   in one tokenizer.json + tokenizer_config. See files_vs_code_notes.md for the
   files-vs-library-code split this question opened up.

4. `tokenizer_config.json` is ~700 bytes; `tokenizer.json` is ~11 MB. What's the
   division of labor between them — what kind of information lives in the tiny
   config vs the huge file, and why does that split make sense?

   [Answered in Q1 + files_vs_code_notes.md.] tokenizer.json = the tokenizer's
   DATA + declarative pipeline spec (vocab 151,643 + merges 151,387 ~9.2 MB, plus
   normalizer/pre_tokenizer/post_processor/decoder specs) = WHAT the tokenizer is.
   tokenizer_config.json = tiny metadata/settings = HOW transformers uses it
   (tokenizer_class to instantiate, which token is eos/pad/bos, model_max_length,
   flags like add_prefix_space/errors/split_special_tokens). The split makes sense
   because the 11 MB is learned vocab/merges that rarely changes, while the config
   is a handful of human-set knobs; and the config's tokenizer_class is what tells
   the library which code to load to interpret the big file.

## Set B — formats and the fast/slow split

5. SentencePiece models (Llama 1/2, T5) ship a single binary `tokenizer.model`
   (a protobuf) instead of JSON. What's the trade-off of a binary-protobuf format
   versus HF's readable JSON for a tokenizer file — think about loading speed,
   inspectability, and debugging.

   [Taught — author didn't know.] Binary protobuf buys compactness + fast parse;
   JSON buys inspectability (open + read it), greppable/diffable (git diff two
   tokenizers), debuggable/editable (fix by hand), and portability (any impl can
   load it). Key insight: a tokenizer loads ONCE at startup, in ms, dwarfed by
   loading the GB model weights — so the binary's speed/compactness advantage is
   basically irrelevant in practice; developer experience (inspect/diff/debug/
   edit/convert) is what matters. That's why the ecosystem converged on JSON
   (HF tokenizer.json). The binary's concrete downsides: opaque (need
   sentencepiece_model.proto + a protobuf parser to read it), vocab-expansion
   pain (parse -> modify -> re-serialize), interop friction (.model -> tokenizer
   .json is a known task). SentencePiece is binary because it's a Google C++/
   protobuf-native library that predates HF's JSON format; Llama 1/2 + T5 built on
   it. Trade-off resolves decisively toward JSON for a tokenizer.

6. tiktoken (GPT-3.5/4) uses `.tiktoken` files storing "mergeable ranks" instead
   of `vocab.json` + `merges.txt`. What do you think "mergeable ranks" means, and
   why might OpenAI use a different format from Hugging Face?

   [Taught — author didn't know; ties to the Q1 "hugging" trace.] Mergeable ranks
   = a single map byte-sequence -> rank (an integer), where the rank does DOUBLE
   DUTY: it's both the token's ID and its merge priority. One map replaces both
   files: it IS the vocab (every token is a key, its rank is its ID), and it
   encodes the merges implicitly — to encode, at each step find the adjacent pair
   (a,b) whose MERGED result a+b is in the map with the smallest rank, merge it,
   repeat until no adjacent merge is in the map. Same greedy algorithm as the
   trace, but you look up the merged result a+b (not the pair) and use its rank as
   priority; no explicit merges list needed, because in BPE a token's ID and its
   merge priority are two views of the SAME learned order. Why OpenAI differs from
   HF: tiktoken is a minimal, fast, byte-level-only BPE (regex split + byte BPE,
   no normalizer, no multi-algorithm pipeline), so it doesn't need tokenizer.json's
   full declarative spec — one compact ranks file suffices. Design-philosophy
   split: tiktoken = minimal BPE engine for speed; HF tokenizers = general
   pipeline framework. (.tiktoken is convertible to vocab.json/merges.txt.)

7. What is the "fast vs slow" tokenizer distinction, and how does it show up in
   the SAVED FILES (which files each produces) and in what the tokenizer can DO
   (a capability the fast one has that the slow one doesn't)?

   _(answer here)_

## Set C — fields, consistency, and gotchas

8. The `added_tokens` section lists 26 special tokens, each with flags like
   `special`, `single_word`, `lstrip`, `rstrip`, `normalized`. Why does a special
   token need these flags — pick one (e.g. `lstrip`/`rstrip` or `normalized`) and
   say what would go wrong without it.

   [Reframed to a Qwen3-specific dig: WHY does the 0.6B text model carry all 26.]
   Qwen3-0.6B's 26 special tokens, grouped: chat (<|endoftext|>, <|im_start|>,
   <|im_end|>); thinking (<think>,</think>); tools (<tool_call>/</tool_call>,
   <tool_response>/</tool_response>); vision (<|vision_start|>/end, <|image_pad|>,
   <|video_pad|>, <|vision_pad|>); visual grounding (<|object_ref_*|>, <|box_*|>,
   <|quad_*|>); code fill-in-middle (<|fim_prefix|>, <|fim_middle|>, <|fim_suffix|>,
   <|fim_pad|>); repo-level code (<|repo_name|>, <|file_sep|>).

   Why a 0.6B TEXT model carries vision/video/tool/FIM tokens it can't use
   (author): it's a SHARED family tokenizer (Q16). The same vocab serves Qwen3-VL
   (vision/box tokens), Qwen3-Coder (FIM/repo tokens), and the tool/thinking
   variants.
   What it buys: (1) family-wide consistency — a token id means the same thing
   across every Qwen3 model, which is EXACTLY what enables speculative decoding
   (Q19: draft + target must share the vocab) — a 0.6B can draft for a bigger
   Qwen3 target because they share this tokenizer; (2) train-once + merging +
   distillation; (3) a clean upgrade path (extend to vision/tools/code without
   changing the tokenizer, slots already reserved).
   What it costs: a few reserved embedding rows the 0.6B text model never trains —
   the undertrained/glitch-token situation from Q16 and the ~267 unused embedding
   rows from exercise 5 — but negligible (26 of 151,936; the vocab is dominated by
   the ~151k regular BPE tokens, Q17). Punchline for the inference book: the
   shared tokenizer is what makes cross-Qwen speculative decoding possible.

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
