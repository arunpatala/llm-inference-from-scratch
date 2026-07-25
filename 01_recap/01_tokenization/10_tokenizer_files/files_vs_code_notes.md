# What lives in the saved files vs in the library code (chapter note)

A conceptual split that makes the whole tokenizer-files picture click: the saved
files are DATA + a DECLARATIVE SPEC; the library is the ENGINE that interprets
them. The `type` fields in tokenizer.json are the seam — names in the file,
implementations in the code.

## In the SAVED FILES (model-specific — the "what")
- Learned data: the vocab (token->id) and the merges (ordered rules). The only
  truly learned, model-specific content.
- A declarative pipeline spec — names + parameters, NOT code:
  `normalizer: {type: "NFC"}` says "use NFC", it does not contain the NFC
  algorithm; likewise pre_tokenizer (regex pattern + ByteLevel), post_processor,
  decoder. These are keys/settings that point at code.
- Config/metadata: tokenizer_class, special-token roles (eos/pad/bos),
  model_max_length, flags (add_prefix_space, errors).
- The chat template: Jinja SOURCE text — the one bit of executable content, stored
  as data.

## In the LIBRARY CODE (model-agnostic — the "how")
- The algorithms: BPE merge-application loop, regex splitter, NFC normalizer,
  byte-level encoder/decoder, Jinja2 interpreter, offset tracking, the word cache,
  the Rust fast backend.
- The tokenizer classes (Qwen2Tokenizer, GPT2Tokenizer...) that assemble the
  pipeline from the spec, plus load/save logic.

## Loading = spec + data -> instantiate code
from_pretrained reads the declarative spec + data from the files, then the
library instantiates the matching code components with those parameters (config-
driven: {type:"NFC"} -> the library's NFC implementation). The file says WHICH
normalizer; the code IS the normalizer.

## Three consequences
- Need a matching library version: the code must understand the spec's `type`
  fields. A brand-new component type won't load in an old `tokenizers` (part of
  the version-mismatch failure mode).
- tokenizer.json is portable/self-describing: because the spec fully declares the
  pipeline, any implementation of the format (Rust, transformers.js) can load the
  same file and reproduce the tokenizer — no model-specific code needed.
- The chat template is the exception: code shipped IN the file, executed per
  request by the library's Jinja interpreter — the security surface
  (chat-template Q5). Everything else in the files is inert data the code
  interprets.

Summary: files = learned vocab/merges + declarative pipeline spec + config;
library = the algorithms that interpret them.
