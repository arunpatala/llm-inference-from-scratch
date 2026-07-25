# Pipeline stages + tensors + chat-template extensions (coverage-gap notes)

Chapter notes for coverage gaps 1-6 (see COVERAGE_GAPS.md) — the near-universal
topics standard curricula teach that our interview didn't name. Group A (1-4)
completes the mechanical "text -> exact tensors the model eats" picture; Group B
(5-6) completes the chat-template picture for modern/agentic inference. Grounded
on Qwen3-0.6B where marked; chat-template items still to interview/ground.

The standard pipeline framing to adopt as a spine:
normalize -> pre-tokenize -> BPE (model) -> post-process -> decode.
We cover BPE + decode; normalization and post-processing were never named.

## Group A — pipeline stages + tensors

### Gap 1 — Normalization (the stage BEFORE BPE)
What: before BPE runs, tokenizers apply Unicode normalization (NFC/NFKC), and
older ones also lowercase / strip accents. Why NFC exists: the same visible
character can have different byte sequences — "é" can be ONE code point (U+00E9)
or "e" + a combining accent (U+0301). Identical on screen, different bytes.
Why it matters: those two forms would tokenize differently unless normalized, so
normalization is what makes "same-looking text" produce the same tokens. But
NFKC is lossy/irreversible (ligature fi -> "fi", full-width -> ASCII), which
breaks two things you reasoned about: byte-floor reversibility (Q11) and exact
prefix-cache match (Q28 — two visually-identical prompts in different Unicode
forms are a silent cache miss unless normalized first).
Grounded on Qwen3-0.6B: composed "é" (U+00E9) and decomposed "e"+U+0301 BOTH
encode to token [963] — so Qwen DOES apply an NFC-style normalization (corrects
an earlier guess that modern byte-level BPE skips normalization; it doesn't).
Without it the two byte sequences (2 bytes vs 3) would tokenize differently.

### Gap 2 — Offset mapping (token <-> character alignment)
What: for each token, which characters of the original string it covers. Fast
tokenizers return an offset table (token i -> chars [start,end)) and word_ids.
Why it matters: many tasks map BACK from a token to a text span — NER/extraction,
guided/constrained (grammar/JSON) decoding, logprob/attention attribution to a
region, streaming highlight. You cannot guess the mapping because subwords don't
align to words, the leading space is inside the token (Q12), and tokenization is
non-compositional (Q27) — the tokenizer must hand you the alignment.
Grounded on Qwen3-0.6B ("Tokenization rocks", offsets):
  'Token'   -> chars (0,5)
  'ization' -> chars (5,12)
  'Ġrocks'  -> chars (12,18)   <- span includes the space at char 12 (the Ġ)
That last row shows why you need the map: 'Ġrocks' covers " rocks" (char 12 is the
space), which you'd never infer from the token string alone.

### Gap 3 — Truncation / context-window / sliding-window
What: when input exceeds the context window you truncate (cut to max_length) or
chunk (split a long doc into overlapping windows via a stride, keeping overflow).
Why it matters: the context window is a hard limit in TOKENS (Q20). Fitting
prompts into it is pure inference work, and it's the flip side of padding — Q18
was "pad short inputs up", this is "cut long inputs down", with the same
left-vs-right question (drop the old system prompt or the recent question?). Long
docs (RAG) chunk with overlap so meaning isn't severed at a boundary. Fertility
(Q24) makes the same text a different token count per model, so where truncation
bites differs across tokenizers.
[to ground: show max_length truncation + stride/overflow on Qwen.]

### Gap 4 — The attention mask (the tensor we never named)
What: the tokenizer returns not just input_ids but an attention_mask — 0/1 marking
real vs padding tokens.
Why it matters: when you batch (Q18) you pad to equal length, and the mask is what
tells the model to IGNORE the pad positions; without it the model attends to
garbage pads and produces wrong output. You derived left-padding needing the pads
masked out in Q18 — the attention mask IS that mechanism. So "tokenize -> tensor
-> model" really passes {input_ids, attention_mask} (+ position_ids). Packing/
varlen (the Q18 catch) replaces the mask with cu_seqlens.
Grounded on Qwen3-0.6B (batch ["hi", "a longer sentence here"], padding):
  ids=[6023, 151643, 151643, 151643]  mask=[1,0,0,0]   <- 3 pad tokens masked
  ids=[64, 5021, 11652, 1588]         mask=[1,1,1,1]
  pad_token='<|endoftext|>' (151643); padding_side=right (set left for generation
  per Q18).

## Group B — chat-template extensions for modern inference

### Gap 5 — Prefill / assistant continuation (continue_final_message)
What: instead of opening a fresh assistant turn (generation prompt, Q5), you hand
the model the START of its reply and make it continue. Mechanically
continue_final_message keeps the final assistant message OPEN (does not close it
with <|im_end|>) so the model resumes inside it.
Why it matters: a core control technique — prefill "```json\n{" to force valid
JSON, prefill "<think>" to force reasoning, generally put words in the model's
mouth. Different from the generation prompt (which opens a NEW empty turn) and
distinct from token healing (Q23), though both continue at a boundary.
[to interview/ground: render continue_final_message vs add_generation_prompt on
Qwen and show the trailing-token difference.]

### Gap 6 — Tool / function-calling rendering in chat templates
What: the chat template (Jinja2) serializes tool DEFINITIONS (as JSON schema)
into the prompt, the model emits structured tool_calls, and results return as a
tool-role message — all rendered into the token sequence, often delimited by
special tokens like <tool_call>/<tool_response> (04_out_of_scope flagged these).
Why it matters: the backbone of agentic inference, and fundamentally a
tokenization/chat-template concern — tool schemas and calls are TEXT serialized
into tokens by the template. So all the chat-template lessons apply: a wrong/stale
template silently breaks tool-calling (the Q6 / chat-template-Q4 silent-
degradation failure), and the tool-boundary special tokens are reserved entries
(Q15).
[to interview/ground: render a tools=[...] chat template on a tool-capable Qwen
and show the serialized schema + tool_call tokens.]

## Through-line
Gaps 1-4 complete the mechanical picture (text -> the exact tensors the model
eats; normalization and the attention mask are the two most load-bearing
omissions). Gaps 5-6 complete the chat-template picture for how models are used
today (structured prefill + agents). None are exotic; each hooks onto something
already reasoned through: reversibility (Q11), non-compositionality (Q27),
padding (Q18), token budget (Q20), silent template degradation (Q6).
