# Chat templates — questions

Answer inline under each question, same as the other question files. Skip or
write "not sure" where there's no real take yet.

1. Chat templates are Jinja2 templates, not just format strings. Why would a
   project choose an actual templating language (loops, conditionals) over a
   simple f-string/format-string template for this job?

   Because the formatting is genuinely conditional and variable-length, not
   simple substitution. The template has to loop over an unknown number of
   messages, emit a system block only if a system message is present (or inject
   a default one), special-case tool-call and tool-response turns, handle the
   assistant `<think>` block, and add the generation prompt only when asked. A
   format string can only substitute fixed slots; it can't loop or branch.
   Jinja2 (loops, conditionals, filters) expresses that logic, which is why the
   template ships as code rather than a fixed string.

2. Walk through what `apply_chat_template` actually does mechanically: the
   input is a list of `{role, content}` dictionaries. What comes out the
   other end, and at what point does tokenization happen relative to that?

   _(answer here)_

3. `add_generation_prompt` and `continue_final_message` sound like they could
   overlap. What's the actual difference in the tokens each one produces,
   and why does the library raise an error if you try to set both at once?

   `add_generation_prompt=True` appends a brand-new *open* assistant turn —
   `<|im_start|>assistant\n` with nothing after it (verified in
   `demos/chat_template_demo.py`). Use it when the last message is a user turn
   and you want a fresh assistant reply. `continue_final_message=True` does the
   opposite bookkeeping: it takes an existing final message (already the
   assistant's) and leaves it open by *dropping* the trailing `<|im_end|>`, so
   the model resumes inside that same turn and finishes it — the prefill /
   put-words-in-the-assistant's-mouth workflow. So one *opens a new* assistant
   turn and the other *keeps the existing final* turn open; asking for both at
   once is contradictory (start-new vs continue-existing), which is why the
   library raises rather than guessing. Token-level: the first adds the
   `<|im_start|>assistant\n` opener; the second removes the final `<|im_end|>`.

4. Using the wrong chat template (or none at all) doesn't error out, it just
   silently degrades quality. A real measured example: swapping Gemma-2-2B's
   own template for Llama-3's format dropped its AlpacaEval score from 41.2
   to 28.8. Why would a "close enough" template still hurt this much,
   instead of just being slightly worse?

   The model was instruction-tuned on exactly one wrapper format, and the
   special-token positions plus the surrounding structure are part of what it
   learned to condition on. `<|im_start|>assistant\n` is the literal learned
   cue for "be the aligned, instruction-following assistant now." Swap in a
   different format and the words are identical but the *structure* the model
   uses to locate the instruction, the turn boundaries, and which mode it's in
   is wrong — it's now off-distribution, and the alignment / instruction-
   following behaviors keyed to that structure fall back toward raw base-model
   behavior. It gets worse when the foreign format uses tokens that aren't this
   model's real control tokens (e.g. Llama's `[INST]`): those aren't in Qwen's
   vocab as special tokens, so they tokenize as ordinary text (several BPE
   pieces), and the model doesn't even receive a clean boundary signal. That's
   why "close enough" isn't close — the model is finely tuned to the exact
   template. (Measured drop cited above from the HF chat-templates blog.)

5. Chat templates are executable code (Jinja2), evaluated fresh on every
   single request, not passive data. What does that imply as an attack
   surface that a static prompt-formatting string wouldn't have?

   The template is Jinja2 code loaded from the model repo's
   `tokenizer_config.json` and run fresh, inside the serving process, on every
   request. A static format string executes nothing; this executes. So pulling
   a model from an untrusted source means also pulling and running *its* code:
   a tampered template is effectively code execution in your inference server,
   per request. Jinja2 has a sandbox, but sandbox escapes have been found, and
   even within bounds a crafted template can misbehave — leak or rewrite
   content, inject tokens, silently break prefix-cache reuse, or DoS the server
   with an expensive render. The lesson is that the template is code you are
   trusting, not passive data. (See ChatBug and the executable-per-request
   backdoor vector in the sources below.)

6. We pulled Qwen3-0.6B's actual chat template earlier and saw it
   special-cases system messages, tool calls, and `<think>` tags. Why does
   the "assistant" role need special handling at all? Why can't
   system/user/assistant all be formatted identically?

   _(answer here)_

7. If the last message in a conversation already has the "assistant" role,
   `apply_chat_template` silently switches to continue-generation mode
   instead of new-turn mode by default. What real workflow does this
   default exist to support?

   _(answer here)_

8. The template renders a list of structured messages into one flat string
   of text. Does the chat template itself ever produce token IDs directly,
   or is it strictly a text-formatting step that happens entirely before
   tokenization?

   _(answer here)_

9. Question 28 (in `../01_questions.md`) already covered that prefix caching
   needs an exact token-level match. Chat templates commonly interpolate
   per-request content (a system prompt, today's date, a tool list) directly
   into the rendered string. Concretely, what part of a typical chat
   template is most likely to accidentally break prefix-cache reuse across
   requests?

   _(answer here)_

10. Chat templates live in `tokenizer_config.json` alongside the tokenizer
    itself, and different checkpoints of "the same" model family can ship
    different templates. What's the actual risk of serving a chat model
    with a mismatched or stale template file, distinct from a tokenizer/
    vocab mismatch (the version-mismatch item in `../04_out_of_scope.md`)?

    _(answer here)_

## Sources (for citation when the chapter is written)

- [Hugging Face: Chat Templates — An End to the Silent Performance Killer](https://huggingface.co/blog/chat-templates) — Q1, Q4
- [Hugging Face docs: Templates for Chat Models](https://huggingface.co/docs/transformers/chat_templating) — Q2, Q3, Q7
- [Hugging Face docs: Writing a chat template](https://huggingface.co/docs/transformers/chat_templating_writing) — Q1, Q8
- [ChatBug: A Common Vulnerability of Aligned LLMs Induced by Chat Templates](https://www.researchgate.net/publication/381580130_ChatBug_A_Common_Vulnerability_of_Aligned_LLMs_Induced_by_Chat_Templates) — Q5
- Chat templates as an inference-time backdoor vector (executable Jinja2 evaluated per-request) — Q5
