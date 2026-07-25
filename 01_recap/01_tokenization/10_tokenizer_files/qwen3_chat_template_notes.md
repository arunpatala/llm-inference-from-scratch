# Qwen3's chat template (chapter note)

Grounded on the real chat_template.jinja shipped with Qwen3-0.6B (4,168 bytes,
saved as a separate file). It's a real Jinja PROGRAM, not a format string (Q10),
and it does three jobs.

## What the template does

1. Tool serialization (gap 6): if `tools` are passed, it renders the function
   signatures as JSON into a system message inside <tools>...</tools>, and
   instructs the model to emit <tool_call>{"name":...,"arguments":...}</tool_call>;
   tool-role responses are wrapped in <tool_response>...</tool_response>. So
   tool-calling is just chat-template text serialization.
2. Thinking handling: it splits each assistant message on '</think>' to separate
   reasoning_content from the answer content, and manages <think> across turns
   (below). At the end, enable_thinking=false injects an empty <think>\n\n</think>.
3. Standard chat structure: <|im_start|>{role}\n{content}<|im_end|>\n per turn,
   plus the generation prompt.

## The multi-turn thinking behavior (author's Q7 war story, in code)

The template STRIPS prior-turn <think> reasoning from history — for previous
assistant turns it keeps only the answer (after </think>), dropping the
reasoning. Only the current turn carries thinking (or gets the injected empty
block under enable_thinking=false). Verified earlier via apply_chat_template.

Why it strips (three reasons, same direction):
- Matches the training convention: Qwen3 trained with thinking on the CURRENT
  turn only, stripped from history (the author's Q7 statement). The template
  encodes that so inference history matches training.
- Context/KV economy: reasoning can be long; carrying it across turns bloats the
  sequence (Q30) for content only needed to produce one answer.
- Keeps the model on-distribution: history WITH prior-turn thinking is a
  distribution the model never saw.

Where the author's Q7 bug came from, relative to the template: NOT the template —
it does the right thing. The bug came from a serving/finetune path that didn't
route through this template (or didn't replicate it): feeding the model's full
generated output, <think> included, back into history. So inference history had
prior-turn thinking that training never had -> off-distribution -> silent quality
degradation.

## The lesson

The chat template IS the executable encoding of the training convention. If a
serving path builds the prompt itself instead of using apply_chat_template — or a
finetune's data doesn't match it — you silently drift off-distribution, no error
(the Q6 / chat-template-Q4 silent-degradation theme). The author's scar is the
canonical example: the template was protecting against exactly the bug, and the
bug was stepping around the template.

## Reasoning stripping vs prefix caching (a real cost of reasoning models)

Dropping prior-turn <think> collides with prefix caching (Q28: reuse needs an
exact, stable token prefix). Walk a 2-turn conversation:
- Turn 1 generation: prompt [sys][user Q1][gen-prompt]; model generates
  <think>r1</think>a1<|im_end|>. So the KV cache built during turn 1 PHYSICALLY
  contains the reasoning r1 (it was in the sequence while generating).
- Turn 2: the template renders turn 1's assistant message as a1 WITHOUT r1
  (stripped). So turn-2's prompt is [sys][user Q1][assistant a1 no-think][user
  Q2][gen-prompt], but turn-1's cached KV was [sys][user Q1][gen-prompt]
  [<think>r1</think>a1]. They DIVERGE at the assistant turn — reasoning present in
  the cache, absent from the new prompt.

Consequence: prefix caching can reuse KV only up to the divergence ([sys][user
Q1]); the assistant turn's KV cannot be reused, because its token content changed
when the reasoning was stripped. Everything from the assistant turn onward is
recomputed. So stripping reasoning BREAKS cross-turn prefix-cache reuse of the
assistant turns — an inference cost reasoning models pay and non-reasoning models
don't.

Silver lining / why it's still right: the stripped history is SHORTER (the
possibly-huge r1 is gone), so the recompute is over fewer tokens than if you kept
the reasoning; and you avoid carrying long reasoning across every turn (Q30 KV
bloat). You trade cache reuse for a smaller prefix.

The unavoidable tension:
- Strip reasoning (Qwen) -> correct distribution + smaller context (Q7), but
  prefix cache can't reuse the assistant turns.
- Keep reasoning -> prefix cache reuses the full turn-1 KV, but context bloat +
  off-distribution history (the Q7 bug).
You can't have both. Qwen chooses correctness over cache reuse. That's why
reasoning models are genuinely harder to serve efficiently in multi-turn: the
very stripping that keeps them correct is what defeats prefix caching. [Open
serving angle: whether an engine caches the stripped form, or keeps reasoning KV
around for reuse.]
