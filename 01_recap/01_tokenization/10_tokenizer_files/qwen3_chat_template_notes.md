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
