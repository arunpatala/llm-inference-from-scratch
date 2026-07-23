---
name: socratic-interview
description: Use when running or resuming the interview step for a book chapter — asking the author a chapter's questions and re-interrogating any wrong, vague, or thin answer until it is correct and complete. This is the execution engine for write-chapter's "interview first" step; write-chapter still owns drafting and style.
---

# Socratic chapter interview

The chapter is built from the author's own answers (see the `write-chapter`
skill). This skill is *how* to run that interview well: not transcribing what
the author says, but interrogating each answer until it actually holds. A
shallow or half-wrong answer, smoothed over and recorded, produces exactly the
generic prose the book exists to avoid. Your job is to make the answer real
before it becomes a chapter.

Two non-negotiables:

1. **The author answers every question in their own voice.** Never write the
   core answer for them. Verification, grounding, clearly-marked additions, and
   questions they explicitly hand off are fine — the substance of an answer is
   theirs.
2. **Keep interrogating until *you* are satisfied**, not until they stop
   talking. "Satisfied" means the answer is technically correct, precise, and
   complete for the chapter's purpose.

## The loop, per question

1. **Ask in small batches** (2-3 questions), in the author's framing. Say
   plainly that rough phrasing, half-formed takes, and "not sure" are all
   welcome — those are the raw material, and "not sure" tells you what to teach.
2. **Read the answer critically before recording anything.** Find the genuine
   weak points — wrong claims, imprecise wording, a missing inference angle, a
   footnote-level non-answer, an unstated assumption. Do not manufacture
   problems; only push on real ones.
3. **Interrogate.** For each weak point, ask a pointed follow-up that makes the
   author reason to the sharper answer themselves. Name *why* you're asking
   (what's thin). One idea per probe when you can.
4. **Verify on the real system** (see below) whenever a claim is checkable.
5. **Record** the refined answer in the author's voice (minimum effective edit),
   plus grounded notes. Mark anything still open with an inline `[...]` note.
6. Repeat until satisfied, then move to the next batch.

## What "not satisfied yet" looks like

Push back when an answer is:

- **Wrong or imprecise** — e.g. attributing a property to the wrong layer
  ("the tokenizer captures word similarity" — no, embeddings do), or vague
  where a number/mechanism is available.
- **Footnote-level / not domain-specific** — e.g. "it's the first step so it
  matters." True of any book. Demand the answer specific to *this* book's
  concern (for an inference book: correctness-under-serving and cost/speed).
- **One-sided** — gives the cost of X but not of not-X; give the full trade-off.
- **Missing the payoff** — the author has the pieces but hasn't connected them.
  Ask the question that forces the connection (this is where the best chapter
  material comes from — e.g. deriving *why* the naive decode loop is wasteful
  leads straight to the next module's motivation).
- **Restated, not explained** — "the model needs numbers" restates the question;
  push for the mechanism ("what operation forces that?").

## Verify claims on the real system

The author's answers give voice and concrete detail; you still owe technical
correctness. When a claim is checkable, check it — do not assert from memory:

- Run it against the real model/tokenizer/code. Prefer a small script that
  prints real values. Respect this project's two-machine split: do all of this
  on the local Mac (CPU/MPS, no CUDA); it is plenty for tokenizer/correctness
  work. See `LOGS/*two_machine_split*`.
- **Predict-then-verify** is the strongest teaching move: have the author (or
  yourself) predict the number, then run it and compare. Surprises that survive
  a real run are gold; predictions that don't survive get corrected on the spot
  — including your own overclaims (say so plainly when the run corrects you).
- Turn anything worth reusing into a small `demos/` script and, if it deserves
  prose, a short note that passes `scripts/style_lint.py`.
- Every number that lands in the chapter must trace to a real run (say how to
  reproduce it) or a real citation — never a plausible-looking guess.

## What not to do

- **Don't fabricate experience.** If a question asks for a personal surprise or
  bug and the author genuinely has none, do not invent one — that fake
  authenticity is the exact smell to avoid. Dig once for a real one from their
  actual work; if there's none, record "no personal surprise here" honestly and
  source the chapter's example from a documented, cited case instead.
- **Don't write the author's core answer.** If they don't know a factual
  answer, you may teach it (grounded) — but mark it clearly as taught, not as
  their voice, and flag anything to verify before drafting.
- **Don't accept the first answer just because it's plausible.** Plausible and
  correct are different; that gap is the whole reason this skill exists.

## Recording conventions

- Answers go inline under each question in the question file, in the author's
  voice, lightly cleaned (fix typos/grammar and cut AI-tells; keep rough
  analogies and blunt phrasing — see `write-chapter` craft §5).
- Label provenance when it isn't the author's own take: `[Taught, not the
  author's experience — verify X]`, `[Assembled from the author's points]`,
  `[pending: confirm with a real run]`. This keeps honest track of whose voice
  each answer is in.
- Cross-reference exercises, demos, and other questions by path/number so the
  chapter, the interview, and the code all point at the same grounded evidence.

## When you're satisfied

The answer is correct, precise, complete for the chapter's purpose, grounded
where checkable, and in the author's voice (or honestly marked otherwise). Say
so, record it, and move on. Then hand off to `write-chapter` for drafting —
this skill fills the interview; that skill turns it into prose.

## Changelog

- Initial version — distilled from the tokenization-chapter interview session
  that established this loop (batch questions, interrogate weak points, verify
  on the real Qwen3 tokenizer on the Mac, predict-then-verify, mark provenance,
  never fabricate). See the memory note `llm-inf-interview-socratic`.
