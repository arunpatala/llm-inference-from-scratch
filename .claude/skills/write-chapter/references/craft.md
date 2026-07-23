# Craft: how to actually write a chapter

`style_rules.md` is the list of smells to cut. This file is the other half —
what to do instead. A chapter that passes the linter with zero hits can still
be boring, muddled, or unclear. This file is about that.

Sources: [Diátaxis](https://diataxis.fr/) (the documentation framework behind
Stripe's, Django's, and Python's own docs), [Google's developer documentation
style guide](https://developers.google.com/style), the
[worked-example effect](https://en.wikipedia.org/wiki/Worked-example_effect)
from cognitive load research, and Rafael Vieira's ["100 things I learned
writing my first technical book"](https://dev.to/viebel/100-things-i-learned-writing-my-first-technical-book-2np3).

## 1. Decide the chapter's mode before writing a word

Diátaxis's core claim: documentation fails when it mixes purposes inside one
piece of writing. Four modes, four different jobs:

- **Tutorial** — the reader does something, under guidance, to build a skill.
  This is what a module's `exercise.py` walkthrough is. Write it as an
  experience you're leading someone through, not a description of one.
- **Explanation** — the reader comes to understand a concept. This is the
  recap chapter and the theory portion of each module. Write it to build a
  mental model, not to enumerate facts.
- **Reference** — the reader looks something up mid-task and leaves. Glossary
  entries, config options, API parameters. Terse, scannable, no narrative.
- **How-to** — the reader already knows the subject and wants to accomplish
  one specific thing. The capstone's deployment steps are this.

Before drafting, name which mode the chapter is in. A chapter that tries to
explain a concept *and* serve as a reference *and* narrate a tutorial at the
same time ends up satisfying none of the three — that muddiness is often what
actually reads as generic, more than any word choice does.

## 2. Concrete before abstract, always

Never open with a general statement. Open with a specific number, a specific
failure, or a specific question — the thing from the interview answers. The
general principle comes *after* the reader has seen one real instance of it,
not before.

Bad order: "KV caching is an important optimization that avoids redundant
computation." Good order: "I profiled naive decoding and watched throughput
fall from 170 tok/s to 78 tok/s once the context passed 500 tokens. Here's
why, and here's the one-line fix." The reader earns the general claim by
seeing the specific evidence first — don't hand them the conclusion before
the evidence exists for them.

This is also just recognizing what the interview questions were for: your
real number, your real confusion, your real mistake *is* the concrete
opening. Don't discard it and write a generic one instead.

## 3. Worked example before the reader's own attempt

Research on the worked-example effect: when someone has little prior
experience with a topic, showing them one fully-solved instance first — before
asking them to solve the general case — produces better learning than having
them attempt the general case cold. It works because solving a novel problem
and understanding a new concept compete for the same limited attention;
separating them lets the reader spend that attention on understanding.

Applied here: every exercise TODO gets a worked numeric example immediately
before it — a tiny concrete case (small sequence length, small tensor shapes,
actual numbers) traced by hand, the way tiny-llm traces a 3-token then a
4-token attention matrix before asking the reader to implement KV caching
generally. Show the trace, then ask for the general version.

## 4. Motivate before mechanism

Establish the cost of *not* knowing this before explaining what it is. If a
reader can't tell you why they should keep reading after the first paragraph,
rewrite the first paragraph. This doesn't mean hype — it means naming the
actual problem this chapter's technique solves, concretely, before describing
the technique.

## 5. Voice

- Second person, direct address — "you" run the benchmark, not "one" or "the
  user."
- Active voice, present tense: "the scheduler evicts finished sequences," not
  "finished sequences are evicted by the scheduler" or "will be evicted."
- One idea per sentence. If a sentence needs a semicolon to hold two claims,
  it's probably two sentences.
- Write like explaining to a sharp colleague at your level, not lecturing
  down and not performing expertise. Google's style guide calls this
  "a knowledgeable friend who understands what you're trying to do" — casual
  and direct, not pedantic.
- Let the reader feel smart. Explain the *why* behind a design choice, not
  the obvious mechanics they can already read from the code.

## 6. Structural discipline per chapter

- One objective per chapter. If it needs "and" in the one-sentence summary of
  what it teaches, split it.
- Difficulty ramps within the chapter — start with the easiest real instance
  of the idea, then complicate it, not the reverse.
- Every code example uses real values, not `foo`/`bar`/`<placeholder>` —
  Stripe's docs are the standard reference for this. Real prompts, real
  tensor shapes, real numbers from an actual run.
- Every code snippet shown in prose must actually run — verified against this
  project's own tests, not just "should work."
- A diagram or worked-out table earns its place only if it does something
  prose can't (an actual attention-matrix trace, an actual memory-layout
  picture) — not decoration.

## 7. Revise in this order

1. Write the content and the worked example first.
2. Write the motivating opening last, once you know what the chapter actually
   contains and can point at its real payoff honestly.
3. Read it once as if hearing it out loud from someone else. Anything that
   sounds like nobody would actually say it gets rewritten.
4. Run `scripts/style_lint.py` — that catches smells, not muddiness. Muddiness
   is caught by step 3.

## Changelog

Techniques the user finds worth adding go here, dated, same as
`style_rules.md`'s changelog.

- (initial version — seeded from Diátaxis, Google's developer style guide,
  worked-example-effect research, and "100 things I learned writing my first
  technical book")
