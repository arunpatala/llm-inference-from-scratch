---
name: chapter-versions
description: Use when turning a topic's accumulated notes/interview/research into an actual deliverable, and you need to choose and produce a specific VERSION — a blog article, a training-focused writeup, an inference-focused writeup, a lightweight subsection, a complete chapter, or a complete chapter with exercises. Owns the "which format, which audience, which depth, pull from which notes" decision; hands the actual prose drafting to write-chapter and the raw material to socratic-interview.
---

# Producing chapter versions from the source notes

A topic in this book accumulates a lot of source material — a Socratic interview
(the author's voice), grounded demos, notes, a research/future-directions
appendix, a history section, a coverage-gap analysis. That is the *single source
of truth*. This skill is about rendering that one source into a specific
**deliverable version**, each with a different audience, depth, and format.

This skill picks the version and assembles it. It does NOT replace:
- `socratic-interview` — produces the raw material (author's answers, grounded).
- `write-chapter` — owns the drafting workflow + the anti-slop style gate. Every
  prose version still goes through it.

## Step 1 — pick the version

Six versions, by audience and purpose. Pick one before writing a word.

| Version | Audience | Diátaxis mode | Length (rough) | Exercises? |
|---|---|---|---|---|
| **Blog article** | curious/lay reader | Explanation (narrative) | 800-1800 words | no |
| **Training-focused** | someone building a tokenizer/model | Explanation + How-to | 1500-3000 | optional |
| **Inference-focused** | serving/inference engineer (this book's core) | Explanation + Reference | 2000-4000 | optional |
| **Lightweight subsection** | reader of a bigger chapter | Explanation (terse) | 400-1200 | no |
| **Complete chapter** | the book's target reader | Explanation-led | full | no |
| **Complete chapter + exercises** | build-it-yourself reader (Core-Module tier) | Explanation + Tutorial | full + code | yes |

Rules of thumb for choosing: a *lay* audience needs background and definitions and
one clear idea (blog); a *technical/expert* audience can handle depth **only when
it's presented with clarity** — logical, scannable, immediate value. Match the
version to who will actually read it, not to how much material you have.

## Step 2 — pull from the right source notes

Each version draws a different slice of the same source. Name the files first.

- **Blog**: the narrative spine + 1-2 grounded examples + the "why it matters"
  hook. (The readable history is a model blog: concept-before-use, one thesis.)
- **Training-focused**: the BPE-training material (train exercise, algorithms
  note, vocab-size), WordPiece/Unigram as alternatives. Inference specifics are
  background only.
- **Inference-focused**: the encode/decode mechanics (merge trace + efficiency),
  chat templates, special tokens, streaming detok, the KV/prefix/spec-decode
  interactions, the tokenizer files, fertility/cost. Training is background only.
- **Lightweight subsection**: the essentials only (what a token is, byte-level
  BPE, the pipeline, the one inference reason it matters), and cross-reference the
  full version for depth.
- **Complete chapter**: (almost) everything — the interview answers as prose, the
  demos, the pipeline, the history hook, a Future-Directions pointer.
- **+ exercises**: the above, plus the existing coding exercises (fill-in-the-
  blank + solution + tests) and the interview questions repurposed as reader
  check-questions.

## Step 3 — shared rules (all versions)

1. **Single source of truth.** Every claim, number, and code snippet traces to
   the existing grounded notes/demos/interview — never invent new facts for a
   version. Reuse the *verified* numbers verbatim (they were measured once).
2. **Preserve the author's voice.** The interview answers are the closest thing
   to a real-voice draft; carry the phrasing, rough analogies, and specific
   complaints through. Do not launder them into generic prose (write-chapter §5).
3. **One Diátaxis mode per piece.** Don't mix explanation + reference + tutorial
   in one version; muddiness reads as generic (write-chapter craft §1).
4. **Run the style gate.** Any prose version goes through `write-chapter`'s
   `scripts/style_lint.py` + the read-aloud check before it counts as done.
5. **Cross-reference, don't duplicate.** Shorter versions link to the fuller ones
   rather than repeating; the source notes stay the shared backing.
6. **Concrete before abstract, motivate before mechanism** (write-chapter craft
   §2, §4) — regardless of version.

## Step 4 — per-version specifics

- **Blog**: one central thesis; concrete hook in the first paragraph; define
  jargon inline or avoid it; cut anything that needs a footnote; end on the
  payoff, not a summary. No exhaustive coverage — depth is the enemy here.
- **Training-focused**: lead with *why train a tokenizer* (the OOV/vocab-fit
  problem), then the algorithm, then the choices (vocab size, algorithm family).
  A worked training trace beats prose.
- **Inference-focused**: lead with the serving cost/consequence; keep training to
  a paragraph of background; foreground the encode/decode/template/KV interactions
  an engineer actually hits. This is the default for THIS book.
- **Lightweight subsection**: ruthless — only what the surrounding chapter needs
  to stand, plus a "for the full treatment, see X" pointer.
- **Complete chapter**: follow write-chapter end-to-end (one mode/objective,
  worked example before the reader's attempt, hook written last). Ends on the
  Future-Directions bridge if the book wants forward motion.
- **+ exercises**: hybrid per the project plan — full narrated walkthrough for
  glue code, fill-in-the-blank + tests for the conceptually meaty function(s);
  each exercise gets a worked numeric example immediately before it
  (worked-example effect, write-chapter craft §3). Repurpose the interview
  questions as end-of-section check-questions.

## Step 5 — assemble, gate, cross-link

1. Draft the chosen version pulling the named source slice.
2. Add exercises/questions only if the version calls for them.
3. Run `scripts/style_lint.py`; read it aloud; fix muddiness.
4. Cross-link to the deeper version and back to the source notes.
5. For the versioned deliverables, keep them in the module (e.g. a `book/` or
   `blog/` output dir) so the source notes and the rendered versions stay
   separate — notes are the source, versions are the product.

## Note
Multiple versions can coexist from one source (single-source publishing): a blog,
a subsection, and a full chapter can all render the same tokenization notes at
different depths. Keep the notes canonical; treat each version as a view.

## Changelog
- Initial version — created for the tokenization chapter, which accumulated a
  large single-source body (interview + demos + research + history + files) and
  needs to render into one or more deliverable formats. Seeded from Diátaxis
  (the four modes), audience-level guidance (lay/technical/expert), and the
  existing write-chapter + socratic-interview skills.
