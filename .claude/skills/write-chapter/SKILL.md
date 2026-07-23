---
name: write-chapter
description: Use whenever drafting, revising, or reviewing prose for this project's book (book/src/*.md, any module README, any Study Note chapter). Enforces an interview-first drafting workflow and screens the result against a growing list of LLM writing tells before the chapter counts as done.
---

# Writing chapters for this book

This project is a personal knowledge base that other people will also read. The
single biggest risk to that is not technical inaccuracy — it's prose that reads
as generic LLM output, which readers now recognize instantly and discount on
sight. This skill exists to prevent that, chapter by chapter.

Two reference files do the actual work, and both matter — one is not a
substitute for the other:

- `references/craft.md` — how to write: choosing the chapter's mode
  (tutorial/explanation/reference/how-to), leading with something concrete,
  worked examples before the reader's own attempt, motivating before
  explaining mechanism, voice. Read this **before** drafting.
- `references/style_rules.md` — what to avoid: banned words, banned
  structural patterns, markdown-formatting discipline. Check the draft
  against this **after**. `scripts/style_lint.py` parses its banned-word list
  directly, so editing that file updates both the human checklist and the
  automated check.

A chapter with zero linter hits can still be muddled or boring — that's what
`craft.md` is for. A chapter that's well-structured can still read as generic
LLM prose — that's what `style_rules.md` is for. Neither alone is sufficient.

## 1. Interview first — always, no exceptions

Never draft a chapter from general knowledge alone. Before writing a single
sentence, ask the user these questions, adapted in wording to the chapter's
specific topic:

1. What's the one thing that actually confused you before this clicked?
2. What's a real number — from a benchmark you ran, or will run on this
   project's hardware — that surprised you?
3. What's the mistake you made, or would obviously make, before you
   understood this?
4. Explain it in one sentence like you're telling a colleague over lunch.
   What do you actually say?
5. What's the mental picture or analogy you use for this, even if it's rough
   or not fully rigorous?
6. What do other write-ups of this topic get wrong, skip, or explain badly?

The chapter is written **from these answers** — quote the user's own phrasing
directly where it works, rather than smoothing it into neutral textbook prose.
If an answer is thin or generic, ask a follow-up. Do not paper over a thin
answer with generic knowledge; a thin answer is a sign the interview isn't
done yet, not a gap to fill from training data.

Technical correctness is still this skill's job — the user's answers give the
chapter its voice and its concrete detail, not its factual content. Verify
every technical claim independently even when it comes from the user.

## 2. Hard factual rule

Every number, benchmark result, or claim in a chapter must trace to one of:

- this project's own `test_*.py` / `benchmark.py` output (say so, and how to
  reproduce it), or
- a real citation with a real, checkable URL.

Never write "studies show," "research indicates," or similar without naming
the specific source. Never invent a benchmark number to illustrate a point —
if the real number isn't measured yet, say that plainly and mark it as
pending, don't fabricate a plausible-looking one.

## 3. Before calling a chapter done

1. Check it has one clear mode and one clear objective (`craft.md` §1, §6) —
   split it if not.
2. Run `python scripts/style_lint.py <path-to-chapter.md>` and resolve every
   flagged line — either fix it or leave a one-line comment in the PR/commit
   explaining why the flagged instance is a deliberate exception.
3. Read the chapter once, mentally, as if saying it out loud to a colleague.
   Any sentence that sounds like nobody would actually say it out loud gets
   rewritten. This catches muddiness the linter can't.
4. Check both reference files once more — they grow over time (see below),
   so a chapter written before an addition may need a second pass.

## 4. Both reference files grow — treat additions as permanent

The user will notice new AI-writing tells, or new craft techniques worth
following, over time and ask for them to be added. When that happens:

- Smells to avoid go in `references/style_rules.md` — banned words in the
  `## Banned Words` list so the linter picks them up automatically,
  structural/formatting rules in their own sections as prose with a short
  example.
- Positive techniques go in `references/craft.md`, in whichever numbered
  section fits, or a new one.

Log the addition in that file's changelog. Never remove an existing entry
without being asked to.
