# Style rules: signs of AI writing to avoid

Single source of truth for `write-chapter`'s prose check. `scripts/style_lint.py`
parses the `## Banned Words` list below directly — add to it and the linter
picks it up with no code change. Everything else here is judgment for a human
(or an LLM acting carefully) to apply, not mechanically checkable.

Sources: [Wikipedia: Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing),
the `blader/humanizer` Claude Skill (derived from the same guide),
["The Last Fingerprint" (arXiv 2603.27006)](https://arxiv.org/html/2603.27006v1)
on markdown-training's effect on LLM prose structure, and
[petergyang/no-ai-slop](https://github.com/petergyang/no-ai-slop).

## Banned Words

One per line. Format: `- phrase (say: alternative)`. The linter matches the
phrase case-insensitively as whole words; the parenthetical is for humans only.

- delve (say: dig into, look at, explore)
- leverage (say: use)
- utilize (say: use)
- facilitate (say: help, make easier)
- harness (say: use, put to work)
- foster (say: build, encourage)
- underscore (say: show, highlight)
- optimize (say: improve, tune) — unless literally describing an optimizer/optimization pass
- navigate (say: deal with, work through) — unless literally about navigation
- endeavor (say: try)
- crucial (say: important, matters)
- pivotal (say: a turning point, mattered a lot)
- transformative (say: changed things)
- groundbreaking (say: new, first of its kind)
- robust (say: solid, reliable) — unless a term of art in the immediate context
- comprehensive (say: thorough, covers everything)
- innovative (say: new, creative)
- seamless (say: smooth, no friction)
- invaluable (say: genuinely useful)
- multifaceted (say: has several parts/angles)
- nuanced (say: name the actual distinction instead)
- holistic (say: name what's actually being combined)
- elevate (say: improve, raise)
- streamline (say: simplify)
- unlock (say: enable, make possible)
- unleash (say: enable, release)
- empower (say: let, allow, give the ability to)
- tapestry (say: mix, combination)
- landscape (say: name the actual space/area/field)
- realm (say: area, field)
- testament (say: proof, sign, evidence)
- paradigm (say: approach, model, way of thinking)
- synergy (say: name the actual combined effect)
- boast (say: has) — the linter also catches "boasts," "boasting"
- showcase (say: shows) — the linter also catches "showcases," "showcasing"
- vibrant (say: name the actual quality)
- nestled (say: located, sits)
- breathtaking (say: name what's actually impressive)
- enduring (say: lasting, long-running)
- cutting-edge (say: new, recent, state of the art with a citation)
- game-changing (say: name the actual effect)
- moreover (say: also, and)
- furthermore (say: also, and)
- additionally (say: also)
- consequently (say: so, because of that)
- nevertheless (say: still, but)
- notably (say: name what's notable directly, or cut it)
- ultimately (say: in the end, or cut it)
- essentially (say: basically, or cut it and just say the thing)
- it's important to note that (say: cut it, state the thing directly)
- it's worth noting that (say: cut it, state the thing directly)
- in today's fast-paced (cut the whole clause)
- in today's digital age (cut the whole clause)
- in the realm of (say: in, for, when it comes to — sparingly)
- plays a vital role in (say: matters for, affects, helps with)
- serves as a testament to (say: shows, proves)
- despite facing challenges (say: name the actual difficulty)
- I hope this helps (cut — not applicable in chapter prose anyway)
- let's dive in (cut, or say: here's how, start directly)
- let's explore (cut, or say: here's what happens)
- at its core (say: cut it, or name the actual core thing)
- paradigm shift (say: cut it, or name the actual change)
- game changer (say: name the actual effect)
- this is huge (cut it, state the actual fact)
- this changes everything (cut it, state the actual fact)
- beacon (say: name the actual role)
- meticulous (say: careful, thorough)
- intricate (say: complex, detailed — or name the actual complexity)
- paramount (say: most important, critical — or name why)
- embark (say: start, begin)
- supercharge (say: speed up, boost — name the actual mechanism)
- ever-evolving (cut the whole clause, or name the actual change)
- at the end of the day (cut it, state the point)
- in the age of (cut the whole clause)
- the reality is (cut it, state the thing directly)
- the truth is (cut it, state the thing directly)
- in terms of (say: for, about — or cut)
- with regard to (say: about, for)
- going forward (cut it, or say: from now on, only if truly needed)
- in this article (cut it — the reader knows what they're reading)

**Empty adverbs.** Cut when they add nothing: just, literally, honestly,
simply, actually, truly, fundamentally, importantly, inherently, inevitably.
Keep only when one carries real emphasis, contrast, or uncertainty the
sentence would lose without it — a hedged "honestly, I'm not sure" is fine;
a reflexive "honestly" in front of a plain statement isn't.

## Banned Structural Patterns

**Negative parallelism.** "It's not just X, it's Y." "Not a mirror but a
portal." Banned regardless of how the two halves are worded — restate as one
direct claim instead.

**Rule of three.** Reflexively listing exactly three items — especially with
matching grammatical structure — to *sound* comprehensive rather than because
there really are three things. If there are two real points or five, say two
or five.

**Formulaic concession-conclusions.** "Despite these challenges, X continues
to..." Say what actually happened instead of reaching for the template.

**Copula avoidance.** Replacing plain "is/are/has" with "serves as," "boasts,"
"stands as." Use the plain verb.

**Vague attribution / weasel wording.** "Researchers say," "it is widely
believed," "observers note" — without naming who. Name the source or cut the
claim.

**Hedging stacks.** More than one hedge in a single claim ("may potentially,"
"could possibly suggest"). Pick one hedge or none.

**Chatbot formalities.** Sycophantic openers ("Great question!"),
knowledge-cutoff disclaimers, "as an AI..." — none of these belong in chapter
prose under any circumstance.

**Superficial -ing tacked-on analysis.** A fact followed by a vague "-ing"
clause that sounds like insight but adds none ("...contributing to broader
understanding of the field"). Cut it or replace with an actual, specific
consequence.

**Throat-clearing openers.** "Here's the thing," "Let me be clear," "I'll be
honest," "The uncomfortable truth is." Cut the wind-up and state the point.

**Faux-insight setups.** "This is the part most people skip," "What nobody
tells you," "The part everyone misses." These flatter the writer as the lone
expert holding secret knowledge. Cut the setup and let the claim stand on its
own: "The part everyone misses: distribution is the moat" becomes
"Distribution is the moat."

**Colon reveals.** A noun phrase, a colon, then a dramatic lowercase reveal:
"The best part: it learns." Different from the list-introducing colon banned
under Markdown Formatting Discipline below — this is a single manufactured
reveal, not a list. Rewrite as a plain sentence instead.

**Negative listing.** "Not a database. Not a cache. A queue." Just say what
it is.

**Dramatic fragmentation.** "X. And Y. And Z." or "That's it. That's the
whole thing." Use complete sentences.

**Rhetorical setups.** "What if I told you...", "Think about it:",
"Plot twist:", or a self-answered "Question? Answer." pair. Drop the
performance and make the point.

**Fake-profound kickers.** A final "deep" line that turns a plain point into
a cute metaphor or mic-drop sentence. Delete it rather than polishing it into
a better metaphor — end on the clearest concrete sentence already in the
draft instead.

**Summary-recap endings.** "In conclusion," "Overall," or a closing
paragraph that just restates what the chapter already said. The reader was
just there — end on the last concrete point or next action instead.

## Markdown Formatting Discipline

LLMs trained on markdown-heavy corpora default to "structured document" mode
even when the right output is flowing prose — this is a *structural* tell,
independent of word choice, and it's the one that matters most for an
all-markdown book. See ["The Last Fingerprint"](https://arxiv.org/html/2603.27006v1).

- **Headers only for genuine hierarchy.** Not one per paragraph. If a section
  is three sentences, it probably doesn't need its own header.
- **Don't bullet-ify an argument.** Bullets are for genuinely parallel,
  enumerable items (API parameters, a literal file list, options with equal
  weight) — not for a chain of reasoning that has a natural prose order.
  If removing the bullet and adding connecting words ("because," "so," "which
  means") reads fine, it should've been prose.
- **Bold is for real emphasis, not jargon-on-first-use.** Don't bold every
  technical term the first time it appears.
- **No reflexive colon-then-list.** Sometimes a sentence is just a sentence;
  it doesn't need to end in a colon that opens a list.
- **Cap em dash usage.** Prefer a period, a comma, or restructuring the
  sentence. In a short passage, use none. In a full chapter, one or two are
  fine when they clearly beat a comma or period — not as a default rhythm
  crutch. Remove clusters.
- **No heading-level skipping, no decorative horizontal rules before every
  heading, no emoji in headings or as section dividers.**

## Counterindicators — don't over-correct

Per the Wikipedia guide, none of these alone indicate AI-written text, and
flagging them in isolation just produces stilted, over-defensive prose. Signs
are meaningful when they **cluster**, not as single instances:

- Perfect grammar
- Formal vocabulary, when the context genuinely calls for it
- A single em dash in a whole chapter
- Consistent, correct formatting

Don't rewrite a paragraph just because it contains one word from the banned
list — check whether it's actually doing work in context or whether it's the
lazy default choice. Reserve real edits for actual clusters of tells.

## Changelog

New smells the user notices go here, dated, in addition to being added above.

- (initial version — seeded from Wikipedia's Signs of AI Writing guide, the
  `blader/humanizer` skill, and the markdown-training-artifacts paper)
- 2026-07-23 — merged in [petergyang/no-ai-slop](https://github.com/petergyang/no-ai-slop):
  new banned words/phrases (paradigm shift, game changer, beacon, meticulous,
  intricate, paramount, embark, supercharge, ever-evolving, and more filler
  phrases), an empty-adverbs rule, and new structural patterns (throat-clearing
  openers, faux-insight setups, colon reveals, negative listing, dramatic
  fragmentation, rhetorical setups, fake-profound kickers, summary-recap
  endings), plus a sharper numeric em-dash rule and an emoji-in-headings rule.
