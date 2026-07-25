# How we learned to chop up text

Accessible narrative version of the tokenization history, written for a reader
with no prior background. (The companion file `tokenization_history.md` is the
dense, dated reference with full citations.)

## The problem underneath everything

A language model does math on numbers, not letters. So before a model can read
the word "hello," something has to turn that text into numbers. The piece that
does this is the **tokenizer**: it chops text into units and hands each unit a
number (an ID). Those units are called **tokens**.

That sounds simple, but it hides a genuine tension that shaped forty years of
research. To turn words into numbers, you need a **list**: a dictionary that says
"this word is number 42, that word is number 43." The list has to be a *fixed,
finite size*, because the model keeps one slot of memory for every entry, and
memory isn't infinite. But language is the opposite of fixed and finite: people
invent new words, misspell old ones, write names, mix in other languages, and
type emoji. So you have a fixed list trying to cover an endless world. The history
of tokenization is the story of escaping that trap.

## First idea: one word, one number

The obvious approach, and the one everyone used for a long time, is to split text
on spaces and give each whole word its own number. "The cat sat" becomes three
tokens: `The`, `cat`, `sat`.

This works until someone types a word that isn't on your list. And someone
*always* does. Your list might have the 50,000 most common English words, but not
"Patala," not "unfriendable," not a typo like "teh," not a Hindi word, not a
brand-new slang term. What do you do with a word you've never listed?

The old answer was a special token usually written `<UNK>`, for "unknown." Every
word not on the list became the same `<UNK>`. And this is where it hurts:

- A rare name, a typo, a foreign word, and a technical term **all become the
  identical blank token.** The model can no longer tell them apart. The actual
  information about what was written is simply gone.
- Languages that build long words out of pieces (Turkish, Finnish) have *way* more
  word-forms than any list can hold, so they hit `<UNK>` constantly.
- Worst of all for a model that *writes*: it can never produce a word that wasn't
  on its list in the first place. You can't generate what you can't represent.

For years this was just accepted as the cost of doing business.

## A crude patch: chop words to their roots

One early fix came from search engines. If you search for "running," you probably
also want pages that say "run" or "ran." So search systems ran a step called
**stemming**: mechanically chop the ends off words to reduce them to a common
root, so "connect," "connected," and "connecting" all become "connect" (the famous
Porter stemmer, 1980).

This helped matching, but notice what it does: it doesn't *understand* the
variations, it *destroys* them. It throws away the ending. It's lossy (you can't
get the original back), it's tuned for English, and it's useless if you want to
*write* text rather than just match it. It patched one symptom of the vocabulary
problem without touching the real disease.

## The opposite extreme: spell everything out of letters

If a fixed list of *words* is the problem, why have words at all? You could work
at the level of individual **characters** instead. English needs only about 26
letters (plus punctuation and digits) to spell *any* word that will ever exist.
Suddenly there's no such thing as an unknown word, because you can always spell it
out.

Researchers tried this around 2015, and it genuinely solved the unknown-word
problem. But it lost for two practical reasons. First, everything gets *long*: the
word "tokenization" is one unit as a word but a dozen units as letters, and long
sequences are slow and expensive for a model to process. Second, the model now
has to *relearn what words even are* from scratch, letter by letter, before it can
do anything useful, which wastes its effort.

So the field had two options that each failed in the opposite way. Whole words:
short and meaningful, but can't handle anything new. Single letters: handles
anything, but long and meaningless. The answer, it turned out, was in between.

## The clever middle: break rare words into reusable pieces

The winning idea is **subwords**: keep common words as single tokens, but break
*rare* words into smaller pieces that themselves get reused. So "the" stays one
token, but "unfriendable" might become "un" + "friend" + "able", three pieces the
model has seen many times before, in many other words.

This gets the best of both worlds. Common text stays short (common words are one
token each). Nothing is ever truly unknown (worst case, a strange word falls back
to very small pieces). And related words share pieces: "friend," "friendly,"
"friendship," and "unfriendable" all contain the "friend" chunk, so they're not
total strangers to the model.

The surprising part is *where this idea came from*. The specific method, called
**byte-pair encoding (BPE)**, was invented in **1994 by Philip Gage, as a
file-compression trick**, completely unrelated to language. His algorithm was
dead simple: find the most common pair of neighboring symbols in a file, replace
every occurrence with a new single symbol, and repeat. Do that a few thousand
times and you've squeezed the file smaller.

In 2015, three researchers (Sennrich, Haddow, and Birch) realized this
compression trick was *exactly* what open-vocabulary language needed. Why? Because
the goal is the same in both cases: **find a small set of units that covers a lot
of text efficiently.** Run BPE's "merge the most common pair" step on a big pile
of text, and the frequent words naturally merge into whole tokens while rare words
stay broken into pieces. That is precisely the fixed-list-but-still-open-ended
property that had eluded the field for decades. A compression algorithm quietly
dissolved a thirty-five-year-old problem. That moment is the hinge of the whole
story.

## A few cousins in the family

Once subwords caught on, variations appeared, and two distinctions are worth
knowing without getting into the weeds:

- **WordPiece** (Google, originally 2012, for Japanese and Korean voice search)
  uses almost the same "merge pieces together" shape as BPE, but decides *which*
  pair to merge by a slightly smarter rule. It merges the pair that best improves
  how well the vocabulary explains the text, rather than just the most frequent
  pair. This is the tokenizer that later powered BERT. (People often wrongly think
  WordPiece was invented for BERT; it's six years older.)
- **Unigram** (2018) flips the direction entirely. Instead of starting from tiny
  pieces and merging up, it starts with a huge pile of candidate pieces and
  *throws away* the least useful ones until the list is the right size. Because
  it's built on probabilities, it can even offer several different ways to chop the
  same word, which helps make models sturdier against noise.
- **SentencePiece** (2018) is a common source of confusion: it's not an algorithm
  at all, it's a popular *software library* that implements the above. Its real
  contribution was handling *spaces* cleanly, so the whole process works on raw
  text in any language and can be perfectly reversed back to the original.

## The last hole: work at the level of raw bytes

Subword BPE still had one small leak. It was usually built from *characters*, and
there are around 150,000 possible characters in the world (every script, symbol,
and emoji). You can't put them all in your base list, so a truly exotic character
could still slip through as unknown.

The fix, introduced with GPT-2 in 2019, was to go one level lower, down to
**bytes**, the raw way computers store *any* text. There are only 256 possible
byte values, and *every* piece of text, in every language, including emoji, is
made of them. Put all 256 in the base list and run BPE on top, and now there is
*no* possible input that can't be represented. The `<UNK>` token that haunted NLP
for decades finally disappeared for good. You simply cannot hand this kind of
tokenizer anything it can't encode.

This is the design nearly every modern model, including the one used in this book,
still uses today.

## Where it all landed

By the early 2020s the algorithm was settled, and the remaining work was just
making it *fast*, fast enough to be an invisible, taken-for-granted step (tools
like OpenAI's tiktoken). The long journey from "one number per word, everything
else is unknown" to "break rare words into reusable byte-level pieces, and nothing
is ever unknown" was complete.

The one-sentence version: **for thirty-five years, tokenization fought a fixed
list against an endless language — and the fight ended when a 1994 file-
compression algorithm turned out to be the perfect peace treaty.**

## If you want to go deeper

The best full scholarly retrospective is Mielke, Alyafeai, and colleagues (2021),
"Between words and characters: A Brief History of Open-Vocabulary Modeling and
Tokenization in NLP" (https://arxiv.org/abs/2112.10508). For a hands-on modern
walkthrough, Andrej Karpathy's "Let's build the GPT Tokenizer" and his minbpe
repository (https://github.com/karpathy/minbpe) are excellent. Full dates,
citations, and attribution corrections are in the companion file
`tokenization_history.md`.
