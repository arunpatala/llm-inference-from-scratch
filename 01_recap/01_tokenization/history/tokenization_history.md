# A history of tokenization (chapter section draft)

Grounded by a research survey (sources at the end); dates/attributions
cross-checked against ACL Anthology / arXiv / dblp. [unverified] flags kept.

Organizing thesis: **tokenization history is the story of escaping the
fixed-vocabulary bottleneck.** From ~1980 to ~2015, NLP lived under a hard
constraint — the vocabulary had to be finite (the embedding table and output
softmax scale with it), yet language is open-ended. Every era's tokenization is
an answer to that tension, and the hinge of the whole story is a 1994 *data-
compression* algorithm getting repurposed for NLP in 2015.

## The through-line: escaping the vocabulary bottleneck

1. **Stemming / lemmatization (1980s).** In information retrieval, "connect",
   "connected", "connecting" should match one query — but with no way to
   *represent* morphology, the fix was to *destroy* it: chop words to a common
   root (Porter stemmer, 1980). Solved surface-form mismatch; cost: lossy,
   irreversible, English-centric, useless for generation. This is *normalization*,
   not tokenization — the prehistory of the OOV problem, not the subword lineage.

2. **Word-level + `<UNK>` (SMT; word2vec 2013; GloVe 2014).** A fixed vocabulary
   is computationally necessary, so you cap it (~30k-100k) and dump everything
   else into a single `<UNK>`. This *created* the central weakness of the era:
   rare words, names, and typos all collapse to the same uninformative token;
   morphologically rich languages (Turkish, Finnish, Russian) blow past any cap
   so `<UNK>` rates explode; and generation can never produce a word it never saw.
   Statistical MT (Moses, 2007) papered over names/numbers by copying unknown
   words through untranslated, but that did nothing for morphology. [This is the
   exact pain the byte-floor guarantee later dissolves — Q11.]

3. **Character-level (Kim et al. 2015, char-aware neural LMs).** Solves OOV
   *completely* — ~26 symbols spell anything, and morphology comes for free; it
   even beat word baselines on morphologically rich languages with far fewer
   parameters. Why it lost as a standalone: sequences get very long (slow, wasted
   capacity) and the model must relearn word structure from scratch, straining
   long-range dependencies. It proved sub-word signal was valuable without being
   the answer. [The Q2 char-vs-word trade-off, playing out historically.]

4. **BPE for NMT (Sennrich, Haddow & Birch) — THE HINGE.** The exact insight:
   *frequent words stay whole; rare words break into meaningful, reusable pieces*
   — giving a fixed vocabulary AND an open vocabulary at the same time, the single
   trick that dissolved the decades-old bottleneck. A *compression* algorithm was
   the right tool because compression IS the goal: find a small set of units that
   covers the corpus efficiently. Names transliterate via characters, compounds
   translate compositionally, cognates share subwords. Problem solved: translate
   rare/unseen words with no `<UNK>` and no unbounded vocabulary.

5. **WordPiece — likelihood, not frequency.** Same bottom-up merging shape as BPE
   but a different merge criterion: BPE greedily merges the *most frequent* pair;
   WordPiece merges the pair that most *increases corpus likelihood* under a
   unigram LM. Born (Schuster & Nakajima 2012) to handle the effectively infinite
   vocabularies of Japanese/Korean *speech*, where whitespace gives no help. The
   likelihood-vs-frequency fork is the key technical branch in the subword family.

6. **Unigram LM (Kudo 2018) — top-down, probabilistic.** BPE/WordPiece are
   bottom-up (start from characters, merge up). Unigram is top-down: start with a
   huge candidate vocabulary and *prune* the least-useful units by corpus
   likelihood (EM) until you hit the target. Because it's a probability model it
   can emit *multiple* scored segmentations, enabling *subword regularization*
   (train on sampled segmentations as augmentation) — more robust than BPE's
   single greedy deterministic split. [Connects to Q27: multiple valid
   tokenizations of one string; Unigram makes that explicit.]

7. **SentencePiece (Kudo & Richardson 2018) — a library, not an algorithm.**
   BPE/WordPiece assumed a prior whitespace tokenizer: fine for English, wrong for
   Japanese/Chinese/Thai, and lossy (you can't always reconstruct spacing).
   SentencePiece treats the raw string (spaces as a symbol, the visible marker)
   as input, so it's language-agnostic, needs no pre-tokenizer, and is losslessly
   reversible. It implements *both* BPE and Unigram. [The reversible-whitespace
   marker is the ancestor of the byte-level space-token you saw as the Q12 leading
   space.]

8. **Byte-level BPE (GPT-2, 2019) — closing the last OOV hole.** Even
   character-level BPE has a base-vocabulary problem: Unicode has ~150k code
   points, so you either include them all (huge) or still have OOV characters
   (emoji, rare scripts). Working over *bytes* caps the base at 256 and guarantees
   any input is representable — so `<UNK>` disappears *entirely*, for any byte
   string in any script. [This is exactly the guarantee you verified in exercise 4
   and reasoned through in Q11.]

9. **tiktoken / standardization (2022+).** Once the algorithm was settled the
   remaining problem was engineering — speed and reproducibility at scale. tiktoken
   (Rust-backed byte-level BPE) made tokenization a fast, boring, shipped-and-
   forgotten dependency. The vocabulary bottleneck that defined NLP from 1980 to
   2015 is gone.

## Dated milestones

- 1980 — Porter stemmer (Porter, *Program* 14(3):130-137). Normalization by
  suffix-stripping; the field's main answer to morphology for decades.
- 2007 — Moses SMT toolkit (Koehn et al., ACL 2007 demo); `tokenizer.perl` the
  de facto rule-based standard; unknown words copied through verbatim.
- 2013 Jan — word2vec (Mikolov et al., arXiv:1301.3781): dense word vectors,
  fixed vocab, `<UNK>`.
- 2014 Oct — GloVe (Pennington, Socher, Manning, EMNLP 2014): same paradigm.
- 2015 Aug — Character-Aware Neural LMs (Kim, Jernite, Sontag, Rush,
  arXiv:1508.06615): char-CNN into a word LSTM; strong evidence subword signal
  matters.
- 1994 Feb — original BPE as data compression (Philip Gage, *The C Users
  Journal* 12(2)). A competitor to LZW; nothing to do with language.
- 2015 Aug — BPE repurposed for NMT hits arXiv (Sennrich, Haddow, Birch,
  arXiv:1508.07909) — same month as Kim et al.
- 2016 Aug — that paper published at ACL 2016 (P16-1162). THE hinge.
- 2012 Mar — WordPiece born (Schuster & Nakajima, "Japanese and Korean Voice
  Search", ICASSP 2012) — three years BEFORE Sennrich, six before BERT.
- 2016 Sep — WordPiece in production (Google NMT / GNMT, arXiv:1609.08144).
- 2018 Apr — Unigram LM + subword regularization (Kudo, arXiv:1804.10959).
- 2018 Oct — BERT popularizes WordPiece (Devlin et al., arXiv:1810.04805);
  ~30k vocab; the `##` continuation marker.
- 2018 Nov — SentencePiece (Kudo & Richardson, EMNLP 2018 demo, D18-2012).
- 2019 Feb — GPT-2 byte-level BPE (Radford et al.): base 256, 50k merges,
  vocab 50,257; `<UNK>` gone.
- ~2022 Dec — tiktoken open-sourced (exact date [unverified]).
- 2024 Feb — Karpathy "Let's build the GPT Tokenizer" + minbpe.

## Myths / commonly-garbled facts (attribution)

- "Sennrich invented BPE." No — **Gage (1994)** invented it as compression;
  Sennrich et al. *repurposed* it for NMT (and their variant merges to build a
  vocabulary, not to compress a file).
- "BPE-for-NLP is 2016." Usually cited as ACL 2016 but it was on **arXiv in Aug
  2015** — a 2015 idea.
- "WordPiece was made for BERT (2018)." No — **Schuster & Nakajima 2012**, for
  Japanese/Korean voice search; popularized by GNMT (2016) then BERT (2018).
- "WordPiece is just BPE." Shared shape, different objective: **BPE = most
  frequent pair; WordPiece = pair that maximizes corpus likelihood.**
- "SentencePiece is an algorithm." No — a **library** implementing both BPE and
  Unigram. People saying "SentencePiece" usually mean **Unigram LM** (Kudo 2018).
- "GPT-2 invented BPE." GPT-2's contribution is the **byte-level base
  vocabulary**, not BPE or subword tokenization.
- "Unigram is a kind of BPE." No — Unigram never merges; it starts big and
  **prunes** (top-down), the opposite direction.
- "Stemming is tokenization." Stemming/lemmatization are lossy *normalization*
  applied after tokenizing — prehistory of the OOV problem, not the subword line.
- [unverified/secondary]: tiktoken's exact release month; BERT's exact 30,522
  vocab figure (from the released vocab file, not stated in the paper); Gage's
  exact page range (secondary catalog, *The C Users Journal* not online).

## How the history maps onto the interview

- OOV / `<UNK>` pain (era 2) -> the byte-floor "no unknown ever" guarantee (Q11,
  exercise 4).
- Morphologically rich languages blowing past the vocab cap -> fertility and the
  multilingual tax (Q24).
- Char-vs-word trade-off (era 3) -> Q2.
- Likelihood-vs-frequency merge fork (WordPiece) -> the merge criterion behind
  the vocab the model actually ships with.
- SentencePiece reversible whitespace marker -> the byte-level leading-space
  token (the Q12 space-attached-to-word behavior).
- Byte-level BPE closing the last OOV hole -> Q11 / the whole byte-floor thread.
- Multiple valid segmentations (Unigram) -> Q27 non-compositionality and the
  canonical-vs-non-canonical debates in Future Directions S4.

## Sources

Primary: Gage 1994 (https://www.derczynski.com/papers/archive/BPE_Gage.pdf) ·
Porter 1980 (https://tartarus.org/martin/PorterStemmer/def.txt) · Moses
(https://aclanthology.org/P07-2045/) · word2vec (https://arxiv.org/abs/1301.3781)
· GloVe (https://aclanthology.org/D14-1162/) · Kim et al. 2015
(https://arxiv.org/abs/1508.06615) · WordPiece origin
(https://research.google/pubs/japanese-and-korean-voice-search/) · Sennrich et al.
(https://aclanthology.org/P16-1162/ , arXiv https://arxiv.org/abs/1508.07909) ·
GNMT (https://arxiv.org/abs/1609.08144) · Unigram/Subword Regularization
(https://arxiv.org/abs/1804.10959) · SentencePiece
(https://aclanthology.org/D18-2012/) · BERT (https://arxiv.org/abs/1810.04805) ·
GPT-2 (https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
Retrospectives: Mielke, Alyafeai et al. 2021, "Between words and characters: A
Brief History of Open-Vocabulary Modeling and Tokenization in NLP"
(https://arxiv.org/abs/2112.10508) — the best scholarly retrospective · Karpathy
2024 minbpe (https://github.com/karpathy/minbpe) · Simon Willison's notes
(https://simonwillison.net/2024/Feb/20/lets-build-the-gpt-tokenizer/) ·
Machine Translate wiki (https://machinetranslate.org/byte-pair-encoding) ·
tiktoken (https://github.com/openai/tiktoken)
