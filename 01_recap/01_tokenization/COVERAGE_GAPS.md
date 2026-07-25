# Tokenization chapter — coverage gaps (from a curricula diff)

A research pass compared our coverage against the standard tokenization curricula
(Karpathy/minbpe, Hugging Face NLP+LLM course Ch2/Ch6, Jurafsky & Martin SLP3,
CS224N, Voita, Mielke et al. 2021 survey, Raschka's LLM-from-scratch). Sources at
the bottom. Bottom line: our inference-focused coverage EXCEEDS standard curricula
on inference topics; below are the genuine gaps to close and the things we can
skip.

## Our confirmed strengths (exceed standard curricula — keep as differentiators)
Chat templates (Jinja2, generation prompt, wrong-template degradation, injection),
streaming detokenization + partial-UTF-8, speculative-decode shared vocab (+FR-Spec/
OOV-token), glitch/undertrained tokens, token healing, fertility/multilingual tax,
prefix caching, KV-cache-motivation-from-the-naive-loop, tied-embedding cost,
padding side + packing/varlen. Standard courses barely touch these.

## Genuine gaps to close (inference-relevant, near-universal in HF material)
Ranked by value. Concept notes for gaps 1-6 (what / why-it-matters / connection,
grounded on Qwen where marked) are written up in
`09_pipeline_stages_and_template_extensions.md` — still to interview/fully ground.

- [ ] 1. NORMALIZATION as an explicit pipeline stage — Unicode NFC/NFKC, accent
      stripping, lowercasing, and its LOSSINESS (irreversible, breaks exact
      round-trip and prefix-cache match). We mention case sensitivity but not
      normalization as a named stage. Universal (HF Ch6, tokenizers pipeline,
      SLP3). Strengthens the prefix-cache (Q28) and streaming (Q21) sections.
      Also: the full pipeline framing normalize -> pre-tokenize -> model ->
      post-process -> decode.
- [ ] 2. OFFSET MAPPING / token<->char alignment (`word_ids`, offsets). Absent
      from our coverage; headline feature of HF Ch6 (NER + QA). Inference-relevant
      for guided/structured decoding, token-level attribution/highlighting,
      logprob->span mapping, grammar-constrained output.
- [ ] 3. TRUNCATION / max-length / context-window management + sliding-window
      stride & overflowing tokens. We cover padding side and packing but never
      truncation or long-input chunking. Directly an inference concern (fitting
      prompts to the context window). HF Ch2/Ch6.
- [ ] 4. ATTENTION MASK (and briefly token-type IDs) as the tensor paired with
      input IDs. Our pipeline goes tokenize -> tensor -> model; the padding/
      attention mask is the universally-taught companion and is load-bearing once
      padding side + packing are in play (Q18). Make it explicit.
- [ ] 5. PREFILL / assistant-message continuation (`continue_final_message`,
      stripping the trailing EOS). We have generation prompt (Q5) and
      chat-template Q3 mentions continue_final_message, but prefilling a partial
      assistant turn (reasoning/JSON prefill) deserves explicit treatment as a
      distinct inference technique.
- [ ] 6. TOOL / FUNCTION-CALLING rendering inside chat templates (tool defs as
      JSON schema, `tool_calls`, `tool`-role responses). Core to agentic
      inference; we list chat templates generically but not tool-call
      serialization. (04_out_of_scope mentions tool-call tokens exist — expand.)
- [x] 7. MULTIMODAL tokenization — DONE. Interviewed in
      `07_image_tokenization/01_questions.md` (all 10 Q answered): patches ->
      continuous embeddings (not IDs) -> projector -> placeholder-token splice;
      the O(n^2)/prefill-dominated serving cost + visual-token pruning; discrete
      VQ for generation vs continuous for understanding; M-RoPE 2D positions;
      "vision vectorizer not tokenizer". Grounded via search (LLaVA/Qwen-VL).
- [ ] 8. FAST vs SLOW tokenizers (Rust `tokenizers` vs pure-Python) as a named
      architectural distinction (fast tokenizers enable offsets + throughput).
      Fold into the Q9/Q30 speed section. (Minor.)

## Softer / historical gaps (mention for completeness)
- [ ] 9. History bridge: FastText char-n-gram embeddings (Bojanowski 2017),
      Character-Aware Neural LM (Kim 2015, char-CNN+Highway), hybrid word-char NMT
      (Luong & Manning 2016) — the bridge from word embeddings to subwords our
      history currently skips (readable version jumps word2vec -> char -> BPE).
- [ ] 10. MORPHOLOGY as the linguistic "why" behind subwords (morphemes,
      inflectional vs derivational, agglutinative-language token tax) — gives the
      fertility/multilingual section a mechanistic grounding. SLP3/CS224N/survey.
- [ ] 11. BPE-Dropout / subword regularization / sampling multiple segmentations
      (Voita; Unigram can sample). Relates to Q27 determinism but is a distinct
      (mostly training-side) concept.
- [ ] 12. Herdan/Heap's Law (vocab grows sublinearly with corpus). SLP3; connects
      to the vocab-scaling-laws Future Direction.

## Commonly taught but intentionally SKIPPED (inference-irrelevant)
Minimum edit distance; Penn Treebank / rule-based / Unix tokenization; sentence
segmentation; token-type IDs & [CLS]/[SEP] post-processor templates (encoder-era;
keep attention mask though); building/training a tokenizer block-by-block +
trainer hyperparameters (training-time); Bayesian nonparametrics / marginalization
/ Morfessor / unsupervised Chinese word segmentation (research depth, low inference
payoff); corpus-linguistics/ELIZA/regex-operator tutorials.

## Sources
Karpathy minbpe github.com/karpathy/minbpe; HF NLP/LLM course chapter2/4-6 &
chapter6/1-9 (huggingface.co/learn); HF tokenizers pipeline + tokenizer_summary;
HF chat_templating(_writing); tiktoken; Jurafsky & Martin SLP3 Ch2
(web.stanford.edu/~jurafsky/slp3/2.pdf); CS224N 2019 lecture12-subwords; Lena
Voita nlp_course; Mielke et al. 2021 (arxiv 2112.10508); Raschka LLMs-from-scratch
ch02 + bpe-from-scratch blog.
