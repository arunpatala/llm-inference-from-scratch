# Tokenization — questions

Answer inline under each question, replacing the placeholder. Skip or write
"not sure" where there's no real take yet — a thin answer is a signal to ask
a follow-up later, not something to fill in generically.

## Set A — building the chapter

1. In your own words: what is a token, and why does the model need numbers
   instead of raw text?

   Text has to be converted into numbers a model can work with. One-hot
   encoding of whole words is the simplest example, but the dictionary is
   fixed — anything outside it becomes an unknown token. The other extreme is
   one-hot encoding every character: now all text is representable, but you've
   lost the word-level structure that related words share.

   So what we want is a scheme where any text can be mapped through a
   dictionary, while still keeping a structural handle on how related words
   share pieces. That's tokenization: split text into word- or subword-level
   pieces called tokens, and have the tokenizer map each to a token ID (which
   later indexes into an embedding/vector — covered in the embeddings chapter).
   New/unseen words are still representable because a word gets broken down
   toward the character level as far as needed.

   Why numbers at all: a neural network's activations are floats/vectors, and
   it can't handle strings directly. Text has to be mapped into IDs (then
   vectors) for the transformer to process it.

   On "unknown tokens": those are a word-level/one-hot problem, not a BPE one.
   A byte-level BPE tokenizer never produces a true unknown — a word is broken
   into smaller subwords until it finds pieces that are in-vocab, and the
   smallest pieces (bytes) are always in the vocab. So there's always a
   representation, worst case byte-by-byte.

   Precise version: the guaranteed floor is the 256 raw *bytes*, not
   characters. BPE runs on the UTF-8 byte stream, so a Unicode character isn't
   atomic — it's encoded to 1-4 bytes first. Three different layers: character
   count != byte count != token count. Real numbers from Qwen3-0.6B's own
   tokenizer (verified on the Mac, no CUDA): "naive" is 5 chars / 6 bytes /
   3 tokens; "cafe" (with the accent) 4 chars / 5 bytes / 2 tokens; the emoji
   U+1FAF8 is 4 bytes / 3 tokens; a family emoji is 18 bytes / 7 tokens; yet
   the whole Japanese greeting konnichiwa merges to 1 token. Common sequences
   earn full merges; rare ones fragment toward bytes. This byte floor is why
   there's never a true unknown (Q11), and it's the direct cause of the
   streaming-decode bug in Q21: decoding one token at a time can emit a broken
   partial-UTF-8 "replacement char" mid-stream, so a correct streaming decoder
   must buffer bytes until they form a valid character. Reproduced on Qwen3:
   the U+1FAF8 emoji streams three replacement chars token-by-token but decodes
   correctly only when its 3 tokens are combined.

2. Why subword tokenization (BPE-style) instead of whole-word or
   character-level — what problem does splitting into pieces like "token" +
   "ization" actually solve?

   Word-level treats closely related forms as completely separate entries —
   "token", "tokens", "tokenization" share nothing, even though they're
   obviously related (stemming/morphology). Character-level does let anything
   be represented, but each character is too small a unit to be meaningful on
   its own, and the sequences get long. Subword splitting gives the best of
   both: related words share a common subword piece ("token"), so they aren't
   totally independent symbols, while common words still stay short.

   Why sequence length matters (the inference angle): char-level can produce
   5-10x more tokens for the same text, and the transformer pays for that on
   both the input and the output side. Attention is quadratic in sequence
   length and KV memory grows with it, so more tokens means more compute and
   more memory. There's also an upper limit — the context length effectively
   shrinks when each unit of text costs more tokens.

   Verified on the real Qwen3 tokenizer (all CUDA-free, on the Mac):
   - `08_coding_exercises/exercises/01_encode_decode.py`: "Tokenization turns
     text into numbers." (37 chars) -> 7 tokens, and "Tokenization" splits
     into `Token` + `ization`, exactly the subword example. Common words are
     one token each.
   - `08_coding_exercises/exercises/06_bytes_per_token.py`: bytes-per-token is
     the compression metric. English prose 4.40, Python code 2.67, a random
     20-digit string 1.00 bytes/token. Higher = the tokenizer found big merges
     (frequent text); lower = falling back toward raw bytes (rare text).

3. Qwen3's vocab is ~151k entries — bigger vocab is a trade-off, not a free
   win. What's the trade-off as you understand it?

   A bigger vocab grows the embedding table, the computation, and the overall
   model size (forward-reference: the embeddings chapter shows exactly what
   that costs). It can also cost you the shared-piece structure between related
   words — in the extreme, if every whole word gets its own entry (word-level),
   morphological variants stop sharing anything again, and rarer entries show
   up less often, so they're less well-trained.

   The trade-off is two-sided. A vocab that's too *small* means the same text
   has to be represented with more tokens (char-level in the extreme), and if
   the pieces are too fine you also lose useful subword overlap. A vocab that's
   too *big* grows the embedding table and the final decode layer (LM head).
   So it's: too many tokens per text (vocab too small) versus too big a
   dictionary and output layer (vocab too big). Bigger vocab shortens
   sequences — which helps decode throughput — but isn't free, which is why
   you don't just make it enormous.

4. We're using the Instruct model, so a raw prompt gets wrapped in a chat
   template before tokenization. What do you think that template actually
   contains, and why can't we just tokenize the user's raw text directly?

   Instruct models are made for the user-asks / assistant-answers pattern,
   fit into next-token prediction. The model is trained with a particular
   template to handle that back-and-forth. The conversation is an array of
   turns, each turn a role plus its text — a JSON array. But the transformer
   can only see text/tokens, never objects, so that array has to be mapped
   down to a flat string first. That mapping is what the chat template does.
   Tokenizing the user's raw text directly would drop all the role/turn
   structure the model was trained to expect.

5. Walk through the actual pipeline: raw prompt string → what happens →
   tensor of token IDs the model consumes. What are the steps as you
   understand them?

   The steps: raw prompt -> build a messages list with a user turn -> apply
   the chat template with add_generation_prompt=True (which appends the open
   assistant turn) -> tokenize the rendered string into a token tensor -> pass
   it to the model -> the model outputs a probability distribution over the
   next token -> take the next token from it -> append it to the sequence and
   run the model again -> repeat.

   Scoping simplification (deliberate): for now the "pick a token" step is
   greedy — just take the single most probable token (argmax over the logits).
   Real sampling (temperature, top-k, top-p) is a separate concern that doesn't
   change the pipeline's shape, so it's deferred to a later chapter.

   The output side closes the loop: each sampled token id is decoded back to
   text (the exact reverse of the encode step in Q1 — id -> token string ->
   characters) so the user sees words, not numbers. Basic version: run the
   tokenizer's decode on the generated ids. The *streaming* subtlety — emitting
   text token-by-token as it's generated without breaking a multi-token
   character (the partial-UTF-8 `�` bug in Q21 / `02a`) — is the part deferred
   to Q8. So the simple mental model for Module 0 stays: greedy id-in, and
   id-out decoded straight back to text.

   That last loop (append the new token, re-run the model) is autoregressive
   decoding, and it's the spine of the whole book. One precision that motivates
   everything: the model doesn't emit just the next token's scores — at every
   input position it emits a full vector of logits over the entire vocabulary
   (151936 for Qwen3). At inference you only use the *last* position's vector,
   softmax it into a distribution over all possible next tokens, and (greedy)
   take its argmax.

   Done naively, every step re-runs the whole sequence, so at step N+1 you
   recompute positions 1..N that you already ran at step N. And here's why that
   is pure waste, not just repetition: causal masking means each position only
   attends to tokens at or before itself, so appending a token at the end
   cannot change any earlier position's output — it is byte-for-byte identical
   every step. That makes it both wasteful (O(n^2) over the generation) and
   safe to avoid. The fix is to store and reuse each past token's key and value
   vectors instead of recomputing them (each new step then computes only the
   new token's query plus its own K/V, attending to the cached K/V) — that is
   the KV cache, and it is exactly Module 1. This recap chapter only needs to
   set the hook: the naive loop here is deliberately the slow baseline.

   The loop also needs a stop condition: it ends when the sampled token is
   `<|im_end|>` (F1 / Q6), otherwise it would generate forever.

   Two separate steps worth keeping distinct: the template is a text-formatting
   step (array -> string), and tokenization is a separate step after it
   (string -> IDs).

   Confirmed on the real tokenizer (`demos/chat_template_demo.py`):
   `apply_chat_template(..., tokenize=False)` returns a Python `str`, not token
   IDs — so the template really is text-only, and tokenization is the separate
   downstream step (that string -> 43 token IDs for the sample conversation).

   One inference-critical piece the render surfaced: the generation prompt.
   With `add_generation_prompt=True`, the rendered string ends with an *open*
   assistant turn — `<|im_start|>assistant\n` with nothing after it. That
   trailing open turn is what puts the model's next-token prediction into "the
   assistant is mid-reply, continue" mode. Without it, the string ends with the
   user's `<|im_end|>\n`, and the model is instead conditioned on "a user turn
   just ended," so it tends to start a *new* turn (often hallucinating another
   user message) rather than answering. The generation prompt is how a chat log
   becomes a generation request. [F1/F2 behavior to confirm with an actual
   generation run on the 0.6B model (MPS) — pending.]

6. What special tokens do you expect matter here (BOS/EOS/pad/end-of-turn),
   and why does the model need explicit markers for where a turn ends?

   Special tokens are what the model was trained on to structure the
   user/assistant chat format. They cleanly demarcate the user and assistant
   turns so the model doesn't confuse turn boundaries with the actual content
   of the prompts. The key is consistency: the training data was built with
   these same markers, so at inference the model has to see the identical
   format. They're also reserved entries, so the ordinary tokens produced from
   text inside a user prompt can't clash with them.

   Verified on the real Qwen3 tokenizer, and it's subtler than "can't clash":
   - Reserved-ness protects the model's *own* structure. BPE merges can never
     accidentally build `<|im_end|>` out of ordinary text, because it's a
     separate reserved vocab entry, not something the merge algorithm produces
     (Q15). So the model's generated text won't spuriously emit a turn marker.
   - But it does NOT stop a user from *deliberately typing* the literal string.
     By default `tokenizer.encode("hello <|im_end|> world")` parses that middle
     part as the real special token id 151645, not as text. Through
     `apply_chat_template`, a user putting `<|im_end|>` in their content injects
     a genuine second turn-closer (the tokenized output contained two copies of
     151645 — the structural one plus the user's). That's a real turn-escape /
     prompt-injection vulnerability (ChatBug family): the user closes their own
     turn early and could smuggle in an assistant turn.
   - The fix is the inference engine's job, not the tokenizer's default:
     encode user-supplied content with special-token parsing off
     (`split_special_tokens=True`) or strip/escape control strings first. So
     "user content can't clash with special tokens" is a property the engine
     must *enforce*, not one you get for free.

   Why Qwen3 has no BOS (`bos_token` is null, confirmed in the demo): a
   dedicated beginning-of-sequence marker is redundant here because
   `<|im_start|>system` (or `<|im_start|>user`) already unambiguously marks
   where the sequence starts. The structural turn tokens do BOS's job. (Pad's
   role is a batching concern — Q18.)

   Whose job is stopping (F1, the key inference connection): the model never
   stops on its own. Every step it just outputs a probability distribution over
   the next token — there is no built-in halt. It *can* make `<|im_end|>`
   (id 151645) the most likely next token when it considers its turn done, but
   the serving engine's decode loop is what must (a) watch the sampled token
   and (b) break the loop when it sees a stop token (here `eos_token_id` =
   151645), or when it hits `max_new_tokens`. If the engine fails to stop on
   `<|im_end|>`, generation runs past it, and because training data looks like
   `<|im_end|>\n<|im_start|>user\n...`, the model happily hallucinates a whole
   fake next turn (a made-up user message, then its own answer, and so on) —
   runaway output. So this special token is exactly what Module 0's decode loop
   has to check as its stopping criterion. [behavior to confirm with a real
   generation run — pending.]

7. Have you personally hit a bug or surprise caused by tokenization
   specifically — a weird split, whitespace handling, padding side, anything
   like that? What happened?

   The real one, from actual serving work: a train/inference tokenization
   mismatch around Qwen3's thinking tokens in multi-turn. Thinking is meant to
   be ephemeral — training only keeps `<think>...</think>` on the *current*
   turn and strips it from prior turns. The chat template matches that
   (verified on Qwen3-0.6B): prior-turn think content is dropped from history,
   and `enable_thinking=False` even stamps an empty `<think>\n\n</think>` onto
   the current turn. The bug is when the serving path *doesn't* match: naively
   feeding the model's full generated output — think tokens included — back
   into the next turn's history. Now from turn 2 onward the model sees a history
   *with* prior think blocks that it never saw in training, so it's
   off-distribution.

   Symptom: quality *quietly* degraded. No error, no crash, nothing in the
   logs — the answers just got worse as the conversation went on. That silent
   failure is the insidious part, and it's the same theme as serving with the
   wrong chat template (Q6 / chat-template Q4): tokenization and format
   mismatches don't throw, they silently make the model dumber. Multi-turn is
   where it bites, because single-turn has no prior turns to mishandle — the
   divergence only exists once there's history to strip (or fail to strip).

   (Separate, grounded-but-not-a-personal-surprise illustration for the
   chapter, from `08_coding_exercises/.../04_no_oov_byte_fallback.py`: nonsense
   ASCII "asdkfjhaslkdjfh" = 9 tokens vs the Japanese greeting こんにちは = 1 —
   token count follows training frequency, not how exotic the text looks.)

8. On the way out: tokens get sampled one at a time during generation. What
   has to happen to turn each new token ID back into the text the user
   actually sees, streaming?

   _(answer here)_

9. Does tokenization run on CPU or GPU, and does its cost matter at all next
   to the model's forward pass, or is it noise?

   [Taught, not from the author's own experience — author knew "CPU" but not
   the why/how. Verify the detokenization-bottleneck claim on a real serving
   profile before drafting.]

   CPU. Tokenization is a string/byte algorithm — regex pre-split, hash-map
   lookups of merge rules, applying merges to a byte sequence. That's branchy,
   sequential, variable-length work on text, not the dense parallel float
   matrix math a GPU exists for. There's no tensor to put on the GPU; you're
   operating on a string, and only the output (a small list of int IDs) is
   moved to the GPU as a tensor. The Rust-backed "fast" tokenizer (`is_fast`
   is True) runs on CPU threads.

   Cost, measured on Qwen3-0.6B (Mac CPU): a 601-token prompt encodes in
   0.58 ms (~1M tokens/sec); single-token decode is 2.4 us. Per request that's
   noise next to the GPU forward pass (many ms) and a full generation
   (seconds) — well under 1%.

   But it flips under load, mostly on the output side: 2.4 us/token is trivial
   alone, but you pay it per token, per request, across every concurrent stream,
   plus Python overhead — so at high concurrency streaming DETOKENIZATION can
   become a real serving bottleneck (vLLM and TGI have specifically optimized
   it). It also flips with very short generations at high QPS (long prompt,
   1-2 output tokens, thousands of requests), a slow pure-Python tokenizer, or
   cold-start building the ~151k merge table. So: CPU; per-request negligible,
   but not always free at scale. Connects to Q30.

10. One sentence for a colleague: why does tokenization deserve its own
    section in an inference book instead of a two-line footnote?

    Tokenization earns its own section because in serving it's both where
    correctness fails silently — a train/inference format mismatch degrades
    quality with no error to catch it — and where cost is set, since the token
    count it produces directly drives KV-cache size, context budget, and
    prefill/decode speed.

    (Assembled from the author's two pillars: silent correctness failure +
    token-count-drives-cost. Deliberately drops the "it's the first step" framing
    as footnote-level / not inference-specific.)

## Set B — grounded in documented tokenizer behavior

11. Byte-level BPE guarantees there's no true "unknown token" — even a
    character never seen in training still encodes via raw bytes. Why does
    that guarantee matter for a production inference engine, versus a
    tokenizer that could just fail on unseen input?

    First, separate two things the guarantee is easy to conflate. It promises
    *representability*: any input string encodes to some valid sequence of
    token IDs, and the tokenizer never silently filters or fails. It says
    nothing about *output quality* — the model may still produce garbage on a
    genuinely unseen word. (Though it isn't blind to it: a new word decomposes
    into subword pieces the model has seen, so there's partial signal — the
    shared-subword point from Q2.)

    The engine payoff: because encoding can never fail, the serving engine
    doesn't need a tokenizer-error path at all — no OOV branch, no "what if this
    is unrepresentable" handling on the request pipeline. That simplification is
    real: an error path there is code you'd otherwise have to write and test.

    The security angle, correctly bounded: a tokenizer that *could* throw on
    some input is an availability hole on a public endpoint, because an attacker
    can send that specific input on purpose — a cheap, asymmetric "poison
    string" that crashes the encoding step. The byte floor deletes that entire
    class: no such string exists, every request is encodable by construction.
    It does NOT make you DoS-proof in general — volumetric floods and giant
    prompts still need rate-limiting and length caps, separate machinery. What
    it removes is specifically the un-encodable-input failure mode.

12. `" hello"` and `"hello"` (leading space or not) tokenize to different IDs
    entirely. Why does that happen mechanically, and has it ever bitten you
    or surprised you?

    Saw the mechanism directly in `08_coding_exercises/exercises/01_encode_decode.py`:
    the token pieces came back as `['Token', 'ization', 'Ġturns', 'Ġtext',
    'Ġinto', 'Ġnumbers', '.']`. That `Ġ` is the leading space, baked into the
    token — byte-level BPE encodes a space (byte 0x20) as the visible symbol
    `Ġ` and attaches it to the *front* of the following word. So the token for
    a mid-sentence word is `" turns"` (space-turns), not `"turns"`. `Token`
    at the start of the sentence has no `Ġ` because there's no leading space.
    That's why `" hello"` and `"hello"` are genuinely different token IDs: the
    space is part of the token, not a separator between tokens.

    Lived version: this bit me finetuning a small LLM on ~500K examples with a
    fixed prompt format. The model overfit to that exact surface form, so a
    slight change at inference — an extra space from a copy-paste, especially in
    the system prompt — degraded quality silently (same silent-mismatch theme as
    Q7). The core problem in one line: the extra space changes the token
    count / token IDs themselves, which puts the input out-of-distribution from
    what the model was trained on. A small model heavily finetuned on one format
    is far more brittle to this than a big, diversely-trained one.

    Token-level mechanism, grounded on Qwen3 (and a correction to a first guess
    that the extra space "re-tokenizes" its neighbors — it doesn't): an extra,
    trailing, or doubled space just becomes a standalone `Ġ` token (id 220),
    with the surrounding words' tokens unchanged. "You are a helpful assistant."
    is 6 tokens; add a trailing space and it's 7 (the extra one is 220).
    "the system prompt" -> `[the, Ġsystem, Ġprompt]`; "the  system prompt" ->
    `[the, Ġ, Ġsystem, Ġprompt]`. So the damage isn't a cascade — it's a lone
    space token sitting in a position the model essentially never saw in
    training, which an overfit model is maximally sensitive to. [setup for
    exercise 2, leading-space/case.]

13. Case matters too — `"BPE"` and `"bpe"` are different tokens. Obvious once
    you know how BPE merges work, or does it still feel like a gotcha?

    Mechanism (author's answer): `'B'` and `'b'` are different bytes (0x42 vs
    0x62), so at the tokenizer level "BPE" and "bpe" are unrelated token
    sequences — there's no built-in relation between the two cases. It's a
    direct byte-floor consequence, not an arbitrary choice. (Whether the *model*
    learns they're related is an embedding-space question — deferred to the
    embeddings chapter; at the tokenizer level there's no relation.)

    The "so what" at serving time [taught — author asked; not from own
    experience]. Two real consequences, both grounded on Qwen3-0.6B:
    - Cost/context: uppercase is rarer in training, so it fragments into more
      tokens (same frequency mechanism as the nonsense-string finding).
      Measured: "assistant" = 1 token, "ASSISTANT" = 3; "hello" = 1, "HELLO" = 2.
      So ALL-CAPS/oddly-cased text costs ~2-3x the tokens for the same letters —
      more context budget, more KV, more per-token cost.
    - Prefix caching (Q28): reuse is exact token-id match, not text match.
      "You are a helpful assistant." -> [2610, 525, 264, ...] but
      "you are a helpful assistant." -> [9330, ...] — differs at token 0, a
      total cache miss. So system-prompt casing that drifts between requests
      silently kills prefix-cache reuse and recomputes the whole prefix every
      time. Latency and cost at scale.

    So: mechanism is obvious once you know it's byte-level; the consequences are
    a genuine gotcha because "it's just capitalization" hides a token-count
    blowup and a silent cache miss.

14. Many modern tokenizers, including Qwen's, deliberately split numbers into
    individual digits instead of merging `"12345"` into one token. Why would
    a tokenizer designer choose that on purpose?

    Three reasons for per-digit splitting (author's answer):
    - Generalization / coverage: any number, of any length, is built from just
      the 10 digit tokens, which are all extremely frequent — so the scheme
      handles arbitrary and large numbers cleanly.
    - Avoids rare-token conflation: if you merged multi-digit numbers into
      single tokens, most of those tokens ("1234", "8571", ...) would be rare
      and therefore poorly trained — inconsistent, undertrained representations
      (connects to the glitch-token problem, Q16).
    - Digit-level arithmetic with positional alignment: giving each digit its
      own token in positional order lets the model do arithmetic and comparison
      digit-by-digit and align place values — which it couldn't if "123" and
      "1234" were arbitrary unrelated tokens.

    Grounded on Qwen3-0.6B: every digit is its own token — "12345" ->
    ['1','2','3','4','5'], "1000000" -> 7 tokens, "3.14" -> ['3','.','1','4'].
    Pure single-digit, no grouping (differs from GPT-4/Llama-3, which chunk
    digits up to 3). The [verify] note is settled.

    Cost (exercise 6): numbers are token-expensive — a random 20-digit string
    is exactly 1.00 bytes/token, so a long ID, phone number, or big calculation
    burns context budget fast at 1 token per digit. Real inference consequence.

15. Special tokens like `<|im_start|>`/`<|im_end|>` aren't produced by the BPE
    merge algorithm at all — they're injected as reserved vocab entries that
    bypass merging entirely. Why do they need to be handled separately?

    A boundary/stop token has to be one atomic ID that ordinary text can
    neither fragment into nor accidentally produce — which is exactly why it
    lives outside the merge algorithm as a reserved entry.

    Without reservation, `<|im_end|>` would just go through normal BPE and come
    out as ~5 ordinary text tokens (verified in Q6: the literal string encodes
    to `[82639, 318, 6213, 91, 29]` with special parsing off). That's useless as
    a stop signal: (a) the engine's decode loop watches for one specific id
    (151645) to halt, and matching a *sequence* of ordinary tokens is fragile;
    and (b) those same ordinary tokens appear in normal text, so you could never
    tell a real turn boundary from text that merely contains those pieces.

    Reserved-ness has to hold in both directions:
    - User input: someone typing the literal string "<|im_end|>" (e.g. asking a
      question *about* special tokens, exactly what we're doing in this
      interview) must not be read as the real control token. That's the engine's
      job to enforce when encoding user content (Q6 injection finding).
    - Model output: the model's own generated answer must never *accidentally*
      emit the boundary mid-sentence. Because `<|im_end|>` is reserved and BPE
      can't build it from ordinary bytes, the marker only appears when something
      deliberately places it — so the model won't randomly end its turn while
      writing a normal reply.

16. "Glitch tokens" (the SolidGoldMagikarp phenomenon) are real vocab entries
    that make models output garbage when they appear, caused by a mismatch
    between the tokenizer's training corpus and the model's training corpus.
    Does that risk apply to a small model like Qwen3-0.6B, in your view?

    Yes — and arguably more so at 0.6B. First, the right framing: this is a
    model-training artifact, NOT a tokenization failure. The tokenizer encodes
    the token fine; the failure is that the model's *embedding* for that token
    was never trained. (And it's a single token, not an unseen combination —
    one glitch token alone produces garbage, whereas an unseen combination of
    well-trained tokens just generalizes imperfectly.)

    The causal chain: the tokenizer is built once, for a big shared corpus, and
    earns a vocab slot for anything frequent in *that* corpus. A smaller model
    is then trained on a subset with less capacity and training budget, so some
    of those tokens are barely or never seen during model training — their
    embeddings stay near random init, and the model has no idea what to do when
    one shows up.

    Maps directly onto Qwen3-0.6B: it reuses Qwen2's tokenizer (the full ~151k
    vocab, shared across the entire Qwen family including the big models), while
    the 0.6B has the least capacity/budget of the family. Exercise 5 showed the
    embedding table has ~267 rows the tokenizer never even emits — the
    definitional extreme: slots that exist with guaranteed-untrained embeddings.
    [optional: hunt for actual undertrained tokens on the real model via the
    "Fishing for Magikarp" method — anomalous embedding norm / near-mean
    embedding — before drafting.]

17. Vocab size isn't free — it directly sets the size of two of the model's
    biggest weight matrices (embedding table and LM head). For Qwen3-0.6B's
    ~151k-token vocab, what do you think that costs in raw parameters, and
    does that change how you think about vocab size as a design choice?

    The embedding table (vocab x hidden) dominates as the model gets smaller.
    Arithmetic, grounded on the real model: 151936 x 1024 = 155.6M params. The
    actual total is 596M (so "0.6B" is ~596M), and that one matrix is 26.1% of
    the model — over a quarter.

    Tying is the lever. Qwen3-0.6B ties the input embedding and the output LM
    head (tie_word_embeddings=True): one matrix used twice. Untied, it would
    need a second 155.6M matrix, pushing the model to 751.6M and making the two
    vocab matrices 41.4% of the whole — for zero new capability. So at 0.6B,
    tying is nearly mandatory, and there's real pressure to keep the vocab
    modest.

    The scaling take: embedding size is fixed by vocab x hidden, while total
    params scale with depth and width^2, so the fixed 155.6M is a shrinking
    fraction as the model grows — ~26% at 0.6B but only ~2% at 8B. That's why
    the big Qwen3 models (8B+) can afford to untie (03_algorithms) and small
    ones can't. Ties back to the Q3 trade-off: bigger vocab shortens sequences
    but grows exactly these two matrices.

    Related technique the author has seen: filtering/pruning vocab tokens during
    finetuning to cut out-of-domain vocab (restricting the output logit space /
    dropping unused token slots). [capture as a note if it recurs — real
    inference-adjacent optimization, connects to the untrained-token discussion
    in Q16.]

    Follow-up the author raised (belongs in the EMBEDDINGS chapter, capture
    there): why untie at all if tying is cheaper? Because input embedding
    (token->vector) and LM head (hidden->logits) do different jobs, and tying
    forces them equal. Tying is both param-saving AND a regularizer/inductive
    bias (Press & Wolf 2017, "Using the Output Embedding to Tie Word
    Embeddings"), which helps small/data-limited models. Untying gives more
    capacity and lets the two matrices specialize independently; at scale the
    regularization is no longer needed, the tying constraint costs expressiveness,
    and the extra params are cheap (~2% at 8B) — so big models untie. Rule falls
    out of scale: parameter/data-constrained -> tie; scale to spend -> untie.

18. When batching multiple prompts of different lengths (Module 2's
    territory), padding side — left vs. right — actually affects
    correctness, not just style. Do you already know why, or want to reason
    through it together here?

    Rule: left-pad to generate, right-pad to train — and the why, derived:

    Batched generation produces the next token for all sequences at the same
    position, the last column. Left-padding aligns every sequence's real last
    token in that last column, so one generation step is correct for all at
    once. Right-padding would leave a PAD in the last column for the shorter
    sequences, so the model would generate "after the pads" — anchored to the
    wrong position — which is the correctness bug.

    Two things have to be right for left-pad to work:
    - Attention mask must exclude the left pad tokens. (Note the asymmetry: with
      RIGHT padding the trailing pads are auto-ignored by causal masking — a
      real token at position i only attends to <= i, never to pads after it — so
      right-pad needs no extra masking to keep the real tokens clean. That's why
      right-pad is fine for training/scoring, where you read logits at known
      positions. Left-pad's pads come BEFORE the real tokens, so they must be
      masked explicitly.)
    - RoPE position ids must be shifted. If you number positions 0,1,2... from
      the leftmost pad, real tokens get offset by the pad count, and sequences
      with different pad counts put identical content at different positions —
      inconsistent. Position ids must be computed from the mask so position 0 is
      the first REAL token, so the same content always lands at the same
      positions regardless of how many pads precede it.

    (Module 2 territory; derived here from causal masking + generate-at-last-
    position + RoPE-depends-on-position.)

    Author's follow-up (real-engine answer, forward-note to FlashAttention /
    continuous-batching modules): production engines mostly don't pad at all —
    they PACK variable-length sequences into one contiguous buffer (no pad
    tokens) and use FlashAttention's varlen path with a `cu_seqlens` array
    marking each sequence's boundaries. Wins: no wasted compute/memory on pads
    (big when lengths vary a lot), and the padding-side trap disappears. The
    same two correctness requirements still apply, enforced differently: (1) no
    cross-sequence attention — `cu_seqlens` gives the kernel a block-diagonal
    structure that replaces the pad mask; (2) per-sequence RoPE positions —
    each packed sequence's position ids reset to 0 at its own start. So padding
    is the naive baseline; packing/varlen is the optimization.

19. Speculative decoding (Module 6) has a hard requirement: draft and target
    models must share the exact same tokenizer and vocabulary, because the
    algorithm compares probability distributions over identical token IDs —
    mismatch collapses the acceptance rate toward zero. Does that settle how
    you'd pick a draft model for Qwen3-0.6B, or is it still open?

    Why mismatch collapses acceptance (part 1): the target verifies the draft's
    proposed tokens by comparing probability distributions over token IDs. If
    the two tokenizers differ, token id 5000 means a different string to each
    model, so the comparison is meaningless and acceptance goes to ~0. Sharing
    the tokenizer is what makes "the target's probability for the token the
    draft proposed" a well-defined quantity.

    The Qwen3-0.6B twist + resolution (author's insight, then grounded): 0.6B is
    already the small model, so a draft would be tiny — and from Q17, the 155.6M
    vocab matrix would *dominate* such a draft, with the LM head over 151k tokens
    as the bottleneck. The fix is to trim the DRAFT's vocab by frequency while
    the TARGET still verifies over the full shared vocab — which keeps the
    output distribution provably unchanged (lossless), because correctness lives
    in the target's full-vocab check, not the draft's proposals. The draft just
    skips rarely-correct low-frequency tokens (most probability mass is in the
    common tokens — same frequency story as Q14 / exercise 4).

    Real, citable family (searched): FR-Spec (frequency-ranked draft vocab, ~75%
    LM-head compute cut, lossless — arxiv 2502.14856), VocabTrim, SlimSpec
    (low-rank draft LM-head — 2605.10453), DynaSpec (context-aware dynamic vocab
    — 2510.13847), NanoSpec (2605.26444). MiniCPM4 ships FR-Spec (2506.07900).
    So for Module 6 this is a concrete, decided-able direction: self/small draft
    with a frequency-trimmed LM head, full-vocab verification on the target.

    Confirmed from the FR-Spec paper (arxiv 2502.14856 / ACL 2025 long.198),
    not guessed — corrects an intuition that trimming works like BPE byte
    fallback (it does not):
    - FR-Spec is TRAINING-FREE: no retraining. It extracts a submatrix (rows for
      the top-m frequent tokens) from the existing frozen full-vocab LM head.
    - The trim is on the LM HEAD (output) ONLY, not the input embedding — the
      draft still reads/embeds the full vocabulary; only its output projection
      is restricted.
    - No decomposition: out-of-subset (rare) tokens are simply never proposed by
      the draft — "not decomposed, approximated, or represented." This is the
      opposite of BPE, which decomposes rare tokens into byte pieces. Here the
      draft omits them and the target produces them whole on rejection.
    - Lossless because the target verifies over the FULL vocab: the draft only
      constrains candidate generation (speedup); verification is mathematically
      unchanged (correctness).
    - Caveat: this is FR-Spec specifically. SlimSpec (arxiv 2605.10453) uses a
      low-rank draft LM head and may involve training — "trim-vocab spec
      decoding" is not monolithic.

    Refinement (author intuited this unprompted; it's a real paper — "Out-of-
    Vocabulary Sampling Boosts Speculative Decoding", arxiv 2506.03206):
    instead of just omitting rare tokens, add ONE aggregate "OOV"/rest token to
    the draft head that carries the combined probability mass of all omitted
    tokens. Then the draft can *predict the rare case itself* — when the OOV
    token fires, it stops drafting and defers to the target (which samples the
    rare token from the full vocab). Lossless via rejection-sampling accounting:
    the draft's rejection probability for the OOV case equals the aggregate mass
    it assigned to non-draft tokens. Why it beats plain FR-Spec: FR-Spec wastes
    a step proposing a wrong frequent token that then gets rejected; the OOV-
    token method knows it's in rare-token territory and defers cleanly, no
    wasted wrong-guess.

20. Different model families use completely different tokenizers, so "1000
    tokens" means a different amount of actual text depending on the model.
    Why does that matter specifically for an inference engine — not just as
    a token-counting curiosity?

    Setup (author): for the same text, a smaller vocab needs more tokens and a
    bigger vocab needs fewer, so token counts for the same answer aren't
    comparable across models with different tokenizers.

    Consequence 1 — throughput benchmarks lie. tokens/sec rewards a fragmented
    tokenizer for inflating the token count. Model A at 100 tok/s with 4
    bytes/token = 400 bytes/s; Model B at 120 tok/s with 2 bytes/token = 240
    bytes/s — so A delivers text ~1.7x faster despite showing *lower* tok/s.
    Two engines/models are only comparable in bytes/sec (or chars/sec), or by
    normalizing tok/s by bytes-per-token. "We hit 120 tok/s vs their 100" can
    mean you're actually slower for the user.

    Consequence 2 — token-denominated budgets. Context window and KV cache are
    counted in tokens, not text. So for the identical document, the model with
    the less efficient tokenizer (more tokens per text) fits *less* of it in the
    same context-window limit (smaller effective context in text terms) and uses
    *more* KV memory. An efficient tokenizer quietly buys more usable context
    and cheaper KV for the same content.

## Set C — harder

21. When a multibyte UTF-8 character (an emoji, a non-Latin script character)
    is split across two token boundaries, streaming generation token-by-token
    can emit a broken partial byte sequence — visible as a "�" replacement
    character mid-stream. This is a real bug that's shipped in more than one
    serving engine. What does a correct streaming decoder need to do
    differently from decoding each new token in isolation?

    [Taught — author didn't have this one; grounded in
    demos/streaming_detokenizer_demo.py, which reproduces both the bug and the
    fix on the real 🫸 -> [9284, 104, 116] tokens.]

    A correct streaming decoder differs in three ways from per-token decode:
    1. It decodes against the ACCUMULATED stream, never a token in isolation —
       all tokens so far, not just the new one.
    2. It has a stability rule for when it's safe to emit: only release text up
       to the last COMPLETE UTF-8 character. If the accumulated bytes end
       mid-character, emit nothing this step and wait.
    3. It holds the incomplete tail between tokens (the leftover bytes that don't
       yet form a full character) plus a record of what's already been emitted,
       so a later token that completes the character emits just the new delta.

    Demo result: naive stream emits '�','�','�' (user sees '���'); correct
    stream buffers on the first two tokens and emits '🫸' once complete.

    Implementations: (a) prefix-decode + stability check (decode the whole
    prefix, emit delta unless it ends incomplete — the demo's approach); (b)
    byte-buffer / incremental UTF-8 decoder (accumulate raw bytes, feed an
    incremental decoder that buffers the partial tail) — what real engines use,
    with byte offsets (vLLM's prefix_offset/read_offset) to avoid re-decoding
    the whole sequence each token. Caveat: the "ends with �" check is a handy
    heuristic but not bulletproof (the model could emit a real �); the robust
    version tests at the byte level (is the trailing byte sequence a complete
    UTF-8 codepoint?), not the string level.

22. GPT-2's tokenizer (and GPT-4's) doesn't run BPE merges on the raw byte
    stream directly — it first splits text into categories (letters, digits,
    punctuation, whitespace) with a regex, and only merges within a category.
    What goes wrong if you skip that pre-split and let BPE merge freely
    across category boundaries?

    Author's core point: you must follow the same process at train and
    inference, and the pre-split keeps "dog" the SAME token regardless of what
    punctuation follows it — a stable representation, better learning.

    The two cost-side consequences without the pre-split (answered together):
    - Vocab bloat: "dog." "dog!" "dog," "dog?" "dog " "dog\"" ... each become
      their own token whenever BPE finds them frequent enough, so one word eats
      dozens of slots. Vocab slots aren't free (Q3/Q17 — each is an embedding-
      table row), so you burn the budget on redundant punctuation-variants
      instead of distinct useful tokens; for a fixed budget, worse coverage and
      more fragmentation elsewhere.
    - Frequency dilution -> worse-trained embeddings (the Q16 link): training
      occurrences of "dog + punctuation" split across all the variants, so each
      ("dog." at N/5, "dog!" at N/20, ...) is rarer and its embedding is
      less-trained — the same undertrained-token mechanism as glitch tokens.
      The model also redundantly relearns that all variants mean "dog," wasting
      capacity.

    So the pre-split concentrates all the signal for "dog" onto one
    well-trained token and lets punctuation be its own token — consistency,
    no vocab explosion, no frequency dilution. [Vocab-bloat and frequency-
    dilution halves answered by the assistant at the author's request; the
    consistency core is the author's.]

23. "Token healing" exists because a prompt can end at a boundary the model
    never actually saw ending there in training — e.g. a prompt ending in
    "New Ent" gets tokenized as-is, even though training data would usually
    continue that into one longer token like "Enterprise". What's the actual
    failure mode this causes at generation time, and how would you fix it?

    [Failure mode corrected + fix taught; the author's fix instinct ("let it
    complete the word, then generate normally") was directionally right but
    too optimistic about the failure.]

    Failure mode: the boundary is off-distribution. In training "Enterprise"
    almost always tokenizes as one token (or Ent+erprise seen together), so the
    model rarely saw a sequence END on "Ent". Freezing "Ent" as the final token
    and asking "what's next?" conditions the model on a boundary it barely
    trained on, so its next-token distribution is skewed. Symptoms: it emits a
    space after "Ent" (starts a new word instead of completing), doubles
    characters, or picks an awkward continuation, instead of smoothly producing
    "Enterprise". The greedy tokenization of the prompt froze an unnatural cut.
    (Same root as the Q12 trailing-space insight.)

    Fix — token healing: (1) back up: drop the last token ("Ent"), possibly to
    the last clean boundary; (2) regenerate constrained so the first generated
    token(s) must start with the removed characters ("Ent"); (3) now the model
    picks the NATURAL token at the boundary — it can emit "Enterprise" as one
    token the way training tokenized it, instead of completing from the awkward
    split. Key point beyond "ask it to complete the word": you remove the frozen
    last token and let the model re-tokenize the boundary itself, constrained to
    match the visible characters. (Guidance and some serving stacks implement
    this.)

24. Tokenizer "fertility" (tokens produced per word) varies enormously by
    language — English measures around 1.2-1.4 tokens/word, some languages
    measure 10-16 tokens/word for equivalent content. Context length and KV
    cache budget are both measured in tokens, not words or characters. What
    does that actually mean for a non-English user hitting the same
    prompt-length limit on the same inference engine?

    Two compounding causes of high fertility (author named the first):
    - Underrepresentation: the language was rare in the tokenizer's corpus, so
      few merges were learned; it falls back toward char/byte level -> more
      tokens.
    - Script bytes-per-char: non-Latin scripts use more UTF-8 bytes per
      character (中 = 3 bytes, emoji = 4, grounded earlier), so even at the byte
      floor one char costs 3-4 tokens. Fertility = underrepresentation x
      bytes-per-char.

    What the non-English user loses (not just "more tokens"):
    - Input: hits the same token limit with far less actual content.
    - Output: the same answer costs more tokens -> more compute/latency, and a
      max_tokens cap cuts them off sooner.
    - Cost: per-token pricing -> they pay more for identical information.
    - So it's an equity issue, not mere inefficiency (the "tokenizer tax" /
      unfairness-between-languages work): same token limit = genuinely less
      capability for their language.

    The fix (author asked) — and the catch: design-time, train a larger, more
    balanced multilingual vocab so those languages earn dedicated tokens (what
    Qwen did for Chinese, per 03_algorithms); or adapt via vocab expansion +
    continued pretraining; or go tokenizer-free / byte-latent (BLT, MEGABYTE —
    04_out_of_scope). The catch: a deployed model's tokenizer is frozen for life
    (Q29), so you CAN'T fix fertility at inference time for an existing model —
    it's a model/tokenizer-DESIGN fix (or requires retraining), not a serving
    fix. For serving a given model the fertility is baked in; the most an engine
    can do is account for it in limits/pricing. That "design fix, not serving
    fix" is itself the inference-engineering takeaway.

25. Qwen3's own model family makes an explicit, size-dependent choice: the
    0.6B/1.7B/4B models tie the input embedding and output LM-head weights
    (one matrix, used twice), while the 8B+ models use two separate
    matrices. *(Verify this against Qwen3-0.6B-Instruct's actual config.json
    before it goes in the chapter — this is a secondary-source claim.)* Why
    would tying make sense at 0.6B specifically but not at 8B?

    VERIFIED across real configs (settles the [verify] note): Qwen3-0.6B
    tie=True (hidden 1024, embed matrix 155.6M ~26% of model); Qwen3-4B
    tie=True (hidden 2560, 389M ~10%); Qwen3-8B tie=False (hidden 4096, 622M
    ~8%). Vocab is fixed at 151936 across all sizes.

    The why is Q17's scaling argument + the regularization-vs-capacity tangent,
    now confirmed. The elegant twist: the embedding matrix gets BIGGER in
    absolute terms as hidden grows (155.6M -> 389M -> 622M, since it's
    vocab x hidden), yet 8B unties despite having the largest matrix — because
    what matters is the FRACTION (26% -> 10% -> 8%). Tie when the matrix is a
    big fraction of the model (0.6B/4B) and regularization helps a small/
    data-limited model; untie when it's a small fraction (8B), the extra params
    are cheap, and the capacity/specialization gain from two independent
    matrices outweighs the regularization tying would give. See Q17 and the
    tie-vs-untie note under Q17 for the full mechanism.

26. Gradient-based jailbreak attacks (GCG and its variants) search for
    adversarial token sequences using gradients through the model's own
    vocabulary — and the resulting attack is tied to that specific
    tokenizer, so a suffix that works against one model's tokenizer often
    doesn't transfer to a model with a different vocab. What does that imply
    about tokenization as part of a model's actual attack surface, rather
    than just an encoding detail?

    Core (author): the vocab is the attack's entry point / search space — the
    model can't be fed anything but tokens, so the attacker optimizes over the
    vocab itself. That makes the tokenizer a security-relevant component, not
    neutral preprocessing: it defines what the attacker can search over.

    Closed models are harder (author) — precise version: GCG is white-box (needs
    gradients), so a closed API blocks *direct* GCG. But transfer attacks
    reintroduce the risk — optimize the suffix on an open surrogate, fire it at
    the closed model — and transfer works much better when the surrogate shares
    the target's tokenizer/vocab. So a closed model on a known/common tokenizer
    is more transferable-attackable; the tokenizer is also the bridge that makes
    cross-model transfer possible.

    Diversity is a weak defense (author): a different tokenizer only stops the
    specific found suffix, not the method — the attacker just re-optimizes.
    Security-through-tokenizer-diversity is friction, not a real mitigation.

    Through-line for the chapter: tokenization is part of the trusted computing
    base, not a neutral front-end. Third security thread after Q6 (special-token
    injection) and Q16 (glitch tokens as exploitable inputs) — here the vocab IS
    the GCG search space.

27. Given a fixed tokenizer (fixed merge rules), is BPE encoding
    deterministic — does the exact same input string always produce the
    exact same token IDs? If yes, where does the "boundary" problem in Q23
    (token healing) actually come from, if not from encoding itself being
    ambiguous?

    Yes, encoding is deterministic: the greedy algorithm applies the fixed
    merges in learned order, no randomness, so the same string always produces
    the same token IDs (grounded: "New Enterprise" -> [3564, 25472] both times).
    Precision on the author's "vice versa is true": the DECODE direction is
    many-to-one (different token sequences can decode to the same text —
    "Ent"+"erprise" and "Enterprise" both read "Enterprise"), but the ENCODER
    never produces those alternatives; it deterministically picks the one greedy
    tokenization. Ambiguity is "multiple valid tokenizations exist," not "the
    encoder is unpredictable."

    Where the boundary problem comes from (author nailed it): BPE is
    deterministic but NOT compositional / not prefix-consistent —
    tokenize(A)+tokenize(B) != tokenize(A+B), and tokenize(prefix) is not a
    prefix of tokenize(whole). Grounded: tokenize("Enterprise") = [85647] (one
    token) but tokenize("Ent")+tokenize("erprise") = [2250, 261, 9671] (three);
    tokenize("New Ent") = [3564, 4863] is NOT a prefix of
    tokenize("New Enterprise") = [3564, 25472]. So freezing the prompt at
    "New Ent" commits to a deterministic-but-unnatural split; the model would
    have preferred [New, Enterprise], but the greedy tokenization of the prefix
    locked in the "Ent" split. Determinism doesn't save you because the
    deterministic tokenization of the prefix != that of the eventual whole —
    which is exactly the deep reason token healing (Q23) exists.

28. Prefix caching (a future chapter) reuses KV cache blocks for requests
    sharing an identical token prefix — reuse is exact-match at the token
    level, not the text level. What's a realistic way a system-prompt
    template could silently break prefix-cache hit rate, purely through how
    it tokenizes?

    Author's answer: a varying field at the START of the system prompt, like an
    appended time or date.

    Why early = catastrophic: matching is from token 0, so a varying token at
    position k makes everything from k onward a miss. At the start (k ~ 2), the
    ENTIRE system prompt misses every request — you recompute the whole (often
    long) system-prompt KV every time. "Today is 2026-07-24. You are a helpful
    assistant. [500 fixed tokens]" gets ~0% prefix reuse, silently, no error —
    same silent-degradation theme as Q7/Q12.

    Fix — order by volatility: fixed content first, dynamic last. "You are a
    helpful assistant. [500 fixed tokens] ... Today is 2026-07-24." Now the long
    fixed prefix is identical across requests (cached/reused) and only the short
    tail after the date is recomputed. Rule: most-stable content at the front,
    most-variable at the back.

    Subtle Q27 angle: because BPE isn't compositional, the fixed token right
    after the variable field can re-tokenize at the seam depending on what the
    variable field ended with (trailing space, digit, punctuation). So even
    careful placement needs a clean, stable delimiter between dynamic and static
    text so the seam tokenizes consistently.

29. A tokenizer's merge vocabulary is learned once, from one training corpus,
    then frozen for the model's entire life. What does it concretely mean
    for a tokenizer to be a bad fit for a domain — e.g. feeding a lot of
    source code or chemistry notation through a tokenizer trained mostly on
    English prose?

    Author: it's the domain version of Q24 — out-of-domain content, even English
    (code, chemistry, math), was underrepresented in the tokenizer's corpus, so
    it fragments like an underrepresented language does.

    "Bad fit" concretely = low bytes-per-token (high fertility) for that domain,
    the same metric as Q24. Grounded on Qwen3-0.6B: English prose 5.33
    bytes/token; Python code 2.40; LaTeX math 1.83; chemistry (aspirin SMILES)
    1.26; chemistry formula "C6H12O6 + 6O2 -> ..." 1.22 — near the byte floor,
    ~4x more tokens than prose for the same content, because there are almost no
    useful merges for those strings.

    The real lesson (honest wrinkle): code is only MODERATE (2.40), not terrible,
    because Qwen was trained on a lot of code and earned code tokens. So "bad
    fit" isn't about a domain being exotic — it's about whether it was in THIS
    tokenizer's corpus. Code in-corpus -> okay; chemistry SMILES not -> near
    byte-floor.

    Consequences (= Q24): domain-heavy prompts cost multiples more tokens ->
    shorter effective context, more compute/KV/cost, model reconstructs meaning
    from near-byte fragments. Can't fix for a deployed model (frozen tokenizer,
    Q24/Q10) — the fix is a domain-aware tokenizer at design time (code models
    add indentation/keyword tokens; chem models add SMILES-aware tokens) +
    retraining.

30. Is tokenization ever actually the bottleneck in a serving engine, or is
    it always negligible next to the GPU forward pass? Under what realistic
    condition — request volume, generation length, tokenizer implementation
    — could it stop being negligible?

    Author's core (sophisticated): decode is normally the heavy part and is
    memory-bandwidth-bound (streaming the weights per token) — which is exactly
    why tokenization usually hides in the noise (the GPU is busy, the tiny CPU
    tokenize overlaps it). It stops being negligible when there's little decode
    to hide behind: prefill-heavy + short generation + small model. Then the
    fixed tokenization cost becomes visible.

    Parallelization (author was unsure): mostly yes, but the two sides differ.
    - Input tokenization parallelizes well: Rust fast tokenizers thread batch
      encoding across requests, and within a sequence the regex pre-split (Q22)
      chops text into independent chunks tokenizable in parallel (BPE merges are
      sequential only within a short chunk).
    - Output detokenization is harder: within a stream there's a sequential
      dependency (the Q21 incremental UTF-8 buffer), and it's per-token,
      per-request. Across streams it parallelizes, but the real cost is
      Python/GIL overhead per token per stream — so engines batch it or push it
      off the critical path (separate threads/processes).

    Input-vs-output takeaway: input tokenize is once-per-request and Rust-fast,
    so it bottlenecks only in the prefill-heavy/short-gen/small-model case
    (author's answer). Output detokenize is per-token-per-stream with Python
    overhead, so it's the MORE COMMON real-world bottleneck — at high-concurrency
    streaming (Q9). Two different conditions for the two sides.

## Sources (for citation when the chapter is written)

- [Karpathy: Let's build the GPT tokenizer](https://simonwillison.net/2024/Feb/20/lets-build-the-gpt-tokenizer/) — Q11, Q12, Q13, Q22
- [SolidGoldMagikarp and other glitch tokens](https://www.kith.org/words/2023/12/10/solidgoldmagikarp-and-other-glitch-tokens/) — Q16
- [Fishing for Magikarp: detecting under-trained tokens](https://arxiv.org/abs/2405.05417) — Q16
- [vLLM issue #7252 — draft/target vocab mismatch](https://github.com/vllm-project/vllm/issues/7252) — Q19
- [Speculative decoding in production: hidden traps](https://tianpan.co/blog/2026-04-17-speculative-decoding-production-hidden-traps) — Q19
- [UTF-8 Plumbing: Byte-level Tokenizers Unavoidably Enable LLMs to Generate Ill-formed UTF-8](https://openreview.net/forum?id=8ExXncFpf6) — Q21
- [llama.cpp issue #8691 — tokenizer not working on partial UTF-8 bytes](https://github.com/ggml-org/llama.cpp/issues/8691) — Q21
- [Guidance docs: token healing](https://guidance.readthedocs.io/en/latest/example_notebooks/tutorials/token_healing.html) — Q23, Q27
- [The Art of Prompt Design: Prompt Boundaries and Token Healing](https://medium.com/data-science/the-art-of-prompt-design-prompt-boundaries-and-token-healing-3b2448b0be38) — Q23, Q27
- [Language Model Tokenizers Introduce Unfairness Between Languages](https://arxiv.org/pdf/2305.15425) — Q24
- [The Tokenizer Tax Across 24 European Languages](https://arxiv.org/html/2605.24718) — Q24
- [Weight tying in language models — when and why LLMs share embeddings](https://medium.com/@vishal09vns/weight-tying-in-language-models-when-and-why-llms-share-embeddings-7f5f5376c625) — Q25
- [Qwen/Qwen3-Embedding-8B discussion — tie_word_embeddings: false](https://huggingface.co/Qwen/Qwen3-Embedding-8B/discussions/17) — Q25
- [What Is the GCG Attack? — FutureAGI Guide](https://futureagi.com/glossary/gcg-attack/) — Q26
