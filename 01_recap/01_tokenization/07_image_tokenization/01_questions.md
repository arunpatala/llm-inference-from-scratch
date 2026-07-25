# Image tokenization — questions

Answer inline under each question, same as the other question files. Skip
or write "not sure" where there's no real take yet. This picks up the
"different mechanism entirely" thread from `../04_out_of_scope.md`.

1. Text tokenization looks up a discrete ID in a fixed vocabulary. In the
   dominant approach used by models like LLaVA and Qwen-VL, is an image
   patch mapped to a discrete ID at all, or is it something else entirely?

   Not a discrete ID. A vision patch is encoded to a continuous embedding
   directly by the vision encoder — it skips the discrete ID and the vocabulary
   lookup that text uses. So the two paths differ: text is ID -> (lookup in a
   fixed table) -> embedding; image is patch -> (vision encoder) -> embedding
   directly. They converge at the EMBEDDING level, not the ID level (there is no
   "image vocabulary" in this approach).

   The bridge: the encoder's output lives in the vision model's space, not the
   LLM's, so an adapter — the "projector"/connector (LLaVA's contribution) — maps
   it into the LLM's token-embedding space. It fixes two things: (1) dimension
   (e.g. CLIP-ViT-L 1024-d -> LLM 4096-d), mechanical; (2) semantic alignment
   (place the image vector where the LLM's text-trained geometry expects that
   meaning), learned from image-text pairs. It's usually the smallest, cheapest
   trainable part — early LLaVA froze the vision encoder AND the LLM and trained
   ONLY this small MLP for alignment. [connects to Q3.]

2. Walk through the actual pipeline for a typical vision-language model:
   image goes in, vision tokens come out the other end, ready to be fed into
   the LLM alongside text tokens. What are the stages in between?

   Full ordered pipeline (author's skeleton + the two ends filled in):
   1. Preprocess: resize the raw image to a fixed resolution, cut into
      fixed-size patches (e.g. 14x14 px), flatten each patch's RGB and linearly
      embed it.
   2. Vision encoder (ViT): self-attention over ALL patches at once, so each
      patch embedding is CONTEXTUALIZED — it reflects the whole image, not just
      its own pixels (unlike text, where the tokenizer treats each token in
      isolation before the model sees it). [author's "each patch through
      encoder" corrected: patches are encoded jointly, not independently.]
   3. Projector (MLP): map the encoder's vectors into the LLM's token-embedding
      space (dimension + semantic alignment — Q1/Q3).
   4. Splice at placeholders (positioning — this is Q6): the text prompt holds a
      run of reserved placeholder tokens (LLaVA `<image>`; Qwen
      `<|vision_start|>...<|image_pad|>...<|vision_end|>`). They tokenize to
      ordinary IDs and get throwaway placeholder embeddings; then, before the
      transformer, those slots are OVERWRITTEN with the real projected image
      embeddings (1:1: N placeholders <-> N patch embeddings). Final sequence:
      [text embeds][image embeds at placeholder positions][text embeds], all
      continuous vectors in one space, attended jointly.

   Why placeholders are the elegant trick: they reserve the POSITION (image
   lands where the marker sat) and the COUNT (the processor expands the
   placeholder to N copies, N = image-token count, which depends on resolution —
   Q4). The tokenizer, chat template, and attention machinery all treat the
   image as "just a run of special tokens" whose embeddings get swapped for real
   image vectors — nothing else in the stack needs to know images exist.

   One-liner: the image doesn't enter as tokens, it enters as embeddings that
   hijack the slots of reserved placeholder tokens. [This answers Q6.]

3. LLaVA's CLIP-ViT-L/14 encoder produces 576 patch tokens plus 1 CLS token
   per image, all continuous vectors, mapped into the LLM's embedding space
   by a 2-layer MLP. There's no discrete "image vocabulary" anywhere in
   that pipeline. So what does "token" actually mean here? Is it the same
   concept as a text BPE token, or something looser?

   [Answered across Q1 and Q10: "token" here is LOOSER — a continuous vector
   occupying a slot in the sequence, not a discrete ID from a vocabulary. The
   576 "tokens" are the 576 patch embeddings (post-2-layer-MLP projector);
   there's no image vocabulary. Same word, different concept — see Q10's
   "vectorizer not tokenizer" capstone.]

4. Qwen2.5-VL computes image token count as H×W / (14×14×4): an 896×896
   image becomes 1024 tokens. Qwen3-VL uses a different patch size (16×16,
   same 4x pooling), giving 784 tokens for the same image. Why would model
   designers change the patch size and pooling factor between versions, and
   what's the actual trade-off they're making?

   [Answered via Q5: patch size + pooling set the image-token count, which is
   the dominant O(n^2) serving cost. Bigger patches / more pooling -> fewer
   tokens -> cheaper + faster + more batch, but coarser detail (fewer, larger
   patches lose fine spatial resolution). Smaller patches -> more tokens -> finer
   detail but quadratic cost. So the version-to-version patch/pooling change is
   the resolution-vs-cost knob (896x896: 1024 tokens at 14px vs 784 at 16px) —
   the same detail-vs-token-budget trade-off as Q5, tuned per model. The design
   axis is exactly the Future-Directions S5 granularity knob, fixed at
   design time here.]

5. A single mid-sized image can cost hundreds to over a thousand tokens,
   more than most entire text prompts we've discussed for Qwen3-0.6B.
   Given that attention is O(n²) and KV cache scales linearly with sequence
   length, what does that imply about serving a vision-language model
   versus a text-only one?

   Foundation (author): image tokens are just tokens — no special KV treatment;
   more tokens = more KV + compute, like text. Correct base, but "just like
   text" undersells it three ways:
   - Magnitude flips the profile: one image ~1000 tokens ~ a long text prompt,
     so a VLM "prompt" is mostly image; multi-image / video (frames x ~1000)
     explodes the sequence.
   - O(n^2) means the image dominates, not adds: a 50-token prompt costs ~2,500;
     add a 1000-token image -> ~1050^2 ~ 1.1M, ~440x more. The quadratic term is
     entirely image tokens.
   - It inverts the Q30 bottleneck: text decode is memory-bandwidth-bound, but
     images arrive all at once as a giant PREFILL, so VLM inference is
     prefill-dominated and compute-bound; and ~1000 tokens of KV per image means
     far fewer concurrent requests fit -> smaller batches -> lower throughput.

   Punchline: because visual tokens are the dominant cost, REDUCING them is the
   single highest-leverage VLM optimization — visual token pruning / pooling /
   resampling (Qwen pools patches; Perceiver-style resamplers compress to a fixed
   budget; OccamToken 2,880 -> ~40 tokens keeping >93% accuracy). It's the
   Future-Directions S5 runtime-granularity idea, but vision is where it bites
   hardest: you can drop most image tokens and barely lose accuracy, which is
   never true for text.

6. Qwen3-0.6B's tokenizer reserves `<|image_pad|>` in its vocab despite
   being text-only. In an actual vision-language model, what is that
   placeholder token's job in the sequence, given that the real image
   content isn't a token ID at all? Where do the actual image embeddings
   get spliced in?

   [Answered together with Q2 above — the placeholder splice. `<|image_pad|>` is
   a reserved token whose throwaway embedding gets overwritten by a real
   projected patch embedding; N pad tokens <-> N patches; Qwen reserves it in the
   shared-family vocab even though 0.6B is text-only.]

7. Some models (VQ-VAE/VQ-GAN-based) use genuinely discrete image tokens: a
   real integer ID into a learned codebook, exactly like text BPE. Why
   would a model need that, when continuous ViT-patch embeddings are
   simpler and already work for image understanding?

   For GENERATION (author got it). To generate an image autoregressively the
   model predicts "the next image token", and to predict a token you need a
   fixed set to pick from (a codebook) so you can put a probability distribution
   over it and sample — exactly like next-text-token over the vocabulary. A
   discrete codebook (e.g. 8192 entries) gives softmax -> sample; a continuous
   vector has no vocabulary to sample from, so plain AR generation doesn't work.
   So the clean split: continuous ViT patches -> image UNDERSTANDING (image in,
   text out, only need to read it); discrete VQ tokens -> image GENERATION
   (predict + sample image tokens). Unified any-to-any models need discrete.

   The author's other guess, prefix caching, is a real minor side-benefit:
   discrete IDs exact-match so they're cacheable like text (Q28), while
   continuous embeddings don't exact-match — but that's a bonus, not the driver.

   This is exactly the frontier debate (survey / Future Directions): "does
   understanding want continuous while generation wants discrete?" (Selftok, the
   discrete-vs-continuous inversion) — an open question the author re-derived.

8. RoPE was designed around a 1D sequence position: token index 0, 1, 2,
   and so on. Qwen2-VL's M-RoPE splits position into three separate
   components for text, image, and video instead. Why can't a single 1D
   position index describe where a patch sits in an image, the way it can
   for a word in a sentence?

   The 2D spatial relationship — which patch is above/below another — is lost in
   1D RoPE (author's core point). Precisely: flatten a grid row by row and two
   vertically-adjacent patches (same column, adjacent rows) end up W positions
   apart (W = grid width), while side-by-side patches are 1 apart. So 1D makes
   vertical neighbors look far away though they physically touch; the flattening
   ordering doesn't match 2D spatial distance.

   Fix — M-RoPE (Multimodal RoPE): give each token a position with SEPARATE
   components (temporal, height, width) instead of one number. Text: all three
   move together (genuinely 1D). Image: the height and width components encode
   the patch's (row, col), so vertical neighbors are 1 apart in height, horizontal
   in width — spatial adjacency preserved. Video: the temporal component adds the
   frame index (the 3rd dimension). So position becomes 2D (3D for video) so each
   patch carries its real grid location.

9. Continuous (ViT-patch) and discrete (VQ-VAE) image tokenizers each lose
   something different: continuous compression is described as diluting
   high-level semantics like object identity, while discrete quantization
   loses fine texture detail. Does that trade-off remind you of anything
   from text tokenization (character vs. word vs. subword), or is it a
   genuinely different kind of trade-off?

   Author's instinct: prefer continuous; if you need detail, tokenize at higher
   resolution; doesn't see when discrete wins performance-wise. Right for
   UNDERSTANDING — discrete underperforms continuous on multimodal understanding
   benchmarks due to the quantization bottleneck. But "when is discrete better"
   isn't about understanding accuracy:
   - Discrete is REQUIRED for generation (Q7) — no codebook, no AR image
     sampling. Its advantage is enabling drawing, not better reading. A unified
     read+draw model faces the tension (Selftok debate).
   - Discrete buys engineering wins: one unified vocab/AR objective for
     text+image, and cacheability (exact-match) — not understanding accuracy.
   - "Just use higher resolution" hits Q5: more patches -> more tokens ->
     O(n^2) explosion, so resolution can't scale freely; that ceiling is why
     people pool/prune/resample visual tokens.

   The Q9 connection: discrete-quantization-loses-texture is the same categorical
   bottleneck as the Geometric Alignment Tax (Future Directions S6) — forcing
   continuous data into discrete buckets throws away the in-between — and it
   rhymes with strawberry/Q27 (a discrete token hides its fine internal
   structure). So discrete image tokens ~ word/subword tokens (fixed vocab,
   hides detail); continuous patches ~ the raw signal. A real rhyme, not a clean
   1:1 map.

10. Given everything above, is "vision tokenizer" actually the right name
    for what a ViT patch encoder does, in the same sense that a BPE
    tokenizer is a tokenizer for text, or is it doing something different
    enough that sharing the name is a little misleading?

    Author's answer: "vision vectorizer" — it produces vectors, not tokens.
    Exactly right. For the continuous ViT case, "tokenizer" is misleading:
    - Type mismatch: a text tokenizer outputs discrete IDs from a vocabulary
      (meaningless row numbers); a ViT outputs continuous, meaning-laden vectors.
    - The ViT collapses three jobs text keeps separate: tokenize (segment into
      patches) + embed (patch -> vector, skipping the ID step, Q1) + contextualize
      (attention across patches, Q2 — which the text embedding table never does).
      So "tokenizer" undersells it; it's tokenizer + embedding table + first
      transformer layers in one. "Vectorizer"/"encoder" is more honest.
    But the answer flips for discrete VQ: a VQ-VAE tokenizer IS a real tokenizer
    in the text sense — integer IDs from a fixed codebook, exactly like BPE (Q7),
    so the name fits there.

    Capstone: "vision tokenizer" is a misnomer for the dominant continuous
    approach (really a vectorizer/encoder) and accurate only for discrete VQ. The
    word "token" survives across both only in the loose sense of "a unit in the
    sequence", not the strict text sense of "a discrete ID from a vocabulary".

## Sources (for citation when the chapter is written)

- [Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/pdf/2409.12191) — Q4, Q8
- [Qwen2-VL Hugging Face model card](https://huggingface.co/docs/transformers/en/model_doc/qwen2_vl) — Q4
- [LLaVA-Mini: Efficient Image and Video Large Multimodal Models with One Vision Token](https://arxiv.org/html/2501.03895v1) — Q3, Q5
- [Compression Tells Intelligence: Visual Coding, Visual Token Technology, and the Unification](https://arxiv.org/pdf/2601.20742) — Q1, Q7, Q9
- [Qwen2-VL's RoPE Variant — M-RoPE](https://medium.com/everyday-ai/qwen2-vls-rope-variant-m-rope-8cfcc4672ea9) — Q8
