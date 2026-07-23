# Image tokenization — questions

Answer inline under each question, same as the other question files. Skip
or write "not sure" where there's no real take yet. This picks up the
"different mechanism entirely" thread from `../04_out_of_scope.md`.

1. Text tokenization looks up a discrete ID in a fixed vocabulary. In the
   dominant approach used by models like LLaVA and Qwen-VL, is an image
   patch mapped to a discrete ID at all, or is it something else entirely?

   _(answer here)_

2. Walk through the actual pipeline for a typical vision-language model:
   image goes in, vision tokens come out the other end, ready to be fed into
   the LLM alongside text tokens. What are the stages in between?

   _(answer here)_

3. LLaVA's CLIP-ViT-L/14 encoder produces 576 patch tokens plus 1 CLS token
   per image, all continuous vectors, mapped into the LLM's embedding space
   by a 2-layer MLP. There's no discrete "image vocabulary" anywhere in
   that pipeline. So what does "token" actually mean here? Is it the same
   concept as a text BPE token, or something looser?

   _(answer here)_

4. Qwen2.5-VL computes image token count as H×W / (14×14×4): an 896×896
   image becomes 1024 tokens. Qwen3-VL uses a different patch size (16×16,
   same 4x pooling), giving 784 tokens for the same image. Why would model
   designers change the patch size and pooling factor between versions, and
   what's the actual trade-off they're making?

   _(answer here)_

5. A single mid-sized image can cost hundreds to over a thousand tokens,
   more than most entire text prompts we've discussed for Qwen3-0.6B.
   Given that attention is O(n²) and KV cache scales linearly with sequence
   length, what does that imply about serving a vision-language model
   versus a text-only one?

   _(answer here)_

6. Qwen3-0.6B's tokenizer reserves `<|image_pad|>` in its vocab despite
   being text-only. In an actual vision-language model, what is that
   placeholder token's job in the sequence, given that the real image
   content isn't a token ID at all? Where do the actual image embeddings
   get spliced in?

   _(answer here)_

7. Some models (VQ-VAE/VQ-GAN-based) use genuinely discrete image tokens: a
   real integer ID into a learned codebook, exactly like text BPE. Why
   would a model need that, when continuous ViT-patch embeddings are
   simpler and already work for image understanding?

   _(answer here)_

8. RoPE was designed around a 1D sequence position: token index 0, 1, 2,
   and so on. Qwen2-VL's M-RoPE splits position into three separate
   components for text, image, and video instead. Why can't a single 1D
   position index describe where a patch sits in an image, the way it can
   for a word in a sentence?

   _(answer here)_

9. Continuous (ViT-patch) and discrete (VQ-VAE) image tokenizers each lose
   something different: continuous compression is described as diluting
   high-level semantics like object identity, while discrete quantization
   loses fine texture detail. Does that trade-off remind you of anything
   from text tokenization (character vs. word vs. subword), or is it a
   genuinely different kind of trade-off?

   _(answer here)_

10. Given everything above, is "vision tokenizer" actually the right name
    for what a ViT patch encoder does, in the same sense that a BPE
    tokenizer is a tokenizer for text, or is it doing something different
    enough that sharing the name is a little misleading?

    _(answer here)_

## Sources (for citation when the chapter is written)

- [Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/pdf/2409.12191) — Q4, Q8
- [Qwen2-VL Hugging Face model card](https://huggingface.co/docs/transformers/en/model_doc/qwen2_vl) — Q4
- [LLaVA-Mini: Efficient Image and Video Large Multimodal Models with One Vision Token](https://arxiv.org/html/2501.03895v1) — Q3, Q5
- [Compression Tells Intelligence: Visual Coding, Visual Token Technology, and the Unification](https://arxiv.org/pdf/2601.20742) — Q1, Q7, Q9
- [Qwen2-VL's RoPE Variant — M-RoPE](https://medium.com/everyday-ai/qwen2-vls-rope-variant-m-rope-8cfcc4672ea9) — Q8
