"""
Demo: the LLM's input embedding table on real Qwen3-0.6B.

Shows: (1) the embedding table is a lookup (vocab x hidden); (2) geometry =
similarity (related tokens have higher cosine similarity than unrelated ones);
(3) the table is TIED (input embedding == LM head); (4) the LM-head cost
(vocab x hidden matmul + softmax every step). Bonus: analogy arithmetic.

Runs on CPU/MPS, no CUDA. Downloads Qwen3-0.6B weights (~1.2 GB) once.

Run: python embedding_demo.py
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
model.eval()

E = model.get_input_embeddings().weight  # [vocab, hidden]
print(f"embedding table shape (vocab x hidden): {tuple(E.shape)}  = {E.numel():,} params")

# (3) tied? input embedding vs LM head weight
lm_head = model.get_output_embeddings().weight
print(f"tied (input embedding is the LM head)? {E.data_ptr() == lm_head.data_ptr()}")
print(f"LM-head shape: {tuple(lm_head.shape)}  -> output logits over {lm_head.shape[0]:,} tokens every step")

def emb(word):
    ids = tok.encode(word, add_special_tokens=False)
    return E[ids].mean(0)  # avg if multi-token, so we compare whole words

def cos(a, b):
    return F.cosine_similarity(a, b, dim=0).item()

print("\n(2) geometry = similarity (cosine of input embeddings):")
cat, dog, tue, king = emb(" cat"), emb(" dog"), emb(" Tuesday"), emb(" king")
print(f"  cos( cat,  dog)     = {cos(cat, dog):.3f}   (related -> higher)")
print(f"  cos( cat,  Tuesday) = {cos(cat, tue):.3f}   (unrelated -> lower)")
print(f"  cos( cat,  king)    = {cos(cat, king):.3f}")

# bonus: analogy king - man + woman ~ queen (nearest token by cosine)
print("\n(bonus) analogy king - man + woman -> nearest tokens:")
v = emb(" king") - emb(" man") + emb(" woman")
sims = F.cosine_similarity(v.unsqueeze(0), E, dim=1)
topk = sims.topk(5).indices.tolist()
print("  nearest:", [repr(tok.decode([i])) for i in topk])
