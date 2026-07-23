"""Solution to exercise 5. Verified output (Qwen/Qwen3-0.6B):

tied (same storage): True
tied matrix: 155,582,464 params
total model: 596,049,920 params
fraction:    26.1%

Matches 03_algorithms_and_qwen3.md's worked-out estimate almost exactly --
that file estimated ~26% from config.json alone; loading the real model and
checking data_ptr() identity confirms it directly, not just implies it.
"""
import torch
from transformers import AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3-0.6B"


def main():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16)

    input_emb = model.get_input_embeddings().weight
    output_emb = model.get_output_embeddings().weight

    same_storage = input_emb.data_ptr() == output_emb.data_ptr()
    print(f"tied (same storage): {same_storage}")

    vocab_size, hidden_size = input_emb.shape
    n_tied_params = vocab_size * hidden_size
    n_total_params = sum(p.numel() for p in model.parameters())
    fraction = n_tied_params / n_total_params

    print(f"tied matrix: {n_tied_params:,} params")
    print(f"total model: {n_total_params:,} params")
    print(f"fraction:    {fraction:.1%}")


if __name__ == "__main__":
    main()
