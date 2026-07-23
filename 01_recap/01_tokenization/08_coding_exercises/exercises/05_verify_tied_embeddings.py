"""
Exercise 5: verify tied embeddings for real.

03_algorithms_and_qwen3.md claims Qwen3-0.6B ties its input embedding and
output LM head weights, and that the tied matrix is ~26% of the model's
total parameters. Don't take config.json's word for it -- load the actual
model and check whether the two weight tensors are literally the same
object in memory, then recompute the percentage yourself.

This downloads ~1.2GB of model weights the first time it runs.

Run: python 05_verify_tied_embeddings.py
"""
import torch
from transformers import AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3-0.6B"


def main():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16)

    input_emb = model.get_input_embeddings().weight
    output_emb = model.get_output_embeddings().weight

    # TODO: check whether input_emb and output_emb share the same underlying
    # storage (not just equal values -- the same memory). Hint: .data_ptr()
    same_storage = None
    print(f"tied (same storage): {same_storage}")

    vocab_size, hidden_size = input_emb.shape

    # TODO: compute the number of parameters in the tied matrix, the total
    # number of parameters in the whole model, and the fraction the tied
    # matrix accounts for.
    n_tied_params = None
    n_total_params = None
    fraction = None

    print(f"tied matrix: {n_tied_params:,} params")
    print(f"total model: {n_total_params:,} params")
    print(f"fraction:    {fraction:.1%}")


if __name__ == "__main__":
    main()
