"""
Exercise 5: how many merges did Qwen3's tokenizer actually learn?

Question 6 asks for the arithmetic linking vocab size to merge count. It
turns out there isn't just one "vocab size" for this model -- there are
three different numbers, and they don't agree. Find all three on the real
tokenizer and reconcile them.

Run: python 05_merge_count_arithmetic.py
"""
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
BASE_BYTES = 256  # byte-level BPE's starting vocabulary
MODEL_CONFIG_VOCAB_SIZE = 151936  # from ../03_algorithms_and_qwen3.md


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # TODO: find two more numbers from the tokenizer object itself:
    #  - tokenizer.vocab_size    (the base BPE vocab: bytes + merges, no added specials)
    #  - len(tokenizer)          (including added/special tokens)
    bpe_vocab_size = None
    tokenizer_len = None

    print(f"tokenizer.vocab_size:     {bpe_vocab_size}")
    print(f"len(tokenizer):           {tokenizer_len}")
    print(f"model config.vocab_size:  {MODEL_CONFIG_VOCAB_SIZE}")

    # TODO: using the RIGHT one of these three numbers (think about which
    # one actually reflects "base vocab + learned merges, nothing else"),
    # estimate how many merge operations BPE training performed.
    n_merges = None
    print(f"estimated merge operations: {n_merges}")

    print()
    unused_rows = MODEL_CONFIG_VOCAB_SIZE - tokenizer_len
    print(f"the embedding matrix has {unused_rows} rows the tokenizer never")
    print("actually produces an ID for. Why would a model ship an embedding")
    print("table bigger than its own vocabulary?")


if __name__ == "__main__":
    main()
