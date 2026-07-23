"""Solution to exercise 5. Verified output (Qwen/Qwen3-0.6B):

tokenizer.vocab_size:     151643
len(tokenizer):           151669
model config.vocab_size:  151936
estimated merge operations: 151387

the embedding matrix has 267 rows that the tokenizer never actually
produces an ID for.

Three different numbers, three different meanings:
- tokenizer.vocab_size (151643) is the base BPE vocabulary alone: 256 raw
  bytes plus every learned merge, nothing else. This is the right number
  for merge-count arithmetic: 151643 - 256 = 151387 merges were performed
  during training.
- len(tokenizer) (151669) adds the 26 actually-used special/added tokens
  (`<|im_end|>`, `<|endoftext|>`, the vision placeholders, and so on) on
  top of the base vocabulary.
- model config.vocab_size (151936) is the embedding matrix's actual row
  count, which is larger still -- 267 rows bigger than the tokenizer ever
  uses. Model builders commonly round the embedding size up to a multiple
  that's efficient for GPU matrix multiplication (e.g. a multiple of 64 or
  128), reserving unused rows rather than shipping an odd-sized matrix.
"""
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
BASE_BYTES = 256
MODEL_CONFIG_VOCAB_SIZE = 151936


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    bpe_vocab_size = tokenizer.vocab_size
    tokenizer_len = len(tokenizer)

    print(f"tokenizer.vocab_size:     {bpe_vocab_size}")
    print(f"len(tokenizer):           {tokenizer_len}")
    print(f"model config.vocab_size:  {MODEL_CONFIG_VOCAB_SIZE}")

    n_merges = bpe_vocab_size - BASE_BYTES
    print(f"estimated merge operations: {n_merges}")

    print()
    unused_rows = MODEL_CONFIG_VOCAB_SIZE - tokenizer_len
    print(f"the embedding matrix has {unused_rows} rows the tokenizer never")
    print("actually produces an ID for. Why would a model ship an embedding")
    print("table bigger than its own vocabulary?")


if __name__ == "__main__":
    main()
