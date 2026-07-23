"""
Exercise 6: bytes-per-token, the actual efficiency metric.

A tokenizer's efficiency is measured in bytes-per-token: how many raw UTF-8
bytes, on average, get compressed into a single token. Implement it, then
run it over three very different kinds of text and see how differently
they compress.

Run: python 06_bytes_per_token.py
"""
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"


def bytes_per_token(tokenizer, text):
    # TODO: return len(text encoded as utf-8 bytes) / len(token ids)
    ...


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    samples = {
        "english_prose": "The quick brown fox jumps over the lazy dog.",
        "python_code": "def foo(x, y):\n    return x + y\n",
        "random_digits": "48291058310293845723",
    }

    for name, text in samples.items():
        ratio = bytes_per_token(tokenizer, text)
        print(f"{name:>15}: {ratio:.2f} bytes/token")

    print()
    print("random_digits should land near 1.00 -- why? (hint: exercise 3)")


if __name__ == "__main__":
    main()
