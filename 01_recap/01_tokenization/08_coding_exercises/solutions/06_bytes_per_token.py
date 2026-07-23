"""Solution to exercise 6. Verified output (Qwen/Qwen3-0.6B):

 english_prose: 4.40 bytes/token
   python_code: 2.67 bytes/token
 random_digits: 1.00 bytes/token

random_digits lands at exactly 1.00 because digit-splitting (exercise 3)
means every single digit is its own token -- one byte in, one token out,
every time. English prose compresses best because common whole words and
subwords got merged during BPE training; code sits in between because
indentation and symbols split more often than prose but less than digits.
"""
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"


def bytes_per_token(tokenizer, text):
    n_bytes = len(text.encode("utf-8"))
    n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    return n_bytes / n_tokens


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
