"""
Exercise 3: digit splitting.

Question 14 claimed Qwen's tokenizer deliberately splits numbers into
individual digits rather than merging them like ordinary words. Confirm it,
and compare against a plain word of similar length.

Run: python 03_digit_splitting.py
"""
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"


def show(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    pieces = tokenizer.convert_ids_to_tokens(ids)
    print(f"{text!r:>10} -> ids={ids} pieces={pieces}")


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    samples = ["12345", "hello", "9", "99999999", "3.14159"]

    # TODO: call show() for each sample in `samples`
    ...

    print()
    print("'12345' is 5 characters and 'hello' is 5 characters. Do they")
    print("produce the same number of tokens? What does that tell you about")
    print("why a tokenizer designer would choose digit-splitting on purpose?")


if __name__ == "__main__":
    main()
