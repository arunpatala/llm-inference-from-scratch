"""
Exercise 2: leading space and case sensitivity.

Confirms questions 12 and 13 from ../01_questions.md with real token IDs,
not just a claim. "hello", " hello", "Hello", and " Hello" -- how many of
these four actually produce the same token ID?

Run: python 02_leading_space_and_case.py
"""
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"


def show(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    pieces = tokenizer.convert_ids_to_tokens(ids)
    print(f"{text!r:>10} -> ids={ids} pieces={pieces}")


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # TODO: call show() on "hello", " hello", "Hello", " Hello" and look at
    # whether any of the four produce the same ids as another.
    ...

    print()
    print("Look at the 'pieces' column: does a leading space become its own")
    print("token, or does it merge into the following word's token?")


if __name__ == "__main__":
    main()
