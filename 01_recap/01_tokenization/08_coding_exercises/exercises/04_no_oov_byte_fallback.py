"""
Exercise 4: no true out-of-vocabulary input.

Byte-level BPE's core guarantee (02_char_vs_word_vs_subword.md, question 11):
there is no such thing as a string this tokenizer can't handle. Throw
genuinely weird input at it -- emoji, nonsense, another script, invisible
characters -- and confirm nothing raises and everything round-trips exactly.

Run: python 04_no_oov_byte_fallback.py
"""
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    samples = [
        "\U0001f525\U0001f680",  # fire + rocket emoji
        "asdkfjhaslkdjfh",  # nonsense string, not a real word in any language
        "こんにちは",  # Japanese: "konnichiwa"
        "\u200b\u200b",  # zero-width space, twice -- invisible characters
    ]

    for text in samples:
        # TODO: encode `text`, decode it back, and check the round trip
        # matches exactly. Print len(ids) and whether it round-tripped.
        ...

    print()
    print("Every sample should round-trip exactly, with zero errors and zero")
    print("<unk> tokens. That's the guarantee -- it isn't best-effort.")


if __name__ == "__main__":
    main()
