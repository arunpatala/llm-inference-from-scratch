"""
Exercise 1: encode / decode round trip.

Load the real Qwen3-0.6B tokenizer, encode a string into token IDs, look at
the individual token strings, then decode back. This is the "hello world"
of tokenization: text -> IDs -> text, and nothing else.

Run: python 01_encode_decode.py
"""
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    text = "Tokenization turns text into numbers."

    # TODO: encode `text` into a list of token IDs (no special tokens)
    token_ids = None

    print(f"text:      {text!r}")
    print(f"token ids: {token_ids}")
    print(f"n_tokens:  {len(token_ids)}  (n_chars: {len(text)})")

    # TODO: decode `token_ids` back into a string
    decoded = None
    print(f"decoded:   {decoded!r}")
    assert decoded == text, "round trip failed"

    # Bonus, already done for you: look at individual token strings, not IDs
    pieces = tokenizer.convert_ids_to_tokens(token_ids)
    print(f"pieces:    {pieces}")
    print("Notice 'Tokenization' splits into 'Token' + 'ization' -- exactly")
    print("the example from 02_char_vs_word_vs_subword.md, for real.")


if __name__ == "__main__":
    main()
