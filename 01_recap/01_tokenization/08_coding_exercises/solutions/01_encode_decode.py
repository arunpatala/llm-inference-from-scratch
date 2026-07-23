"""Solution to exercise 1. Verified output (Qwen/Qwen3-0.6B):

text:      'Tokenization turns text into numbers.'
token ids: [3323, 2022, 10577, 1467, 1119, 5109, 13]
n_tokens:  7  (n_chars: 37)
decoded:   'Tokenization turns text into numbers.'
pieces:    ['Token', 'ization', 'Ġturns', 'Ġtext', 'Ġinto', 'Ġnumbers', '.']
"""
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    text = "Tokenization turns text into numbers."

    token_ids = tokenizer.encode(text, add_special_tokens=False)

    print(f"text:      {text!r}")
    print(f"token ids: {token_ids}")
    print(f"n_tokens:  {len(token_ids)}  (n_chars: {len(text)})")

    decoded = tokenizer.decode(token_ids)
    print(f"decoded:   {decoded!r}")
    assert decoded == text, "round trip failed"

    pieces = tokenizer.convert_ids_to_tokens(token_ids)
    print(f"pieces:    {pieces}")
    print("Notice 'Tokenization' splits into 'Token' + 'ization' -- exactly")
    print("the example from 02_char_vs_word_vs_subword.md, for real.")


if __name__ == "__main__":
    main()
