"""Solution to exercise 2. Verified output (Qwen/Qwen3-0.6B):

     'hello' -> ids=[14990] pieces=['hello']
    ' hello' -> ids=[23811] pieces=['Ġhello']
     'Hello' -> ids=[9707] pieces=['Hello']
    ' Hello' -> ids=[21927] pieces=['ĠHello']

All four are distinct token IDs. 'Ġ' is byte-level BPE's marker for "a space
preceded this" -- it isn't whitespace itself, it's baked into the token, so
a leading space isn't a separate token, it's merged into the following
word's token identity.
"""
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"


def show(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    pieces = tokenizer.convert_ids_to_tokens(ids)
    print(f"{text!r:>10} -> ids={ids} pieces={pieces}")


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    for text in ["hello", " hello", "Hello", " Hello"]:
        show(tokenizer, text)

    print()
    print("Look at the 'pieces' column: does a leading space become its own")
    print("token, or does it merge into the following word's token?")


if __name__ == "__main__":
    main()
