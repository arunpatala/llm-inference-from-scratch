"""Solution to exercise 3. Verified output (Qwen/Qwen3-0.6B):

     '12345' -> ids=[16, 17, 18, 19, 20] pieces=['1', '2', '3', '4', '5']
     'hello' -> ids=[14990] pieces=['hello']
         '9' -> ids=[24] pieces=['9']
  '99999999' -> ids=[24, 24, 24, 24, 24, 24, 24, 24] pieces=['9']*8
   '3.14159' -> ids=[18, 13, 16, 19, 16, 20, 24] pieces=['3','.','1','4','1','5','9']

'12345' (5 chars) becomes 5 tokens; 'hello' (5 chars) becomes 1 token.
Digit-splitting is deliberate: it forces the model to see every digit
individually rather than merging common short numbers into single opaque
tokens, which is why models with digit-split tokenizers tend to do
multi-digit arithmetic more reliably.
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

    for text in samples:
        show(tokenizer, text)

    print()
    print("'12345' is 5 characters and 'hello' is 5 characters. Do they")
    print("produce the same number of tokens? What does that tell you about")
    print("why a tokenizer designer would choose digit-splitting on purpose?")


if __name__ == "__main__":
    main()
