"""
Exercise 5: does the chat template ever skip the tokenizer?

Question 8: is there a special path where the chat template produces token
IDs directly, or is it strictly text-formatting before ordinary
tokenization? Test it directly: render to a string yourself, tokenize that
string yourself, and compare against apply_chat_template's tokenize=True
shortcut.

Run: python 05_tokenize_vs_render.py
"""
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"

MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
]


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # TODO: get the rendered string (tokenize=False), then manually
    # tokenize it with tokenizer.encode(..., add_special_tokens=False).
    rendered_str = None
    ids_manual = None

    # TODO: get the IDs directly via tokenize=True on the same messages.
    ids_direct = None

    print("manual: ", ids_manual)
    print("direct: ", ids_direct)
    print("identical:", ids_manual == ids_direct)
    print()
    print("If they're identical, what does that prove about whether")
    print("tokenize=True is doing something the tokenizer itself couldn't")
    print("do on the already-rendered string?")


if __name__ == "__main__":
    main()
