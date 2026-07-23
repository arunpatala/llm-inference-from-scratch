"""
Exercise 1: apply_chat_template, the basics.

Question 2 (../01_questions.md) asked what apply_chat_template actually
does mechanically. Render the same conversation with add_generation_prompt
True and False, and look at exactly what those extra tokens are.

Run: python 01_apply_chat_template_basics.py
"""
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"

MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
]


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # TODO: render MESSAGES with tokenize=False, once with
    # add_generation_prompt=True and once with add_generation_prompt=False.
    with_prompt = None
    without_prompt = None

    print("--- add_generation_prompt=True ---")
    print(repr(with_prompt))
    print()
    print("--- add_generation_prompt=False ---")
    print(repr(without_prompt))
    print()
    print("What exact characters does add_generation_prompt=True add at the")
    print("end? What would happen if you fed 'without_prompt' straight to")
    print("the model and asked it to generate?")


if __name__ == "__main__":
    main()
