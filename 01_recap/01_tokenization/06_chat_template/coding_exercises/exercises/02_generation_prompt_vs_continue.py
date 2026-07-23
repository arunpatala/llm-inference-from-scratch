"""
Exercise 2: add_generation_prompt vs. continue_final_message.

Question 3: what's the actual difference, and why can't both be set at
once? Render a prefilled assistant turn with continue_final_message=True,
then deliberately trigger the conflict error.

Run: python 02_generation_prompt_vs_continue.py
"""
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"

MESSAGES_PREFILL = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "The answer is"},
]


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # TODO: render MESSAGES_PREFILL with tokenize=False and
    # continue_final_message=True. Print it.
    rendered = None
    print("--- continue_final_message=True ---")
    print(repr(rendered))
    print()

    # TODO: call apply_chat_template with BOTH add_generation_prompt=True
    # and continue_final_message=True in a try/except, and print the
    # exception type and message.
    print("--- both flags set at once ---")
    ...

    print()
    print("Does the rendered string end mid-sentence, ready for the model")
    print("to keep writing 'The answer is' -- or does it start a fresh turn?")


if __name__ == "__main__":
    main()
