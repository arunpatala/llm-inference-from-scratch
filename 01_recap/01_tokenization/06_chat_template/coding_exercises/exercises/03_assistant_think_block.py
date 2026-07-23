"""
Exercise 3: what special-casing the "assistant" role actually looks like.

Question 6 asked why "assistant" needs special handling at all. Render a
conversation ending in a user turn, and a conversation ending in an
assistant turn (a prefill), and compare. Look specifically for `<think>`.

Run: python 03_assistant_think_block.py
"""
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"

ENDS_IN_USER = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
]

ENDS_IN_ASSISTANT = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "The answer is"},
]


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # TODO: render ENDS_IN_USER with add_generation_prompt=True, and
    # ENDS_IN_ASSISTANT with continue_final_message=True. Both tokenize=False.
    rendered_user_ending = None
    rendered_assistant_ending = None

    print("--- ends in user (add_generation_prompt=True) ---")
    print(repr(rendered_user_ending))
    print("contains '<think>':", "<think>" in rendered_user_ending)
    print()
    print("--- ends in assistant (continue_final_message=True) ---")
    print(repr(rendered_assistant_ending))
    print("contains '<think>':", "<think>" in rendered_assistant_ending)
    print()
    print("System and user turns never get a <think> block. Only the")
    print("assistant role does, and only automatically when its content is")
    print("being rendered. Why would system/user never need one?")


if __name__ == "__main__":
    main()
