"""Solution to exercise 3. Verified output (Qwen/Qwen3-0.6B):

--- ends in user (add_generation_prompt=True) ---
'<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n<|im_start|>user\\nWhat is 2+2?<|im_end|>\\n<|im_start|>assistant\\n'
contains '<think>': False

--- ends in assistant (continue_final_message=True) ---
'<|im_start|>user\\nWhat is 2+2?<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\nThe answer is'
contains '<think>': True

Qwen3's default template inserts an empty '<think>\\n\\n</think>\\n\\n' block
specifically when rendering assistant content -- even when that content is
just a plain-text prefill with no reasoning in it at all. System and user
turns are just facts being stated to the model; only the assistant turn is
something the model itself produces, and Qwen3 supports two response modes
(thinking and non-thinking, see the out-of-scope note on reasoning tokens).
The empty think block is the template's way of marking "no reasoning was
provided here" in the same structural slot a real reasoning trace would
occupy, keeping the format consistent whether or not thinking was used.
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

    rendered_user_ending = tokenizer.apply_chat_template(
        ENDS_IN_USER, tokenize=False, add_generation_prompt=True
    )
    rendered_assistant_ending = tokenizer.apply_chat_template(
        ENDS_IN_ASSISTANT, tokenize=False, continue_final_message=True
    )

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
