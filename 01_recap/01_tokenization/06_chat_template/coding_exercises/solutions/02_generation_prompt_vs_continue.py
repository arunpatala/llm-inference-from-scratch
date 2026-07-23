"""Solution to exercise 2. Verified output (Qwen/Qwen3-0.6B):

--- continue_final_message=True ---
'<|im_start|>user\\nWhat is 2+2?<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\nThe answer is'

--- both flags set at once ---
error raised: ValueError continue_final_message and add_generation_prompt
are not compatible. Use continue_final_message when you want the model to
continue the final message, and add_generation_prompt when you want to add
a...

The rendered string ends mid-sentence with no closing '<|im_end|>' after
"The answer is" -- the model is meant to keep writing from exactly that
point, not start a new turn. That's why the two flags conflict: one means
"the conversation is complete, add a fresh empty turn," the other means
"the conversation is mid-turn, don't close it." They describe mutually
exclusive states of the same last message.
"""
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"

MESSAGES_PREFILL = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "The answer is"},
]


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    rendered = tokenizer.apply_chat_template(
        MESSAGES_PREFILL, tokenize=False, continue_final_message=True
    )
    print("--- continue_final_message=True ---")
    print(repr(rendered))
    print()

    print("--- both flags set at once ---")
    try:
        tokenizer.apply_chat_template(
            MESSAGES_PREFILL,
            tokenize=False,
            add_generation_prompt=True,
            continue_final_message=True,
        )
        print("no error raised")
    except ValueError as e:
        print("error raised:", type(e).__name__, str(e)[:200])

    print()
    print("Does the rendered string end mid-sentence, ready for the model")
    print("to keep writing 'The answer is' -- or does it start a fresh turn?")


if __name__ == "__main__":
    main()
