"""Solution to exercise 1. Verified output (Qwen/Qwen3-0.6B):

--- add_generation_prompt=True ---
'<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n<|im_start|>user\\nWhat is 2+2?<|im_end|>\\n<|im_start|>assistant\\n'

--- add_generation_prompt=False ---
'<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n<|im_start|>user\\nWhat is 2+2?<|im_end|>\\n'

add_generation_prompt=True appends exactly '<|im_start|>assistant\\n' with no
closing '<|im_end|>' -- an open invitation for the model to start writing
and eventually close it itself. Feeding 'without_prompt' straight to the
model would leave it staring at a completed, closed conversation with no
signal that it's now its turn to speak -- it would have to somehow infer
that from context rather than from an explicit marker.
"""
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"

MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
]


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    with_prompt = tokenizer.apply_chat_template(
        MESSAGES, tokenize=False, add_generation_prompt=True
    )
    without_prompt = tokenizer.apply_chat_template(
        MESSAGES, tokenize=False, add_generation_prompt=False
    )

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
