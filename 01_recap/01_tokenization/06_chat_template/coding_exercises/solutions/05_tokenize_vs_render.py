"""Solution to exercise 5. Verified output (Qwen/Qwen3-0.6B):

manual:  [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 3838, 374, 220, 17, 10, 17, 30, 151645, 198, 151644, 77091, 198]
direct:  [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 3838, 374, 220, 17, 10, 17, 30, 151645, 198, 151644, 77091, 198]
identical: True

They're exactly identical. That proves apply_chat_template's tokenize=True
is a convenience wrapper, not a separate code path -- internally it renders
the Jinja template to a flat string first, then hands that string to the
ordinary tokenizer with add_special_tokens=False (the template already
wrote out every special token itself). There is no mechanism by which a
chat template produces token IDs without text ever existing as an
intermediate step.
"""
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"

MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
]


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    rendered_str = tokenizer.apply_chat_template(
        MESSAGES, tokenize=False, add_generation_prompt=True
    )
    ids_manual = tokenizer.encode(rendered_str, add_special_tokens=False)

    ids_direct = tokenizer.apply_chat_template(
        MESSAGES, tokenize=True, add_generation_prompt=True
    )["input_ids"]

    print("manual: ", ids_manual)
    print("direct: ", ids_direct)
    print("identical:", ids_manual == ids_direct)
    print()
    print("If they're identical, what does that prove about whether")
    print("tokenize=True is doing something the tokenizer itself couldn't")
    print("do on the already-rendered string?")


if __name__ == "__main__":
    main()
