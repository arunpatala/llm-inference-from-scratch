"""
Exercise 4: how much of the prefix cache does dynamic content actually break?

Question 9: a system prompt with per-request content (here, today's date)
can break prefix-cache reuse. But how much of it -- everything after the
change, or genuinely everything? Render two conversations whose system
prompts differ by one digit and measure exactly where they diverge.

Run: python 04_prefix_cache_break.py
"""
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"


def render_ids(tokenizer, system_prompt, user_msg):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )["input_ids"]


def common_prefix_length(ids_a, ids_b):
    # TODO: return how many leading tokens are identical between ids_a and
    # ids_b, stopping at the first position where they differ.
    return 0


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    ids_a = render_ids(tokenizer, "You are a helpful assistant. Today is 2026-07-23.", "What is 2+2?")
    ids_b = render_ids(tokenizer, "You are a helpful assistant. Today is 2026-07-24.", "What is 2+2?")

    common = common_prefix_length(ids_a, ids_b)
    print(f"total length: {len(ids_a)} tokens")
    print(f"common prefix: {common} tokens")
    print(f"ids_a[common:] = {ids_a[common:]}")
    print(f"ids_b[common:] = {ids_b[common:]}")
    print()
    print("The date changed by one digit. Does the ENTIRE rest of the")
    print("sequence diverge, or does it re-converge right after the")
    print("changed token? What determines which one happens?")


if __name__ == "__main__":
    main()
