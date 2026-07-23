"""Solution to exercise 4. Verified output (Qwen/Qwen3-0.6B):

total length: 40 tokens
common prefix: 21 tokens
ids_a[common:] = [18, 13, 151645, 198, 151644, 872, 198, 3838, 374, 220, 17, 10, 17, 30, 151645, 198, 151644, 77091, 198]
ids_b[common:] = [19, 13, 151645, 198, 151644, 872, 198, 3838, 374, 220, 17, 10, 17, 30, 151645, 198, 151644, 77091, 198]

It re-converges immediately after the single changed token. Position 21 is
the one digit that differs (day "3" vs "4"), and every token after that --
the rest of the date, the closing tags, the entire user turn, the
generation prompt -- is byte-for-byte identical again in both sequences.

That's because Qwen's tokenizer splits digits individually (exercise 3 in
../../05_bpe/coding_exercises), so a one-digit change costs exactly one
token's worth of divergence, not a shift in every subsequent token
position. A prefix-caching engine reusing KV blocks for the first 21 tokens
and recomputing only from token 21 onward would be exactly correct here --
the breakage is real, but it's much smaller than "the whole cache is
invalidated." A change that altered the *length* of the tokenized system
prompt (not just a same-length digit swap) would shift every position
after it and break the entire remaining prefix, though.
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
    common = 0
    for x, y in zip(ids_a, ids_b):
        if x != y:
            break
        common += 1
    return common


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
