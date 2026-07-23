"""
Demo: what apply_chat_template actually does, and why the generation prompt
matters.

Shows the two-step pipeline:
  messages (list of {role, content} dicts)
    -> chat template renders a flat STRING (text only, with special tokens)
    -> tokenizer turns that string into token IDs

Also shows the generation prompt: add_generation_prompt=True appends an open
assistant turn (`<|im_start|>assistant\n` with nothing after) so the model's
next-token prediction continues as the assistant.

Runs on CPU/MPS, no CUDA. Downloads the Qwen3-0.6B tokenizer once.

Run: python chat_template_demo.py
"""
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
    {"role": "user", "content": "And 3+3?"},
]


def show_render():
    # Step 1: array -> string (text only). tokenize=False proves it's a str.
    s = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("=== rendered string (tokenize=False) ===")
    print(s)
    print(f"--- type: {type(s).__name__} (text, not token IDs) ---\n")

    # Step 2: string -> token IDs (separate step).
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)["input_ids"]
    print(f"=== tokenized: {len(ids)} token IDs ===\n{ids}\n")


def show_generation_prompt_difference():
    # The only difference add_generation_prompt makes: the trailing open turn.
    without = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    with_gp = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("=== last 40 chars WITHOUT generation prompt ===")
    print(repr(without[-40:]))
    print("=== last 40 chars WITH generation prompt ===")
    print(repr(with_gp[-40:]))
    print("The extra `<|im_start|>assistant\\n` is what cues the model to speak.\n")


def show_special_tokens():
    print("=== special tokens ===")
    for t in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]:
        print(f"  {t!r:16} -> id {tok.convert_tokens_to_ids(t)}")
    print(f"  eos_token = {tok.eos_token!r} (id {tok.eos_token_id}) <- the stop signal")
    print(f"  bos_token = {tok.bos_token!r} <- Qwen3 has no BOS; im_start marks the start")


if __name__ == "__main__":
    show_render()
    show_generation_prompt_difference()
    show_special_tokens()
