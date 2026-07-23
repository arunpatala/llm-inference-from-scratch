"""
Demo: the full inference pipeline end to end (Q5), on the real Qwen3-0.6B.

  messages (list of dicts)
    -> chat template renders a STRING (with special tokens + generation prompt)
    -> tokenizer -> input_ids tensor
    -> model.generate samples new token IDs one at a time
    -> stop when <|im_end|> (id 151645) is emitted
    -> decode the NEW ids back to text

Also demonstrates two failure modes discussed in the interview:
  F1: if the loop does NOT stop on <|im_end|>, the model rambles into a
      hallucinated next turn.
  F2: if the generation prompt is missing, the model tends to open a new turn
      instead of answering.

Runs on MPS or CPU, no CUDA. Downloads Qwen3-0.6B weights (~1.2 GB) once.
Greedy decoding (do_sample=False) so the output is reproducible.

Run: python generation_pipeline_demo.py
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3-0.6B"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(DEVICE)
model.eval()

IM_END = tok.convert_tokens_to_ids("<|im_end|>")      # 151645, the real stop token
END_OF_TEXT = tok.convert_tokens_to_ids("<|endoftext|>")  # 151643, a DIFFERENT token

messages = [{"role": "user", "content": "What is 2+2? Answer in one word."}]


def gen(text, max_new_tokens, eos_token_id):
    ids = tok(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=eos_token_id,
            pad_token_id=END_OF_TEXT,
        )
    new_ids = out[0][ids["input_ids"].shape[1]:]
    return ids["input_ids"].shape[1], new_ids


def main():
    print(f"device: {DEVICE}\n")

    # --- Correct full pipeline (stops on <|im_end|>) ---
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    n_in, new_ids = gen(text, max_new_tokens=32, eos_token_id=IM_END)
    print("=== correct pipeline (stops on <|im_end|>) ===")
    print(f"prompt tokens: {n_in}")
    print(f"generated ids: {new_ids.tolist()}")
    print(f"last id: {new_ids[-1].item()}  (151645 = <|im_end|> -> stopped cleanly)")
    print(f"decoded (skip specials): {tok.decode(new_ids, skip_special_tokens=True)!r}\n")

    # --- F1: do NOT stop on <|im_end|> (use a token it won't emit) ---
    n_in, new_ids = gen(text, max_new_tokens=60, eos_token_id=END_OF_TEXT)
    print("=== F1: EOS-stop disabled -> runaway into a fake turn ===")
    print(f"decoded (keep specials): {tok.decode(new_ids, skip_special_tokens=False)!r}\n")

    # --- F2: no generation prompt ---
    text_no_gp = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    n_in, new_ids = gen(text_no_gp, max_new_tokens=20, eos_token_id=IM_END)
    print("=== F2: no generation prompt -> model opens a new turn ===")
    print(f"decoded (keep specials): {tok.decode(new_ids, skip_special_tokens=False)!r}")


if __name__ == "__main__":
    main()
