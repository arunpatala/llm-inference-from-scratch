"""
Demo: naive per-token detokenization emits broken '�' mid-stream; a correct
streaming detokenizer buffers across tokens and only emits complete UTF-8.

The bug (see 02a_bytes_vs_chars_vs_tokens.md): a multibyte character can span
several tokens (🫸 = 3 tokens in Qwen3). Decoding each token alone yields a
partial byte sequence that renders as the replacement char '�'.

The fix: decode against the ACCUMULATED token stream and only release text that
is stable (doesn't end in an incomplete character). Real engines (vLLM, TGI) do
this with byte offsets; this demo uses the simplest correct version.

Runs on CPU, no CUDA.

Run: python streaming_detokenizer_demo.py
"""
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

# 🫸 fragments into 3 tokens under Qwen3; stream them one at a time.
STREAM = tok.encode("🫸", add_special_tokens=False)
REPLACEMENT = "�"  # '�'


def naive_stream(token_ids):
    """WRONG: decode each new token in isolation."""
    out = []
    for tid in token_ids:
        piece = tok.decode([tid])
        out.append(piece)
        print(f"  token {tid}: emit {piece!r}")
    return "".join(out)


def correct_stream(token_ids):
    """RIGHT: decode the accumulated stream, only release stable text."""
    seen = []
    emitted = ""
    result = ""
    for tid in token_ids:
        seen.append(tid)
        full = tok.decode(seen)
        # If the accumulated text ends in a replacement char, the last
        # character isn't complete yet -> hold, emit nothing this step.
        if full.endswith(REPLACEMENT):
            print(f"  token {tid}: (incomplete char, buffer & wait) emit ''")
            continue
        delta = full[len(emitted):]
        emitted = full
        result += delta
        print(f"  token {tid}: emit {delta!r}")
    return result


def main():
    print(f"streaming tokens for '🫸': {STREAM}\n")
    print("=== naive (decode each token alone) ===")
    naive = naive_stream(STREAM)
    print(f"user sees: {naive!r}   <- three broken replacement chars\n")

    print("=== correct (buffer until valid UTF-8) ===")
    correct = correct_stream(STREAM)
    print(f"user sees: {correct!r}   <- the emoji, emitted once complete")


if __name__ == "__main__":
    main()
