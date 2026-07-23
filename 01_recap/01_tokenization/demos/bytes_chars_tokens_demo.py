"""
Demo: characters vs UTF-8 bytes vs Qwen3 tokens are three different counts,
and why naive per-token streaming can emit broken "replacement characters".

Byte-level BPE (what Qwen3 uses) runs on the UTF-8 *byte* stream, not on
Unicode characters. Its base vocabulary is the 256 possible byte values, which
is why there is never a true "unknown" token. A single Unicode character is
encoded to 1-4 bytes first, so one character can span several tokens.

Runs on CPU/MPS, no CUDA. Downloads the Qwen3-0.6B tokenizer (~a few MB) once.

Run: python bytes_chars_tokens_demo.py
"""
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


def show_counts():
    samples = ["A", "é", "中", "🙂", "hello", "café", "naïve", "こんにちは", "👨‍👩‍👧"]
    print(f"{'text':12} {'chars':>5} {'utf-8 bytes':>11} {'tokens':>6}")
    print("-" * 40)
    for s in samples:
        ids = tok.encode(s, add_special_tokens=False)
        print(f"{s!r:12} {len(s):>5} {len(s.encode('utf-8')):>11} {len(ids):>6}")


def show_streaming_break():
    # U+1FAF8 fragments into 3 tokens under Qwen3's tokenizer.
    s = "🫸"
    ids = tok.encode(s, add_special_tokens=False)
    print(f"\nstreaming demo on {s!r}: {len(s.encode('utf-8'))} bytes -> {len(ids)} tokens {ids}")
    print("  naive (decode each token as it arrives):")
    for i, tid in enumerate(ids):
        print(f"    token {i} (id {tid}) -> {tok.decode([tid])!r}   (broken partial UTF-8)")
    print(f"  correct (buffer until valid): decode(all) = {tok.decode(ids)!r}")


if __name__ == "__main__":
    show_counts()
    show_streaming_break()
