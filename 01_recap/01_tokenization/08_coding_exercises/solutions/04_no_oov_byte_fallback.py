"""Solution to exercise 4. Verified output (Qwen/Qwen3-0.6B):

'🔥🚀' -> n_tokens=2 roundtrip_ok=True
'asdkfjhaslkdjfh' -> n_tokens=9 roundtrip_ok=True
'こんにちは' -> n_tokens=1 roundtrip_ok=True
'\u200b\u200b' -> n_tokens=1 roundtrip_ok=True

Every sample round-trips exactly. No exceptions, no <unk>. Byte-level BPE's
base vocabulary is the 256 possible byte values, so any Unicode string --
however strange -- decomposes into bytes that are all already in vocab.
"""
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    samples = [
        "\U0001f525\U0001f680",  # fire + rocket emoji
        "asdkfjhaslkdjfh",  # nonsense string, not a real word in any language
        "こんにちは",  # Japanese: "konnichiwa"
        "\u200b\u200b",  # zero-width space, twice -- invisible characters
    ]

    for text in samples:
        ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(ids)
        print(f"{text!r} -> n_tokens={len(ids)} roundtrip_ok={decoded == text}")

    print()
    print("Every sample should round-trip exactly, with zero errors and zero")
    print("<unk> tokens. That's the guarantee -- it isn't best-effort.")


if __name__ == "__main__":
    main()
