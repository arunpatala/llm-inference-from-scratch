# Characters, bytes, and tokens are three different counts

A Unicode character, a UTF-8 byte, and a token are three separate things, and
their counts rarely match. Keeping them straight is what makes "byte-level BPE"
mean something specific instead of sounding like jargon.

Start from what byte-level BPE actually runs on. It does not run on characters.
It runs on the UTF-8 byte stream. A Unicode character is encoded into 1 to 4
bytes first, and BPE merges those bytes. The base vocabulary is the 256
possible byte values, so every possible byte is already in the vocabulary. That
is the reason there is never a true unknown token: any input, in any script,
decomposes into bytes, and all 256 bytes are known. Worst case, a piece of text
falls all the way back to one token per byte, but it always encodes.

So one character can span several tokens, because it was several bytes to begin
with. Here are real counts from Qwen3-0.6B's own tokenizer, produced by
`demos/bytes_chars_tokens_demo.py` (runs on CPU or MPS, no CUDA needed):

| text | characters | UTF-8 bytes | Qwen3 tokens |
|---|---|---|---|
| `hello` | 5 | 5 | 1 |
| `café` | 4 | 5 | 2 |
| `naïve` | 5 | 6 | 3 |
| `中` | 1 | 3 | 1 |
| `こんにちは` | 5 | 15 | 1 |
| `🫸` | 1 | 4 | 3 |
| `👨‍👩‍👧` | 5 | 18 | 7 |

Two things fall out of this table. First, frequency decides fragmentation: the
whole Japanese greeting `こんにちは` was common enough in training to earn a
single merged token, while a rarer emoji stays split into its raw bytes. This
is the same mechanic covered in `02_char_vs_word_vs_subword.md`, seen at the
byte level. Second, token count is not a proxy for character count, which
matters directly for anything measured in tokens: context length, KV cache
budget, and per-request cost.

## Why this causes a real streaming bug

Because a character can span multiple tokens, a decoder that turns each new
token back into text on its own can emit a broken partial byte sequence,
visible as the replacement character `�`. The `🫸` row above is a live example:
its 3 tokens are `[9284, 104, 116]`, and decoding each one alone produces `�`,
`�`, `�`. Only decoding all three together reconstructs `🫸`.

```
token 0 (id 9284) -> '�'
token 1 (id 104)  -> '�'
token 2 (id 116)  -> '�'
decode([9284, 104, 116]) -> '🫸'
```

A correct streaming decoder holds bytes back until they form a complete, valid
UTF-8 character before emitting anything. This is the concrete version of
question 21, and it has shipped as a real bug in more than one serving engine.
The fix belongs in the inference engine's detokenizer, not in the tokenizer.
