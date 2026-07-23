# Two-machine split: local Mac (dev) vs CUDA box (benchmarks)

This repo was planned and started on a Linux **NVIDIA RTX 4060 Ti** box (see
`20260722_1327_environment_setup.md`). It is now also being worked on from a
**MacBook Pro (Apple M4)**. These are different machines with different
capabilities, and the book's "every number traces to a real benchmark" rule
means the split has to be explicit — a throughput number measured on the Mac
is not the same claim as one measured on the 4060 Ti, and can't be swapped in
for it.

## The two machines

| | Local Mac (this machine) | CUDA box (original) |
|---|---|---|
| Chip | Apple M4, 10-core CPU (4P+6E), 10-core GPU | RTX 4060 Ti, Ada, cc 8.9 |
| Accel backend | MPS / Metal 4 | CUDA 12.4 |
| Memory | 16 GB unified (shared with OS) | 16 GB dedicated VRAM + 32 GB RAM |
| Mem bandwidth | unified LPDDR | ~288 GB/s (128-bit) — the binding constraint there |
| torch | 2.7.1 (MPS), no CUDA | 2.6.0+cu124 |
| triton | NOT installed / not available on MPS | 3.2.0 |
| transformers / tokenizers / accelerate / fastapi / uvicorn / pytest | installed | installed |
| sentencepiece | NOT installed here | installed |

Reality check run on the Mac: `torch.cuda.is_available() == False`,
`torch.backends.mps.is_available() == True`, `import triton` -> ModuleNotFound.

## What IS possible on the local Mac

Anything that doesn't touch CUDA or a CUDA-only library. Use it as the primary
dev/authoring/test environment:

- All prose/chapter writing + `scripts/style_lint.py` (pure Python).
- Tokenizer work end-to-end — the entire `01_recap/01_tokenization` module,
  including every coding exercise (BPE, chat templates, byte fallback,
  tied-embedding check). Runs on CPU via `tokenizers`/`transformers`, no GPU.
- Loading Qwen3-0.6B in fp16 (~1.2 GB) on **MPS** for correctness work:
  logits-vs-HF verification, naive decode loop, KV-cache correctness, sampling.
  Enough VRAM headroom for a 0.6B model, though 16 GB is *shared* with the OS.
- Pure-PyTorch module logic that is backend-agnostic (write it `.to(device)`
  with `device = "mps" if available else "cpu"`), and its `pytest` correctness
  tests.
- `common/` infra, `scripts/doctor.py`, `download_weights.py` — all authorable
  and unit-testable here.
- FastAPI capstone *server code* (fastapi/uvicorn present) — the HTTP layer and
  its logic, tested against a CPU/MPS backend.

## What is NOT possible here — needs the CUDA box

- Anything importing `triton` or writing Triton kernels.
- Custom CUDA kernels, FlashAttention (the real kernel), FP8-on-Ada work
  (Part III topics).
- Any published **benchmark number** for the book. All throughput/latency/
  memory figures that go in a chapter must be measured on the 4060 Ti and
  labeled with that hardware. MPS numbers are for local sanity only and must
  never be presented as the book's benchmark.
- `bitsandbytes`-style CUDA-only INT8/INT4 quant paths (Module 5) — the kernel
  side. Algorithm logic can be prototyped on CPU/MPS; the fast path can't.
- Continuous batching / paged-attention throughput validation at scale, and
  the vLLM/SGLang comparison runs (those installs live on the Linux box).

## Working rule

Develop and correctness-test on the Mac; treat the 4060 Ti as the benchmark
authority. When a chapter needs a real number, mark it PENDING and measure it
on the CUDA box before the chapter is called done — never fill it with an MPS
number or a plausible guess.
