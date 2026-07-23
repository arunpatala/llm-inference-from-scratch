# Environment setup

## Hardware
- GPU: NVIDIA RTX 4060 Ti, 16GB VRAM, Ada Lovelace (compute capability 8.9)
- ~288 GB/s memory bandwidth (128-bit bus) — the binding constraint for this card, not compute
- 165W power limit, driver 580.159.03, CUDA 13.0 runtime, nvcc toolkit 12.0
- 32GB system RAM, 370GB free disk

## Conda environment
- Created `llm-inf-scratch` (python 3.12) via `/home/arun/miniconda3`
- Installed: `torch==2.6.0+cu124`, `transformers==5.14.1`, `accelerate`, `triton==3.2.0`,
  `huggingface_hub`, `fastapi`, `uvicorn`, `pytest`, `matplotlib`, `sentencepiece`
- Install command used cu124 wheel index: `pip install torch --index-url https://download.pytorch.org/whl/cu124`

## mdBook
- No apt package, no cargo/rustc on this machine — installed prebuilt binary manually
- `mdbook v0.5.4` binary placed at `~/.local/bin/mdbook`
- Added `export PATH="$HOME/.local/bin:$PATH"` to `~/.bashrc`

## Other conda envs already on this machine (pre-existing, unrelated)
- `sglang_development/.conda`, `sglang_development/.conda_vllm`, `truecaptcha_monorepo/.conda`
- Noted as useful later: real vLLM/SGLang installs exist elsewhere on this machine for
  benchmarking our from-scratch engine against the real thing.
