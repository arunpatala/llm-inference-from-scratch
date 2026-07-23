# Course/book plan: llm-inference-from-scratch

## What this project is
Not a one-shot course — a living personal knowledge base on LLM inference
engineering, built and run on this GPU, structured so it stays useful and
shareable to others as it grows. Positioning: the PyTorch/CUDA equivalent of
`tiny-llm` (which is MLX/Apple-Silicon-only).

## Competitive research
Cloned 10 reference repos into `RESEARCH/REPOS_reference/` (sibling to the
course repo, not inside it — avoids license/attribution mess and git bloat):
`GeeeekExplorer/nano-vllm`, `ovshake/nano-vllm`, `skyzh/tiny-llm`,
`jianzhnie/mini-vllm`, `jmaczan/tiny-vllm`, `achi9629/llm-inference-engine`,
`pytorch-labs/gpt-fast`, `FareedKhan-dev/Building-llama3-from-scratch`,
`Wenyueh/MinivLLM`, `xlite-dev/Awesome-LLM-Inference`.

Key findings:
- `skyzh/tiny-llm` is the closest real precedent — genuine week-by-week
  course, fill-in-the-blank + reference solution + tests + mdBook site. Only
  gap: MLX/Apple Silicon only, excludes the much larger NVIDIA/Linux audience.
- `achi9629/llm-inference-engine` is closest in *scope* (GPT-2, KV cache,
  paged cache, continuous batching, FastAPI serving, A100 benchmarks) but is
  a finished portfolio project, not something a reader builds themselves.
- `ovshake/nano-vllm` has a fun "narrator/x-ray/dashboard/tutorial mode" idea
  but it's a finished engine you *watch*, not build.
- None of the 10 repos use Docker — native venv/conda + pinned requirements
  + an environment-check script is the universal norm in this space.
- Gap identified (our opening): nobody combines tiny-llm's pedagogical rigor
  with the PyTorch/CUDA mainstream stack and full serving-stack breadth.

## Locked decisions (via AskUserQuestion)
- **Exercise style**: Hybrid — full narrated walkthrough for boilerplate/glue
  code, fill-in-the-blank + tests only for the conceptually meaty function(s)
  per module.
- **Environment**: native conda/pip is primary; one optional `Dockerfile`
  lives only in the capstone module for zero-setup demoing.
- **Publishing**: mdBook site (buildable to GitHub Pages later).
- **Scope**: trimmed from an original 13-module idea to 8 core modules.
- **Model**: `Qwen/Qwen3-0.6B-Instruct` throughout all modules (changed from
  an initial GPT-2-124M pick). Real modern architecture (RoPE, RMSNorm,
  GQA + QK-norm, SwiGLU), ~1.2GB in fp16, trivial on this GPU, same model
  `tiny-llm` uses — makes this course a direct CUDA/PyTorch companion to it.

## Extensibility requirement (added mid-planning)
The book must support indefinite future growth without restructuring. Solved
with a Parts structure, folder-mirrored, numbering resets per part so new
chapters never require renumbering existing ones:

- **Part I — Foundations**: the 8 core modules below. Fixed once built.
- **Part II — Serving Systems** (future): prefix caching, chunked prefill,
  disaggregated prefill/decode, structured decoding, multi-LoRA, MoE serving.
- **Part III — Model Internals** (future): RoPE/GQA/SwiGLU-from-scratch deep
  dive, FlashAttention kernel, Triton, FP8 on Ada.
- **Part IV — Reading Notes** (future): one chapter per paper/technique read.

Two content tiers, so growth stays low-friction:
- **Core Module** — `exercise.py` + `solution.py` + `test_*.py` +
  `benchmark.py` + full book chapter. For foundational, build-it-yourself
  content.
- **Study Note** — book chapter + one demo script, no exercise/test split.
  For "I read/tried something new" entries. Part IV defaults to this tier.

Planned but not yet built: `scripts/new_chapter.py --part N --slug foo --tier
module|note` to scaffold a new chapter's folder + insert its SUMMARY.md entry
in one command.

Front matter (unnumbered, before Part I): Preface, Setup, **Prerequisites:
Transformer Fundamentals Recap** (added per explicit request — attention
equation, RoPE, GQA+QK-norm, RMSNorm, SwiGLU, mapped to the actual Qwen3
module names used in Module 00), Glossary. Recap is Study-Note tier.

## Part I — Foundations, final 8-module list
0. Qwen3 architecture from scratch + naive decoding (RoPE/GQA/RMSNorm/SwiGLU
   TODOs, assemble, verify logits vs real HF checkpoint, recompute-everything
   decode loop — shows O(n²))
1. KV cache
2. Batched inference
3. Continuous batching
4. Paged attention
5. Quantization (weight-only INT8/INT4)
6. Speculative decoding — draft-model pairing still undecided (self-
   speculative vs a second small same-tokenizer model); defer to when this
   module is actually built
7. Capstone — OpenAI-compatible FastAPI server + benchmark shootout
   (includes the one optional Dockerfile)

## Repo layout (drafted, partially scaffolded as empty dirs only)
```
llm-inference-from-scratch/
├── README.md
├── requirements.txt
├── scripts/            (doctor.py, download_weights.py, new_chapter.py — none written yet)
├── common/             (model_qwen3.py, tokenizer.py, bench_utils.py — none written yet)
├── book/                (mdBook source — not initialized yet)
├── modules/part1-foundations/00_.../07_capstone   (empty dirs only)
├── results/
├── .claude/skills/write-chapter/   (built — see writing_style log)
└── LOGS/
```

## Status at end of this session
Plan is approved through model choice and module list. No course content
(code or book chapters) written yet — only the writing-style skill (separate
log) and empty module directory skeletons. Next step on resume: build
`common/` infra + `scripts/doctor.py` + `download_weights.py`, then the
Prerequisites recap chapter (first real use of the interview workflow), then
Module 00 end-to-end.
