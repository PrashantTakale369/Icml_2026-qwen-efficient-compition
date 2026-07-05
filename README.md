# ICML 2026 — The Efficient Qwen Competition

**Challenge:** Minimize inference latency for Qwen3.5-4B on a single NVIDIA A10G GPU while preserving model quality.

**Our result: Rank #20 out of 550+ submissions worldwide.**

The competition pushed participants to explore every layer of the inference stack — quantization, pruning, speculative decoding, and kernel-level optimizations. This repo contains our complete solution: all scripts, notebooks, and the serving stack exactly as submitted.

---

## What we did and why

The core insight was that **Qwen3.5-4B is memory-bandwidth bound at decode time**. Cutting the amount of data the GPU has to move per token is the fastest path to latency wins. We stacked 7 complementary techniques, each validated independently before combining.

### 1. W8A16 Main Body Quantization
We quantize all main transformer Linear layers to INT8 weights with FP16 activations (W8A16) using GPTQ with Hessian-aware calibration on UltraChat. This halves the memory footprint of every weight matrix, which directly halves the memory bandwidth consumed per decode step — the primary bottleneck. Quality loss is near-zero because GPTQ minimizes the layer-wise output error, not just the weight error.

### 2. MTP Head Grafting (FP16)
The GPTQ quantizer silently drops `mtp.*` tensors because they are non-standard HuggingFace keys — but vLLM's loader expects them since `config.json` declares `mtp_num_hidden_layers=1`. Without this step the model crashes on load. We copy the 15 MTP tensors from the original FP16 base into the W8A16 checkpoint as a prerequisite for Step 3.

### 3. MTP Head INT8 Quantization
vLLM's speculative decoding requires the MTP head to use the **same** quantization format as the main body. If the head stays FP16 while the body is INT8 the loader either raises a format mismatch or silently disables speculation. We quantize the 7 MTP projection layers (self-attn + MLP) to the same INT8 group-128 pack-quantized format as the main body.

### 4. GDN (GatedDeltaNet) INT8 Quantization
24 of the 32 layers in Qwen3.5-4B are GatedDeltaNet (linear-attention) layers — they dominate decode compute. We tried INT4 first; it tanked GPQA from 0.75 to 0.50. INT8 is safe and proven: MMLU 0.676 / IFEval 0.881 / GPQA 0.75. There is a critical vLLM constraint: it fuses `in_proj_qkv+in_proj_z → in_proj_qkvz` and `in_proj_b+in_proj_a → in_proj_ba` at load time. Both members of each fused pair **must** share the same quant scheme or vLLM raises an error — so all 5 projection types are quantized together.

### 5. Vision Encoder Removal
The benchmark is text-only, so the vision encoder (~0.6 GB of parameters) is pure dead weight. We remove it in two coordinated steps: (a) a **shim** (`shim/sitecustomize.py`) that monkeypatches `Qwen3_5ForConditionalGeneration.__init__` before vLLM loads — because vLLM hardcodes building the vision tower and you cannot skip it via config alone; (b) a **weight-stripping script** that deletes the 297 `visual.*` tensors from the safetensors file, because after the shim prevents the module from being built, the loader crashes when it finds weights with nowhere to go.

### 6. Speculative Decoding (MTP=8)
Qwen3.5-4B ships with a built-in Multi-Token Prediction head trained to predict the next 8 tokens. We enable vLLM's MTP self-speculation with `num_speculative_tokens=8`. At temperature=0 (which all benchmark evals use) acceptance rates are high and speculation is lossless — every accepted draft token is one fewer full-model forward pass.

### 7. Prefix Caching
Enable vLLM's prefix caching (`--enable-prefix-caching`). The benchmark includes repeated-prompt categories where the same long system prompt or few-shot prefix appears across many queries. Caching the KV state for those prefixes gives +8.5% on repeated-prompt latency (long category: 7.77× → 10.19×) at zero quality cost.

---

## Repository structure

```
.
├── 01_w8a16_main_body.ipynb     # Step 1: W8A16 GPTQ quantization of main body
├── 02_mtp_graft_fp16.ipynb      # Step 2: Graft FP16 MTP head into W8A16 checkpoint
├── 03_mtp_head_int8.ipynb       # Step 3: Quantize MTP head projections to INT8
├── 04_gdn_int8.ipynb            # Step 4: Quantize GDN linear_attn layers to INT8
├── 05_strip_vision.ipynb        # Step 5: Strip vision encoder weights + clean config
├── 06_serve_and_test.ipynb      # Step 6: Build Docker image, run, smoke test
|
└── Dockerfile                   # Container: pins vLLM 0.19.0, bakes runtime env vars
```

---

## How to reproduce

Run the notebooks in order (01 → 06). Each notebook ends with a verify cell — confirm it passes before moving to the next step.

```
01_w8a16_main_body.ipynb   →  qwen3.5-4b-w8a16-clean/
02_mtp_graft_fp16.ipynb    →  (in-place update)
03_mtp_head_int8.ipynb     →  (in-place update)
04_gdn_int8.ipynb          →  qwen3.5-4b-final-combo/
05_strip_vision.ipynb      →  (in-place update, vision stripped)
06_serve_and_test.ipynb    →  Docker build + smoke test
```

**Requirements:**
- Single NVIDIA A10G GPU (sm86)
- vLLM 0.19.0 
- `llmcompressor`, `compressed-tensors`, `safetensors`, `torch`, `fastapi`, `uvicorn`, `httpx`

---

## Verified results

| Metric | Score | Gate |
|--------|-------|------|
| MMLU | 0.676 | ≥ 0.621 ✅ |
| IFEval | 0.881 | ≥ 0.814 ✅ |
| GPQA | 0.727 | ≥ 0.630 ✅ |
| Latency speedup | ~10.4× | > 2.974× ✅ |
