FROM adaptfm/adaptfm-base:latest

# Do NOT upgrade vLLM — base image ships 0.19.0, the only version compatible with
# the eval host's CUDA 12.4 driver.

# ──────────────────────────────────────────────────────────────────────────────
# FINAL COMBO CONFIG (validated: GPQA 0.727 / IFEval 0.826, ~10.4x dev)
#   1. W8A16 main body         -> model weights
#   2. MTP head INT8           -> model weights
#   3. GDN (linear_attn) INT8  -> model weights
#   4. vision encoder REMOVED  -> shim (skips building visual) + weights stripped
#   5. prefix caching ON       -> wrapper default
#   6. PERF_MODE               -> NOT set (it failed GPQA; removed)
#   7. speculative decoding MTP=8
# ──────────────────────────────────────────────────────────────────────────────


COPY model/ /opt/ml/model/
COPY 2_serve.py /opt/program/my_serve.py
# Item 4: vision-removal shim, auto-imported via PYTHONPATH before vLLM builds.
COPY shim/ /opt/shim/

ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV HF_HUB_OFFLINE=1
ENV PYTHONPATH=/opt/shim
ENV SPEC_TOKENS=8
# PERF_MODE intentionally NOT set. PREFIX_CACHE unset -> prefix caching ON (default).

ENTRYPOINT ["python3", "/opt/program/my_serve.py"]
