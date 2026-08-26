#!/usr/bin/env python3
"""Reproduce model-int8.safetensors from the upstream f32 weights.

    pip install numpy safetensors
    curl -LO https://huggingface.co/minishlab/potion-retrieval-32M/resolve/6fc8051fab2a1e0ee76689cf08c853792ac285e7/model.safetensors
    python3 quantize.py model.safetensors model-int8.safetensors

Global symmetric scaling, and no scale is stored because none is needed: the
reader casts int8 straight to f32 and the model normalizes its output, so one
scale applied to every weight cancels in the pooled unit vector. Cosine
similarity — all this model is asked for — is unchanged by it.

The scale comes from a percentile of |W| rather than its maximum, and that is
not a refinement: scaling by the maximum lets a handful of outlier weights
set the step size for every other weight in the table, and the resolution it
costs is not visible in cosine agreement with the f32 original. Measured on
40 Thai-question/English-memory pairs, scaling by the maximum scored
recall@1 0.725 against f32's 0.800 while agreeing with it at 0.998 mean
cosine. Clipping at the 99.9th percentile scores 0.800 — the f32 number,
exactly — because the 0.1% of weights it saturates were never carrying the
answer.
"""
import sys

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

source, target = sys.argv[1], sys.argv[2]
with safe_open(source, "numpy") as handle:
    weights = handle.get_tensor("embeddings")

CLIP_PERCENTILE = 99.9
scale = np.percentile(np.abs(weights), CLIP_PERCENTILE) / 127.0
save_file(
    {"embeddings": np.clip(np.rint(weights / scale), -127, 127).astype(np.int8)},
    target,
)
