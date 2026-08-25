#!/usr/bin/env python3
"""Reproduce model-int8.safetensors from the upstream f32 weights.

    pip install numpy safetensors
    curl -LO https://huggingface.co/minishlab/potion-retrieval-32M/resolve/6fc8051fab2a1e0ee76689cf08c853792ac285e7/model.safetensors
    python3 quantize.py model.safetensors model-int8.safetensors

Global symmetric scaling, and no scale is stored because none is needed: the
reader casts int8 straight to f32 and the model normalizes its output, so one
scale applied to every weight cancels in the pooled unit vector. Cosine
similarity — all this model is asked for — is unchanged by it.
"""
import sys

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

source, target = sys.argv[1], sys.argv[2]
with safe_open(source, "numpy") as handle:
    weights = handle.get_tensor("embeddings")

scale = np.abs(weights).max() / 127.0
save_file(
    {"embeddings": np.clip(np.rint(weights / scale), -127, 127).astype(np.int8)},
    target,
)
