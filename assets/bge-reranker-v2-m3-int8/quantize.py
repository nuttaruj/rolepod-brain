"""Quantise the reranker to int8, reproducibly.

The published weights are float. This is what turns them into what the release
ships, and it is here so anyone can check that the file in the release is the
file this script produces — the same reason `potion-multilingual-128M` carries
its own.

    python3 quantize.py <fp32-model.onnx> <out.onnx>

Dynamic quantisation: weights become int8 on disk, activations are quantised on
the fly at inference. No calibration set, so nothing depends on which corpus
happened to be around when someone ran it — the same input file gives the same
output file, which is the property a checksum has to be able to promise.

Two things learned the hard way, both worth keeping:

**Quantise from float32, never from the float16 build.** Quantising the fp16
graph succeeds and produces a model that will not load: `DequantizeLinear`
rejects a float16 scale, and the failure arrives at load time rather than at
quantisation time.

**Declare the default tensor type.** The published graph has been through
onnxruntime's o3 fusion, so shape inference cannot type every intermediate and
gives up on the first MatMul output. `DefaultTensorType` tells it what it could
not work out. Without it: "Unable to find data type for weight_name=..."

Measured against the float16 build, fifteen queries on a 22k-event brain:

    file size     1136 MB -> 569 MB
    session load     1.5s -> 0.3s
    30 pairs        1616ms -> 1135ms
    mean rank change -3.36 -> -3.32   (unchanged; the difference is noise)
"""

import sys
import time

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    source, target = sys.argv[1], sys.argv[2]
    started = time.time()
    quantize_dynamic(
        model_input=source,
        model_output=target,
        weight_type=QuantType.QInt8,
        # One file, not a graph plus a sidecar that has to agree with it.
        use_external_data_format=False,
        extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
    )
    print(f"quantised {source} -> {target} in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
