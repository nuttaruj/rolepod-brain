# bge-reranker-v2-m3, int8

## What this is

A cross-encoder: it reads a (question, entry) pair and returns one number for
how well the entry answers the question. Not a language model — it generates
nothing, so there is no prompt to write and no answer to parse.

Upstream: [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3),
Apache-2.0, 0.6B parameters, XLM-RoBERTa large. The ONNX export used as the
source is [`EmbeddedLLM/bge-reranker-v2-m3-onnx-o3-cpu`](https://huggingface.co/EmbeddedLLM/bge-reranker-v2-m3-onnx-o3-cpu).
`quantize.py` in this directory turns that into what the release ships.

## Why this model and not a smaller one

`jinaai/jina-reranker-v1-tiny-en` is 130 MB and four layers, which is what
basic-memory ships. It is English-only, so it is not a candidate here.

`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` is multilingual, Apache-2.0, and
118 MB int8 with official builds for every architecture — the obvious choice on
paper. Measured across fifteen queries on a real 22k-event brain, it does not
do the job:

    mean rank change of a reference reranker's picks   (negative is better)
      mmarco-mMiniLMv2 0.1B    +0.95      (English +2.0, Thai -4.7)
      bge-reranker-v2-m3 0.6B  -3.36      (English -2.4, Thai -8.5)

The small model left English rankings slightly worse than it found them. On
inspection it lost two queries of three; the large one lost none of four. The
likeliest reason is not language but shape: mMARCO is passage ranking, a
question against fifty to a hundred words of prose, and this corpus is titles
of five to ten words. Feeding the snippet as well moved it from +2.32 to +0.95
— the right direction, not far enough.

## Why int8

    file size          1136 MB (fp16) -> 569 MB
    session load          1.5s -> 0.3s
    30 pairs            1616ms -> 1135ms
    mean rank change     -3.36 -> -3.32

Half the bytes, a third off the inference, and quality inside the noise.

## Why local at all

Reranking through a host CLI is measured at a median of 12.2s on this brain.
The same judgement here costs 1.1s of inference, spends no subscription, needs
no credential, and starts no process. Smaller cross-encoders were measured
against this one before it was chosen; the comparison is in the note above.

## The runtime that runs it

The model is only half of what gets downloaded. ONNX Runtime itself arrives
beside it, one library per platform, published in our release as
`onnxruntime-<target>` and verified against `SHA256SUMS` like everything else.

These are Microsoft's own binaries, copied unmodified out of the official
release archives at
[`microsoft/onnxruntime`](https://github.com/microsoft/onnxruntime/releases),
MIT licensed:

| target | archive | version | sha256 of the library |
|---|---|---|---|
| `aarch64-apple-darwin` | `onnxruntime-osx-arm64-1.28.0.tgz` | 1.28.0 | `dc19bbcb2f5c9fb3c68b4f9248aa0a35065ff702c5dbeae75eac54a74da97b6d` |
| `x86_64-apple-darwin` | `onnxruntime-osx-x86_64-1.23.0.tgz` | 1.23.0 | `091d265e49da84ac8eafd6ff76b67688555192a272d784a252d550a858797d6f` |
| `x86_64-unknown-linux-gnu` | `onnxruntime-linux-x64-1.28.0.tgz` | 1.28.0 | `1461ef7cc3d9e49982591721683cc3e3a55580aeca9a5254e7aac47b75ee4bab` |
| `aarch64-unknown-linux-gnu` | `onnxruntime-linux-aarch64-1.28.0.tgz` | 1.28.0 | `f1ec1a08eb99bd6e5401340f0a2b101381bf4694415480291dc13bcaa30f9ec7` |

Those hashes are in the release workflow too, and a build stops rather than
ships if upstream ever answers with something else under the same name. They
are what makes "copied unmodified" a claim you can check.

Two versions rather than one, because Microsoft stopped building for Intel
macOS after 1.23. That platform gets the newest runtime it can have; the rest
get the newest there is. On the same thirty pairs, 1.28 scores them in 1.161s
and 1.23 in 1.575s, and the difference in the ranking is adjacent swaps inside
an identical top ten - same set, same first result.

The alternative was one version everywhere: 1.22 is the last that covers all
four. It takes 3.208s on the same batch. Paying that on every platform to
avoid maintaining a two-row table is not a trade worth making.

Nothing links ONNX Runtime into the binary. That is what lets every platform
carry the reranker at all - the prebuilt runtime the `ort` crate ships is
what demanded glibc 2.38, and it has no Intel macOS build. Microsoft's
official binaries need glibc 2.27.

## What is not established

That thirty candidates rank better than fifteen for this model. Only that
thirty is affordable — 1.1s against a CLI's 21.8s for the same width. Worth
measuring properly once `recalled.opened` has enough behaviour recorded to
judge a ranking without asking a model what it thinks.
