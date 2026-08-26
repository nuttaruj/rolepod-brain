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
no credential, and starts no process. `brief/10-cross-encoder-spike.md` has the
full measurement, including what was tried and rejected.

## What is not established

That thirty candidates rank better than fifteen for this model. Only that
thirty is affordable — 1.1s against a CLI's 21.8s for the same width. Worth
measuring properly once `recalled.opened` has enough behaviour recorded to
judge a ranking without asking a model what it thinks.
