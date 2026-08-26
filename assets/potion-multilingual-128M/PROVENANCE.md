# potion-multilingual-128M

Static embedding model, vendored so that semantic search needs no download, no
second runtime, and no network at any point.

| | |
|---|---|
| Upstream | https://huggingface.co/minishlab/potion-multilingual-128M |
| Revision | `73908c3438cf03b6a01bcb9611d62b23d0726f08` — pinned, because a Hugging Face branch is mutable |
| Author | MinishLab |
| License | MIT |
| Lineage | distilled from `BAAI/bge-m3` via [Model2Vec](https://github.com/MinishLab/model2vec), trained on 101 languages |
| Tokenizer lineage | `BAAI/bge-m3`, MIT (per the model's `config.json`) |
| Dimensions | 256 |
| Rows | 500,353 — one per vocabulary entry, which is what makes a token id a safe row index |

`model-int8.safetensors` is not the upstream file. Upstream ships `embeddings`
as f32 at 489 MB; this is the same tensor quantized to int8 at 122 MB, by
global symmetric scaling:

    s  = max(|W|) / 127
    Wq = clip(round(W / s), -127, 127).astype(int8)

No scale is stored, and none is needed. `model2vec-rs` reads an int8 tensor by
casting straight to f32, and the model normalizes its output, so a single
scale applied to every weight cancels in the pooled unit vector — and cosine
similarity, which is all we ask of it, is unchanged by it.

Measured against the f32 original over 3,000 randomly pooled token sets of
3–40 tokens: mean cosine 0.99916, worst 0.97680.

**That number is not the one to trust, and this file said otherwise once.**
An earlier revision scaled by `max(|W|)` and reported 0.99828 mean cosine as
proof the quantization was free. It was not: on the retrieval task below it
scored recall@1 0.725 where f32 scored 0.800 — a tenth of the answer, invisible
in cosine agreement, because a handful of outlier weights were setting the step
size for the whole table. Clipping the scale at the 99.9th percentile of |W|
saturates 0.1% of weights and scores **exactly** f32's numbers. Cosine
agreement with the original is a weak proxy for whether a quantized table
still ranks the same things first; the ranking has to be measured directly.

## Why this model and not the English one

The model this replaces, `potion-retrieval-32M`, was a quarter the size and
tuned for English retrieval. It was replaced because it could not answer a
question asked in any other language at all — not badly, at all. Measured on
40 pairs of a Thai question and the English memory that answers it, retrieving
the right English text out of the other 39:

| | recall@1 | recall@5 | MRR |
|---|---|---|---|
| potion-retrieval-32M (512d) | 0.025 | 0.125 | 0.107 |
| potion-multilingual-128M (256d) | **0.800** | **0.975** | **0.866** |

Those are the numbers for the int8 file vendored here, not for the f32
original — they are the same numbers, which is the point of the clipping
above.

`0.025` is one in forty: chance. The English model was not ranking those
memories poorly, it was ranking them randomly, because a Thai sentence
tokenizes into 30 single consonants in its vocabulary and pools to noise.

The half-width vectors cost nothing measurable on the English corpus this
brain actually holds. Session co-membership as ground truth, 25 resamples of
the real event store, mean ± population standard deviation:

| | recall@1 | recall@5 | MRR |
|---|---|---|---|
| potion-retrieval-32M (512d) | 0.447 ± 0.091 | 0.651 ± 0.095 | 0.552 ± 0.080 |
| potion-multilingual-128M (256d) | 0.456 ± 0.067 | 0.659 ± 0.074 | 0.557 ± 0.062 |

Every difference is an order of magnitude inside the spread. A single draw of
that benchmark shows a regression or an improvement depending on the seed,
which is why it is reported as a mean over resamples and not as one number.

## Verifying this

`quantize.py` in this directory is the whole transformation, and the checksums
below are what it produced — on any machine, which took a second attempt to
be true. The scale was an interpolated percentile at first, and an
interpolated percentile differs by an ULP between platforms: macOS and Linux
ran the same script over the same input and produced two different files.
It is an exact order statistic now, a value that is actually in the array, so
every platform computes the identical scale and therefore the identical file.
The release workflow rebuilds it from upstream and checks it against this
line, which is the only thing that makes the claim checkable rather than
merely stated. A 122 MB blob nobody can read is worth exactly the
provenance attached to it.

```
sha256, upstream model.safetensors (f32, 489 MB, at the revision above)
  14b5eb39cb4ce5666da8ad1f3dc6be4346e9b2d601c073302fa0a31bf7943397

sha256, vendored here
  41f5e8169c8b280471115f41bb0f2664554fabdf762fc6063dd9c722f4080eb0  model-int8.safetensors
  19f1909063da3cfe3bd83a782381f040dccea475f4816de11116444a73e1b6a1  tokenizer.json
  595e4cab2093732efd5dbe084fd5c1826b5eea693b73b4c1fd971672867d2e54  config.json
```

`tokenizer.json` and `config.json` are upstream, unmodified.

The inference this file feeds is written out in `src/embed.rs` rather than
delegated, to keep 489 MB of expanded table off the heap. Its equivalence to
the reference implementation is asserted by a test, against the reference as a
dev-dependency, so the two cannot quietly drift apart.
