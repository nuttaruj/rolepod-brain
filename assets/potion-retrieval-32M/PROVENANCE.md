# potion-retrieval-32M

Static embedding model, vendored so that semantic search needs no download, no
second runtime, and no network at any point.

| | |
|---|---|
| Upstream | https://huggingface.co/minishlab/potion-retrieval-32M |
| Revision | `6fc8051fab2a1e0ee76689cf08c853792ac285e7` — pinned, because a Hugging Face branch is mutable |
| Author | MinishLab |
| License | MIT |
| Lineage | a finetune of `minishlab/potion-base-32M`, itself distilled from a sentence transformer via [Model2Vec](https://github.com/MinishLab/model2vec) |
| Tokenizer lineage | `baai/bge-base-en-v1.5`, MIT (per the model's `config.json`) |
| Dimensions | 512 |
| Rows | 63,091 — one per vocabulary entry, which is what makes a token id a safe row index |

`model-int8.safetensors` is not the upstream file. Upstream ships `embeddings`
as f32 at 129 MB; this is the same tensor quantized to int8 at 32.3 MB, by
global symmetric scaling:

    s  = max(|W|) / 127
    Wq = clip(round(W / s), -127, 127).astype(int8)

No scale is stored, and none is needed. `model2vec-rs` reads an int8 tensor by
casting straight to f32, and the model normalizes its output, so a single
scale applied to every weight cancels in the pooled unit vector — and cosine
similarity, which is all we ask of it, is unchanged by it.

Measured against the f32 original over 3,000 randomly pooled token sets of
3–40 tokens: mean cosine 0.99969, worst 0.99933. A quarter of the size for
three ten-thousandths of the answer.

`tokenizer.json` and `config.json` are upstream, unmodified.

## Verifying this

`quantize.py` in this directory is the whole transformation, and the checksums
below are what it produced. A 32 MB blob nobody can read is worth exactly the
provenance attached to it.

```
sha256, upstream model.safetensors (f32, 129 MB, at the revision above)
  07609e5bd33aad37900b3fd62f4ec96f6daec88ca4d46b9d8b928bfababf6ea0

sha256, vendored here
  d50d32a8f57ed92a5814632167fc87797c540b5ec9e8567335b90c07f3cf25b6  model-int8.safetensors
  7d75cbc54318138807c401b0f0c9721117c628b39de8e8e0edb6cb17e0ee7d18  tokenizer.json
  63c00d90824c832c04ec1d02b6a983fb90489bf049f29fbff15ba481b8a432ee  config.json
```

The inference this file feeds is written out in `src/embed.rs` rather than
delegated, to keep 129 MB of expanded table off the heap. Its equivalence to
the reference implementation is asserted by a test, against the reference as a
dev-dependency, so the two cannot quietly drift apart.
