# FsColbert.Models.MxbaiEdgeColbertV0_32M.Onnx.Int8

Optional packaged model assets for FsColbert's default ColBERT encoder.

This package includes the int8 ONNX assets from
`lightonai/mxbai-edge-colbert-v0-32m-onnx`:

- `model_int8.onnx`
- `tokenizer.json`
- `onnx_config.json`

Applications that reference this package can use FsColbert retrieval without
downloading the Mxbai Edge ColBERT model from Hugging Face at runtime. If the
package is not referenced, FsColbert can still use its normal Hugging Face
download fallback.

## Asset Path

The package exposes assets under:

```text
FsColbert/Models/mxbai-edge-colbert/
```

For non-MAUI projects, the package copies the assets to the build output under
that same relative path. For .NET MAUI projects, the package registers them as
`MauiAsset` items with the same logical path.

## Licensing

The package license expression is `MIT AND Apache-2.0`.

See `THIRD-PARTY-NOTICES.md`, `licenses/LIGHTON-ONNX-MIT.txt`, and
`licenses/APACHE-2.0.txt` for attribution and license text.
