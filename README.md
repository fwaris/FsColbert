# FsColbert

FsColbert is a small F#/.NET library for mobile-friendly ColBERT-style retrieval. It is built for later use from a .NET MAUI app such as FsKame.

The first implementation uses:

- `lightonai/mxbai-edge-colbert-v0-32m-onnx` INT8 ONNX model
- Hugging Face byte-level BPE tokenization via `Microsoft.ML.Tokenizers`
- ONNX Runtime inference via `Microsoft.ML.OnnxRuntime`
- ColBERT late-interaction MaxSim scoring
- FsKame-like text chunking defaults: 1800 characters with 250 overlap
- TF-IDF inverted-index candidate prefiltering before dense scoring
- binary persistence for local on-device indexes

## Basic Flow

```fsharp
open System.Net.Http
open FsColbert

async {
    use http = new HttpClient()

    let! files =
        ModelCatalog.ensureDownloadedAsync
            http
            "/path/to/appdata/FsColbert/Models/mxbai-edge-colbert"
            ModelCatalog.mxbaiEdgeColbertInt8

    use encoder = OnnxColbertEncoder.Load files

    let source =
        SourceDocuments.fromFsKamePdf
            "pdf-id"
            "Handbook.pdf"
            "/path/to/Handbook.pdf"
            "PDF text extracted by FsKame/PdfPig"
            true

    let! index =
        IndexBuilder.createWithDefaults
            encoder
            [ source ]
            (Some(fun progress -> printfn "%A" progress))

    let! hits =
        Search.queryWithDefaults
            encoder
            index
            "What does the handbook say about local indexing?"

    let context = SearchHits.renderContext 900 hits
    return context
}
```

## FsKame Integration Notes

FsKame already has the right lifecycle:

1. `PdfLibrary` copies PDFs into app data.
2. `KnowledgeSources.readPdfText` extracts text with PdfPig.
3. `SourceAgent` loads chunks and ranks them for each final transcript.

The clean integration path is to keep FsKame's PDF extraction, then replace `KnowledgeSources.loadChunks` / `KnowledgeSources.rank` with an FsColbert-backed state:

- Map each selected `PdfDocumentSource` to `SourceDocument`.
- Build or load a persisted `ColbertIndex` when sources change.
- On transcript finalization, call `Search.queryWithDefaults`.
- Map `SearchHit.reference` back into FsKame's existing `SourceChunk` record for `OracleAgent`.

The library intentionally does not reference FsKame, MAUI UI types, or PdfPig. That keeps it portable across Android, iOS, Mac Catalyst, and test projects.
