# FsColbert

FsColbert is an F#/.NET library for local ColBERT-style retrieval over documents. It
targets `net10.0` and is designed for applications that need on-device or local
semantic search over PDFs, Markdown, and pre-chunked text without taking a dependency
on a specific UI framework.

The library currently provides:

- ColBERT late-interaction retrieval with MaxSim dense scoring.
- ONNX Runtime inference for `lightonai/mxbai-edge-colbert-v0-32m-onnx`.
- Hugging Face byte-level BPE tokenization through `Microsoft.ML.Tokenizers`.
- Section-aware PDF and Markdown passage extraction.
- TF-IDF candidate filtering with optional keyword-weighted expansion before dense reranking.
- Parallel batch indexing through `FSharp.Control.AsyncSeq`.
- Binary `.fsci` index persistence and index bundle manifests.
- Docling-compatible document types, JSON serialization, passage conversion, and a standard hybrid assembly pipeline.
- ONNX helpers for Docling layout detection and figure classification models.
- Thin integration helpers for [FsVoice](https://github.com/fwaris/FsVoice)-style PDF source and context rendering workflows.

FsColbert does not reference FsKame, .NET MAUI, or app UI types. Host applications own
storage, source selection, OCR/rasterization providers, and any user-facing lifecycle.

## Requirements

- .NET SDK with `net10.0` support.
- ONNX Runtime-compatible target platform.
- Network access when using `ModelCatalog.ensureDownloadedAsync` to fetch model files.
  Apps that cannot access Hugging Face at runtime can reference
  `FsColbert.Models.MxbaiEdgeColbertV0_32M.Onnx.Int8` and use
  `ModelCatalog.ensureAvailableAsync` to resolve packaged files before falling
  back to download.

Key package dependencies are:

- `Microsoft.ML.OnnxRuntime`
- `Microsoft.ML.Tokenizers`
- `PdfPig`
- `FSharp.Control.AsyncSeq`
- `F23.StringSimilarity`
- `FSharp.DI`

## Basic Retrieval Flow

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
        SourceDocuments.create
            "guide"
            "Guide"
            "/docs/guide.txt"
            "Local document text to index and search."
            true

    let! index =
        IndexBuilder.createWithDefaults
            encoder
            [ source ]
            (Some(fun progress ->
                printfn "%d/%d %A"
                    progress.completedPassages
                    progress.totalPassages
                    progress.currentSource))

    let! hits =
        Search.queryWithDefaults
            encoder
            index
            "What local document text is indexed?"

    return SearchHits.renderContext 900 hits
}
```

## Reading Documents

Use `PdfDocuments` when the source file is a PDF and plain PdfPig extraction is
enough:

```fsharp
open FsColbert

async {
    let source = PassageSource.create "handbook" "Handbook" "/docs/Handbook.pdf"
    let! passages = PdfDocuments.readPassages ChunkOptions.fsKameDefaults source "/docs/Handbook.pdf"

    match passages with
    | Ok passages ->
        // Use createFromPassages when passages have already been produced.
        return passages
    | Error message ->
        failwith message
}
```

Use `MarkdownDocuments` for Markdown files. It preserves nested heading context by
prefixing passages with section paths such as `Section: Guide > Setup > macOS`.

For already-split text, use `SourceDocuments.createPreChunked` or build `PassageRef`
values directly and call `IndexBuilder.createFromPassages`.

## Indexing And Search

Default chunking is tuned for FsKame-like PDF context:

```fsharp
ChunkOptions.fsKameDefaults
// maxChars = 1800
// overlapChars = 250
// minChars = 20
```

`IndexingOptions.defaults` reads these environment variables when present:

- `FSCOLBERT_INDEX_PARALLELISM`
- `FSCOLBERT_MODEL_REPLICAS`
- `FSCOLBERT_INDEX_BATCH_SIZE`

`OnnxColbertEncoder.Load` also honors `FSCOLBERT_MODEL_REPLICAS` unless a replica
count is passed explicitly.

Search uses TF-IDF to select candidates, then reranks candidates with ColBERT MaxSim.
`SearchOptions.defaults` returns up to 6 results, considers up to 128 lexical
candidates, and uses reciprocal rank fusion with dense and lexical scores.

When an app has query expansion terms, pass them separately so dense scoring still
uses the original query:

```fsharp
let! hits =
    Search.queryWithDefaultsAndSearchTerms
        encoder
        index
        "car repair"
        [ "automobile"; "maintenance manual" ]
```

## Keyword Elaboration

Passages can carry `keywords` that are indexed into TF-IDF without modifying the raw
passage text. This is useful for synonyms, figure labels, table metadata, or
application-supplied expansion terms.

Implement `IPassageKeywordGenerator` and pass it through
`KeywordElaborationOptions.withGenerator` when building an index:

```fsharp
let keywordOptions = KeywordElaborationOptions.withGenerator generator

let! index =
    IndexBuilder.createFromPassagesWithOptionsAndTfidfOptionsAndKeywordElaborationOptions
        encoder
        ChunkOptions.fsKameDefaults
        IndexingOptions.defaults
        TfidfOptions.defaults
        keywordOptions
        passages
        None
```

## Persistence

Indexes can be saved and loaded as `.fsci` files:

```fsharp
IndexPersistence.save "/indexes/handbook.fsci" index

let loaded =
    IndexPersistence.load "/indexes/handbook.fsci"
```

The current binary format is version 3 and stores TF-IDF options plus passage
keywords. Version 2 indexes are still readable and load with empty keyword lists.

For packaged or prebuilt indexes, use `IndexBundle` manifests:

```fsharp
match IndexBundle.loadCompatible IndexBundleCompatibility.fsKameDefaults "/bundle/manifest.json" with
| Ok bundle ->
    bundle.indexes |> List.iter (fun entry -> printfn "Loaded %s" entry.source.sourceDisplayName)
| Error errors ->
    errors |> List.iter eprintfn "%s"
```

## Docling Support

FsColbert includes a Docling-oriented document model for apps that want structured
PDF conversion before indexing:

- `DoclingJson` serializes and deserializes the supported Docling JSON subset.
- `DoclingPassages.toPassages` converts Docling texts, tables, and pictures into
  searchable `PassageRef` values with derived keywords.
- `DoclingPdfNative.readPageCells` extracts native PDF text cells with PdfPig.
- `DoclingStandardHybrid` assembles page images, OCR/native cells, layout
  predictions, and optional figure classifications into a `DoclingDocument`.
- `DoclingLayoutOnnx` and `DoclingFigureClassifierOnnx` provide ONNX-backed
  implementations for layout prediction and figure classification.

Model manifests are available for:

```fsharp
ModelCatalog.doclingLayoutHeronOnnx
ModelCatalog.doclingDocumentFigureClassifierV25Onnx
```

Download those files with `ModelCatalog.ensureDoclingOnnxDownloadedAsync`.

The library defines rasterizer and OCR interfaces, but the host app supplies concrete
implementations:

- `IDoclingPageRasterizer`
- `IDoclingOcrProvider`
- `IDoclingLayoutPredictor`
- `IDoclingFigureClassifier`

Cancelable variants are available for long-running conversion and indexing paths.

## FsKame Integration Notes

The FsKame-oriented integration layer is intentionally thin:

- `SourceDocuments.fromFsKamePdf` and `fromFsKamePdfChunked` map selected PDFs into
  FsColbert source records.
- `SearchHits.renderContext` formats ranked passages for prompt/context injection.
- `SearchHits.sourceInventory` formats selected source names.

A typical FsKame integration keeps the existing PDF library and selection lifecycle,
then swaps chunk loading and ranking for an FsColbert index:

1. Map each selected PDF into `SourceDocument`, `PreChunkedDocument`, or `PassageRef`.
2. Build or load a persisted `ColbertIndex` when selected sources change.
3. On transcript finalization, call `Search.queryWithDefaults` or
   `Search.queryWithDefaultsAndSearchTerms`.
4. Render hits with `SearchHits.renderContext`, or map `SearchHit.reference` back to
   the host app's source chunk model.

## Build And Test

```bash
dotnet restore
dotnet build FsColbert.slnx
dotnet test FsColbert.slnx
```

The test project covers chunking, Markdown/PDF-oriented document processing,
TF-IDF candidate selection, dense reranking, persistence compatibility, Docling JSON
and passage conversion, cancellation behavior, and index bundle validation.
