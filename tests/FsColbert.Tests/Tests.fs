module FsColbert.Tests

open System
open System.IO
open System.Net.Http
open System.Numerics
open System.Threading
open FsColbert
open Xunit

let private vectorWithDim dim tokenIds vectors =
    { tokenIds = tokenIds
      vectors = vectors
      tokenCount = tokenIds.Length
      embeddingDim = dim }

let private vector tokenIds vectors = vectorWithDim 2 tokenIds vectors

let private passageWithKeywords sourceId index text keywords embedding =
    { reference =
        { sourceId = sourceId
          sourceDisplayName = $"PDF: {sourceId}"
          sourceLocation = $"/tmp/{sourceId}.pdf"
          index = index
          text = text
          sectionPath = []
          contentRole = PassageContentRole.Unknown
          pageNumbers = []
          layoutLabels = []
          captions = []
          keywords = keywords }
      embedding = embedding
      terms = Text.terms text }

let private passage sourceId index text embedding =
    passageWithKeywords sourceId index text [] embedding

let private passageWithSection sourceId index text sectionPath keywords embedding =
    { reference =
        { sourceId = sourceId
          sourceDisplayName = $"PDF: {sourceId}"
          sourceLocation = $"/tmp/{sourceId}.pdf"
          index = index
          text = text
          sectionPath = sectionPath
          contentRole = PassageContentRole.MainBody
          pageNumbers = []
          layoutLabels = []
          captions = []
          keywords = keywords }
      embedding = embedding
      terms = Text.terms text }

let private passageWithMetadata sourceId index text sectionPath contentRole pageNumbers keywords embedding =
    let reference =
        { sourceId = sourceId
          sourceDisplayName = $"PDF: {sourceId}"
          sourceLocation = $"/tmp/{sourceId}.pdf"
          index = index
          text = text
          sectionPath = sectionPath
          contentRole = contentRole
          pageNumbers = pageNumbers
          layoutLabels = []
          captions = []
          keywords = keywords }

    { reference = reference
      embedding = embedding
      terms = Text.terms (PassageContext.contextualText reference) }

type private StaticKeywordGenerator(results: PassageKeywordResult list) =
    interface IPassageKeywordGenerator with
        member _.GenerateKeywordsAsync passages =
            async {
                let requested =
                    passages
                    |> List.map (fun passage -> passage.sourceId, passage.index)
                    |> Set.ofList

                return
                    results
                    |> List.filter (fun result -> Set.contains (result.sourceId, result.passageIndex) requested)
            }

type private FailingHttpMessageHandler() =
    inherit HttpMessageHandler()

    override _.SendAsync(_: HttpRequestMessage, _: CancellationToken) =
        raise (InvalidOperationException "HTTP should not be used when model files are available locally.")

let private index passages =
    { config = EncoderConfig.mxbaiEdgeColbert
      chunkOptions = ChunkOptions.fsKameDefaults
      tfidfOptions = TfidfOptions.defaults
      passages = passages
      tfidf = Tfidf.buildWithOptions TfidfOptions.defaults passages
      createdAt = DateTimeOffset.UtcNow }

let private writeConfig (writer: BinaryWriter) (config: EncoderConfig) =
    writer.Write config.queryLength
    writer.Write config.documentLength
    writer.Write config.embeddingDim
    writer.Write config.padTokenId
    writer.Write config.maskTokenId
    writer.Write config.clsTokenId
    writer.Write config.sepTokenId
    writer.Write config.queryPrefixId
    writer.Write config.documentPrefixId
    writer.Write config.doLowerCase
    writer.Write config.normalizeOutput
    writer.Write config.skiplistWords.Count

    for word in config.skiplistWords do
        writer.Write word

let private writeVector (writer: BinaryWriter) (embedding: MultiVector) =
    writer.Write embedding.embeddingDim
    writer.Write embedding.tokenCount
    writer.Write embedding.tokenIds.Length

    for tokenId in embedding.tokenIds do
        writer.Write tokenId

    writer.Write embedding.vectors.Length

    for value in embedding.vectors do
        writer.Write value

let private writeTfidf (writer: BinaryWriter) (idx: TfidfIndex) =
    writer.Write idx.passageCount
    writer.Write idx.averageDocumentLength
    writer.Write idx.vocabulary.Count

    for KeyValue(term, termInfo) in idx.vocabulary |> Seq.sortBy _.Key do
        writer.Write term
        writer.Write termInfo.documentFrequency
        writer.Write termInfo.inverseDocumentFrequency
        writer.Write termInfo.postings.Length

        for posting in termInfo.postings do
            writer.Write posting.passageOrdinal
            writer.Write posting.termFrequency

let private writeVersion2Index path (idx: ColbertIndex) =
    use stream = File.Create path
    use writer = new BinaryWriter(stream)
    writer.Write "FSCOLBERT-IDX"
    writer.Write 2
    writeConfig writer idx.config
    writer.Write idx.chunkOptions.maxChars
    writer.Write idx.chunkOptions.overlapChars
    writer.Write idx.chunkOptions.minChars
    writer.Write(idx.createdAt.ToUnixTimeMilliseconds())
    writer.Write idx.passages.Length

    for passage in idx.passages do
        writer.Write passage.reference.sourceId
        writer.Write passage.reference.sourceDisplayName
        writer.Write passage.reference.sourceLocation
        writer.Write passage.reference.index
        writer.Write passage.reference.text
        writer.Write passage.terms.Count

        for term in passage.terms do
            writer.Write term

        writeVector writer passage.embedding

    writeTfidf writer idx.tfidf

let private writeStringList (writer: BinaryWriter) (values: string list) =
    writer.Write values.Length

    for value in values do
        writer.Write value

let private writeVersion3Index path (idx: ColbertIndex) =
    use stream = File.Create path
    use writer = new BinaryWriter(stream)
    writer.Write "FSCOLBERT-IDX"
    writer.Write 3
    writeConfig writer idx.config
    writer.Write idx.chunkOptions.maxChars
    writer.Write idx.chunkOptions.overlapChars
    writer.Write idx.chunkOptions.minChars
    writer.Write idx.tfidfOptions.textWeight
    writer.Write idx.tfidfOptions.keywordWeight
    writer.Write(idx.createdAt.ToUnixTimeMilliseconds())
    writer.Write idx.passages.Length

    for passage in idx.passages do
        writer.Write passage.reference.sourceId
        writer.Write passage.reference.sourceDisplayName
        writer.Write passage.reference.sourceLocation
        writer.Write passage.reference.index
        writer.Write passage.reference.text
        writeStringList writer passage.reference.keywords
        writer.Write passage.terms.Count

        for term in passage.terms do
            writer.Write term

        writeVector writer passage.embedding

    writeTfidf writer idx.tfidf

let private writeVersion4Index path (idx: ColbertIndex) =
    use stream = File.Create path
    use writer = new BinaryWriter(stream)
    writer.Write "FSCOLBERT-IDX"
    writer.Write 4
    writeConfig writer idx.config
    writer.Write idx.chunkOptions.maxChars
    writer.Write idx.chunkOptions.overlapChars
    writer.Write idx.chunkOptions.minChars
    writer.Write idx.tfidfOptions.textWeight
    writer.Write idx.tfidfOptions.keywordWeight
    writer.Write(idx.createdAt.ToUnixTimeMilliseconds())
    writer.Write idx.passages.Length

    for passage in idx.passages do
        writer.Write passage.reference.sourceId
        writer.Write passage.reference.sourceDisplayName
        writer.Write passage.reference.sourceLocation
        writer.Write passage.reference.index
        writer.Write passage.reference.text
        writeStringList writer passage.reference.keywords
        writeStringList writer passage.reference.sectionPath
        writer.Write passage.terms.Count

        for term in passage.terms do
            writer.Write term

        writeVector writer passage.embedding

    writeTfidf writer idx.tfidf

let private writeVersion5Index path (idx: ColbertIndex) =
    use stream = File.Create path
    use writer = new BinaryWriter(stream)
    writer.Write "FSCOLBERT-IDX"
    writer.Write 5
    writeConfig writer idx.config
    writer.Write idx.chunkOptions.maxChars
    writer.Write idx.chunkOptions.overlapChars
    writer.Write idx.chunkOptions.minChars
    writer.Write idx.tfidfOptions.textWeight
    writer.Write idx.tfidfOptions.keywordWeight
    writer.Write(idx.createdAt.ToUnixTimeMilliseconds())
    writer.Write idx.passages.Length

    for passage in idx.passages do
        writer.Write passage.reference.sourceId
        writer.Write passage.reference.sourceDisplayName
        writer.Write passage.reference.sourceLocation
        writer.Write passage.reference.index
        writer.Write passage.reference.text
        writeStringList writer passage.reference.keywords
        writeStringList writer passage.reference.sectionPath
        writer.Write(PassageContentRole.storageValue passage.reference.contentRole)
        writer.Write passage.reference.pageNumbers.Length

        for pageNumber in passage.reference.pageNumbers do
            writer.Write pageNumber

        writer.Write passage.terms.Count

        for term in passage.terms do
            writer.Write term

        writeVector writer passage.embedding

    writeTfidf writer idx.tfidf

[<Fact>]
let ``chunkText preserves overlap friendly chunks`` () =
    let options =
        { maxChars = 10
          overlapChars = 2
          minChars = 1 }

    let chunks = Text.chunkText options "abcdefghijklmnopqrstuvwxyz"

    Assert.Equal<int list>([ 0; 1; 2; 3 ], chunks |> List.map fst)
    Assert.Equal("abcdefghij", chunks |> List.head |> snd)
    Assert.Equal("ijklmnopqr", chunks[1] |> snd)

[<Fact>]
let ``chunkSectionedBlocks keeps section heading with following body`` () =
    let options =
        { ChunkOptions.fsKameDefaults with
            maxChars = 200
            overlapChars = 20 }

    let chunks =
        DocumentChunking.chunkSectionedBlocks
            options
            [ "Title"
              "ABSTRACT"
              "This paper introduces a retrieval method for scientific documents."
              "INTRODUCTION"
              "The introduction motivates the problem." ]

    Assert.True(
        chunks
        |> List.exists (fun chunk -> chunk.StartsWith("Section: ABSTRACT") && chunk.Contains("retrieval method"))
    )

[<Fact>]
let ``passagesFromBlocks emits bounded section-aware passages`` () =
    let options =
        { ChunkOptions.fsKameDefaults with
            maxChars = 80
            overlapChars = 10
            minChars = 1 }

    let source = PassageSource.create "paper" "Paper" "/tmp/paper.pdf"

    let passages =
        DocumentChunking.passagesFromBlocks
            options
            source
            [ "ABSTRACT"
              "This abstract has enough text to require more than one small passage for indexing." ]

    Assert.All(passages, fun passage -> Assert.True(passage.text.Length <= options.maxChars))
    Assert.Equal("paper", passages.Head.sourceId)
    Assert.Equal("Paper", passages.Head.sourceDisplayName)
    Assert.Equal<string list>([ "ABSTRACT" ], passages.Head.sectionPath)

[<Fact>]
let ``section matching tolerates small spelling mistakes`` () =
    Assert.True(DocumentSections.matches "abtract" "ABSTRACT")
    Assert.True(DocumentSections.matches "intrduction" "INTRODUCTION")
    Assert.False(DocumentSections.matches "results" "REFERENCES")

[<Fact>]
let ``markdown passages preserve nested heading context`` () =
    async {
        let path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.md")

        do!
            File.WriteAllTextAsync(
                path,
                "# Project Guide\n\nIntro paragraph.\n\n## Setup\n\nInstall the tool.\n\n### macOS\n\nRun the installer.\n\n- Open settings\n- Enable indexing\n"
            )
            |> Async.AwaitTask

        try
            let source = PassageSource.create "guide" "Guide" path
            let! result = MarkdownDocuments.readPassages ChunkOptions.fsKameDefaults source path

            match result with
            | Error err -> failwith err
            | Ok passages ->
                Assert.Contains(
                    passages,
                    fun passage ->
                        passage.text.StartsWith("Section: Project Guide > Setup > macOS")
                        && passage.text.Contains("Run the installer")
                )

                Assert.Contains(
                    passages,
                    fun passage ->
                        passage.text.StartsWith("Section: Project Guide > Setup > macOS")
                        && passage.text.Contains("Enable indexing")
                )
        finally
            if File.Exists path then
                File.Delete path
    }
    |> Async.RunSynchronously

[<Fact>]
let ``termFrequencies counts repeated terms`` () =
    let frequencies, total = Text.termFrequencies "Apple apple banana."

    Assert.Equal(3, total)
    Assert.Equal(2, frequencies["apple"])
    Assert.Equal(1, frequencies["banana"])

[<Fact>]
let ``maxSim sums each query token best document token`` () =
    let query = vector [| 1; 2 |] [| 1.0f; 0.0f; 0.0f; 1.0f |]
    let doc = vector [| 3; 4 |] [| 0.9f; 0.1f; 0.2f; 0.8f |]

    let score = Scoring.maxSim query doc

    Assert.Equal(1.7f, score, 3)

[<Fact>]
let ``maxSim handles vectorized dimensions with scalar tail`` () =
    let dim = Vector<float32>.Count + 3
    let query = vectorWithDim dim [| 1 |] (Array.create dim 1.0f)

    let doc =
        vectorWithDim dim [| 2; 3 |] (Array.append (Array.create dim 1.0f) (Array.create dim 2.0f))

    let score = Scoring.maxSim query doc

    Assert.Equal(float32 (dim * 2), score, 3)

[<Fact>]
let ``idf gives rare terms higher weight than common terms`` () =
    let idx =
        index
            [ passage "one" 0 "common rare" (vector [| 1 |] [| 1.0f; 0.0f |])
              passage "two" 0 "common shared" (vector [| 1 |] [| 1.0f; 0.0f |])
              passage "three" 0 "common shared" (vector [| 1 |] [| 1.0f; 0.0f |]) ]

    let rare = idx.tfidf.vocabulary["rare"]
    let common = idx.tfidf.vocabulary["common"]

    Assert.True(rare.inverseDocumentFrequency > common.inverseDocumentFrequency)

[<Fact>]
let ``tfidf candidates prefer unique matching passage`` () =
    let idx =
        index
            [ passage "one" 0 "alpha beta" (vector [| 1 |] [| 1.0f; 0.0f |])
              passage "two" 0 "gamma delta" (vector [| 1 |] [| 1.0f; 0.0f |]) ]

    let candidates = Tfidf.scoreQuery idx.tfidf "gamma" |> Tfidf.topCandidates 10
    let ordinal, score = candidates[0]

    Assert.Equal(1, ordinal)
    Assert.True(score > 0.0f)

[<Fact>]
let ``tfidf supplied search terms broaden candidate lookup`` () =
    let idx =
        index
            [ passage "one" 0 "automobile repair manual" (vector [| 1 |] [| 1.0f; 0.0f |])
              passage "two" 0 "kitchen recipe notes" (vector [| 1 |] [| 1.0f; 0.0f |]) ]

    let literalCandidates = Tfidf.scoreQuery idx.tfidf "car" |> Tfidf.topCandidates 10

    let expandedCandidates =
        Tfidf.scoreQueryWithSearchTerms idx.tfidf "car" [ "automobile" ]
        |> Tfidf.topCandidates 10

    Assert.Empty literalCandidates
    Assert.Equal(0, expandedCandidates[0] |> fst)

[<Fact>]
let ``tfidf indexes keyword-only terms without changing raw passage text`` () =
    let idx =
        index
            [ passageWithKeywords
                  "one"
                  0
                  "benefits waiting period details"
                  [ "orthodontia"; "dental braces" ]
                  (vector [| 1 |] [| 1.0f; 0.0f |])
              passage "two" 0 "kitchen recipe notes" (vector [| 1 |] [| 1.0f; 0.0f |]) ]

    let candidates = Tfidf.scoreQuery idx.tfidf "orthodontia" |> Tfidf.topCandidates 10
    let ordinal, score = candidates[0]

    Assert.Equal(0, ordinal)
    Assert.True(score > 0.0f)
    Assert.DoesNotContain("orthodontia", idx.passages[0].reference.text)

[<Fact>]
let ``keyword elaboration generator attaches generated terms before tfidf build`` () =
    async {
        let passages =
            [ { sourceId = "one"
                sourceDisplayName = "One"
                sourceLocation = "/tmp/one.md"
                index = 0
                text = "benefits waiting period details"
                sectionPath = []
                contentRole = PassageContentRole.Unknown
                pageNumbers = []
                layoutLabels = []
                captions = []
                keywords = [] } ]

        let generator =
            StaticKeywordGenerator(
                [ { sourceId = "one"
                    passageIndex = 0
                    keywords = [ "orthodontia"; "dental braces" ] } ]
            )

        let! elaborated = IndexBuilder.elaborateKeywords (KeywordElaborationOptions.withGenerator generator) passages

        let indexed =
            elaborated
            |> List.map (fun passage ->
                { reference = passage
                  embedding = vector [| 1 |] [| 1.0f; 0.0f |]
                  terms = Text.terms passage.text })

        let tfidf = Tfidf.buildWithOptions TfidfOptions.defaults indexed
        let candidates = Tfidf.scoreQuery tfidf "orthodontia" |> Tfidf.topCandidates 10

        Assert.Equal<string list>([ "orthodontia"; "dental braces" ], elaborated.Head.keywords)
        Assert.Equal(0, candidates[0] |> fst)
    }
    |> Async.RunSynchronously

[<Fact>]
let ``queryEncoded reranks tfidf candidates with dense score and maxResults`` () =
    let idx =
        index
            [ passage "one" 0 "apple topic" (vector [| 1 |] [| 1.0f; 0.0f |])
              passage "two" 0 "apple topic" (vector [| 1 |] [| 0.0f; 1.0f |])
              passage "three" 0 "orange topic" (vector [| 1 |] [| 1.0f; 0.0f |]) ]

    let options =
        { SearchOptions.defaults with
            maxResults = 1
            candidateLimit = 0
            denseWeight = 1.0f
            lexicalWeight = 0.0f }

    let queryEmbedding = vector [| 1 |] [| 0.0f; 1.0f |]
    let hits = Search.queryEncoded options idx "apple" queryEmbedding

    Assert.Single hits |> ignore
    Assert.Equal("two", hits.Head.reference.sourceId)

[<Fact>]
let ``queryEncodedWithSearchTerms uses supplied terms for candidates but original embedding for dense score`` () =
    let idx =
        index
            [ passage "one" 0 "automobile topic" (vector [| 1 |] [| 0.0f; 1.0f |])
              passage "two" 0 "kitchen topic" (vector [| 1 |] [| 1.0f; 0.0f |]) ]

    let options =
        { SearchOptions.defaults with
            maxResults = 1
            candidateLimit = 10
            denseWeight = 1.0f
            lexicalWeight = 0.0f }

    let queryEmbedding = vector [| 1 |] [| 0.0f; 1.0f |]
    let literalHits = Search.queryEncoded options idx "car" queryEmbedding

    let expandedHits =
        Search.queryEncodedWithSearchTerms options idx "car" [ "automobile" ] queryEmbedding

    Assert.Empty literalHits
    Assert.Single expandedHits |> ignore
    Assert.Equal("one", expandedHits.Head.reference.sourceId)

[<Fact>]
let ``persistence round trips an index`` () =
    let passage =
        passageWithMetadata
            "pdf-1"
            0
            "The system indexes local PDF passages."
            [ "Guide"; "Claims" ]
            PassageContentRole.MainBody
            [ 2; 3 ]
            [ "insurance claims"; "policy support" ]
            (vector [| 1; 2 |] [| 1.0f; 0.0f; 0.0f; 1.0f |])

    let passage =
        { passage with
            reference =
                { passage.reference with
                    layoutLabels = [ "section_header"; "caption" ]
                    captions = [ "Figure 1: Claim workflow." ] } }

    let index = index [ passage ]

    let path = IO.Path.Combine(IO.Path.GetTempPath(), $"{Guid.NewGuid():N}.fsci")

    try
        IndexPersistence.save path index
        let loaded = IndexPersistence.load path

        Assert.Equal(1, loaded.passages.Length)
        Assert.Equal("PDF: pdf-1", loaded.passages.Head.reference.sourceDisplayName)
        Assert.Equal<string list>([ "insurance claims"; "policy support" ], loaded.passages.Head.reference.keywords)
        Assert.Equal<string list>([ "Guide"; "Claims" ], loaded.passages.Head.reference.sectionPath)
        Assert.Equal(PassageContentRole.MainBody, loaded.passages.Head.reference.contentRole)
        Assert.Equal<int list>([ 2; 3 ], loaded.passages.Head.reference.pageNumbers)
        Assert.Equal<string list>([ "section_header"; "caption" ], loaded.passages.Head.reference.layoutLabels)
        Assert.Equal<string list>([ "Figure 1: Claim workflow." ], loaded.passages.Head.reference.captions)
        Assert.Equal(TfidfOptions.defaults.keywordWeight, loaded.tfidfOptions.keywordWeight)
        Assert.Equal<float32 array>(passage.embedding.vectors, loaded.passages.Head.embedding.vectors)
        Assert.Equal(1, loaded.tfidf.passageCount)
        Assert.True(loaded.tfidf.vocabulary.ContainsKey "system")
        Assert.True(loaded.tfidf.vocabulary.ContainsKey "claims")
        Assert.Equal(1, loaded.tfidf.vocabulary["system"].documentFrequency)
        Assert.Equal(0, loaded.tfidf.vocabulary["system"].postings[0].passageOrdinal)
    finally
        if IO.File.Exists path then
            IO.File.Delete path

[<Fact>]
let ``persistence loads version 2 indexes with empty keywords`` () =
    let oldIndex =
        index
            [ passage
                  "old-pdf"
                  0
                  "The prior index format had only raw text."
                  (vector [| 1; 2 |] [| 1.0f; 0.0f; 0.0f; 1.0f |]) ]

    let path = IO.Path.Combine(IO.Path.GetTempPath(), $"{Guid.NewGuid():N}.fsci")

    try
        writeVersion2Index path oldIndex
        let loaded = IndexPersistence.load path

        Assert.Equal(1, loaded.passages.Length)
        Assert.Empty loaded.passages.Head.reference.keywords
        Assert.Empty loaded.passages.Head.reference.sectionPath
        Assert.Equal(PassageContentRole.Unknown, loaded.passages.Head.reference.contentRole)
        Assert.Empty loaded.passages.Head.reference.pageNumbers
        Assert.Empty loaded.passages.Head.reference.layoutLabels
        Assert.Empty loaded.passages.Head.reference.captions
        Assert.Equal(TfidfOptions.defaults.textWeight, loaded.tfidfOptions.textWeight)
        Assert.Equal(TfidfOptions.defaults.keywordWeight, loaded.tfidfOptions.keywordWeight)
        Assert.True(loaded.tfidf.vocabulary.ContainsKey "prior")
    finally
        if IO.File.Exists path then
            IO.File.Delete path

[<Fact>]
let ``persistence loads version 3 indexes with empty section path`` () =
    let oldIndex =
        index
            [ passageWithKeywords
                  "old-pdf"
                  0
                  "The version three index format had text and keywords."
                  [ "legacy keyword" ]
                  (vector [| 1; 2 |] [| 1.0f; 0.0f; 0.0f; 1.0f |]) ]

    let path = IO.Path.Combine(IO.Path.GetTempPath(), $"{Guid.NewGuid():N}.fsci")

    try
        writeVersion3Index path oldIndex
        let loaded = IndexPersistence.load path

        Assert.Equal<string list>([ "legacy keyword" ], loaded.passages.Head.reference.keywords)
        Assert.Empty loaded.passages.Head.reference.sectionPath
        Assert.Equal(PassageContentRole.Unknown, loaded.passages.Head.reference.contentRole)
        Assert.Empty loaded.passages.Head.reference.pageNumbers
        Assert.Empty loaded.passages.Head.reference.layoutLabels
        Assert.Empty loaded.passages.Head.reference.captions
        Assert.True(loaded.tfidf.vocabulary.ContainsKey "legacy")
    finally
        if IO.File.Exists path then
            IO.File.Delete path

[<Fact>]
let ``persistence loads version 4 indexes with empty content role and pages`` () =
    let oldIndex =
        index
            [ passageWithSection
                  "old-pdf"
                  0
                  "The version four index format had section context but no role metadata."
                  [ "Paper"; "Abstract" ]
                  [ "abstract" ]
                  (vector [| 1; 2 |] [| 1.0f; 0.0f; 0.0f; 1.0f |]) ]

    let path = IO.Path.Combine(IO.Path.GetTempPath(), $"{Guid.NewGuid():N}.fsci")

    try
        writeVersion4Index path oldIndex
        let loaded = IndexPersistence.load path

        Assert.Equal<string list>([ "Paper"; "Abstract" ], loaded.passages.Head.reference.sectionPath)
        Assert.Equal(PassageContentRole.Unknown, loaded.passages.Head.reference.contentRole)
        Assert.Empty loaded.passages.Head.reference.pageNumbers
        Assert.Empty loaded.passages.Head.reference.layoutLabels
        Assert.Empty loaded.passages.Head.reference.captions
        Assert.True(loaded.tfidf.vocabulary.ContainsKey "role")
    finally
        if IO.File.Exists path then
            IO.File.Delete path

[<Fact>]
let ``persistence loads version 5 indexes with empty layout metadata`` () =
    let oldIndex =
        index
            [ passageWithMetadata
                  "old-pdf"
                  0
                  "The version five index format had role and page metadata."
                  [ "Paper"; "Results" ]
                  PassageContentRole.MainBody
                  [ 5; 6 ]
                  [ "results" ]
                  (vector [| 1; 2 |] [| 1.0f; 0.0f; 0.0f; 1.0f |]) ]

    let path = IO.Path.Combine(IO.Path.GetTempPath(), $"{Guid.NewGuid():N}.fsci")

    try
        writeVersion5Index path oldIndex
        let loaded = IndexPersistence.load path

        Assert.Equal<string list>([ "Paper"; "Results" ], loaded.passages.Head.reference.sectionPath)
        Assert.Equal(PassageContentRole.MainBody, loaded.passages.Head.reference.contentRole)
        Assert.Equal<int list>([ 5; 6 ], loaded.passages.Head.reference.pageNumbers)
        Assert.Empty loaded.passages.Head.reference.layoutLabels
        Assert.Empty loaded.passages.Head.reference.captions
        Assert.True(loaded.tfidf.vocabulary.ContainsKey "version")
    finally
        if IO.File.Exists path then
            IO.File.Delete path

type private StaticDoclingLayoutPredictor(predictions: DoclingLayoutPrediction list) =
    interface IDoclingLayoutPredictor with
        member _.PredictLayoutAsync pages =
            async {
                let requested = pages |> List.map _.pageNo |> Set.ofList

                return
                    predictions
                    |> List.filter (fun prediction -> requested.Contains prediction.pageNo)
                    |> Ok
            }

type private CountingCancelableDoclingLayoutPredictor(predictions: DoclingLayoutPrediction list) =
    let mutable cancelableCalls = 0
    let mutable legacyCalls = 0

    member _.CancelableCalls = cancelableCalls

    member _.LegacyCalls = legacyCalls

    interface IDoclingLayoutPredictor with
        member _.PredictLayoutAsync _ =
            async {
                legacyCalls <- legacyCalls + 1
                return Ok predictions
            }

    interface ICancelableDoclingLayoutPredictor with
        member _.PredictLayoutAsync(pages, cancellationToken) =
            async {
                cancellationToken.ThrowIfCancellationRequested()
                cancelableCalls <- cancelableCalls + 1
                let requested = pages |> List.map _.pageNo |> Set.ofList

                return
                    predictions
                    |> List.filter (fun prediction -> requested.Contains prediction.pageNo)
                    |> Ok
            }

type private StaticDoclingFigureClassifier(classes: DoclingFigureClass list) =
    interface IDoclingFigureClassifier with
        member _.ClassifyAsync _ = async { return Ok classes }

let private ocr text l t r b =
    { text = text
      bbox = DoclingGeometry.topLeftBox l t r b
      confidence = Some 0.99 }

let private nativeCell text l bottom r top =
    { text = text
      bbox = DoclingGeometry.bottomLeftBox l bottom r top
      confidence = None }

let private cluster id label l t r b =
    { id = id
      label = label
      confidence = 0.95f
      bbox = DoclingGeometry.topLeftBox l t r b
      cells = [] }

[<Fact>]
let ``docling geometry converts origins and computes overlap`` () =
    let bottomLeft = DoclingGeometry.bottomLeftBox 10.0 10.0 60.0 30.0
    let topLeft = DoclingGeometry.toTopLeft 100.0 bottomLeft

    Assert.Equal(70.0, topLeft.t, 3)
    Assert.Equal(90.0, topLeft.b, 3)
    Assert.Equal(TopLeft, topLeft.coordOrigin)

    let cell = DoclingGeometry.topLeftBox 10.0 10.0 30.0 30.0
    let region = DoclingGeometry.topLeftBox 20.0 10.0 40.0 30.0

    Assert.Equal(0.5, DoclingGeometry.intersectionOverSelf cell region, 3)

[<Fact>]
let ``docling cells scale native coordinates and prefer native over overlapping ocr`` () =
    let image = DoclingRgbImage.solid 400 800 255uy 255uy 255uy
    let native = [ nativeCell "Native text" 10.0 320.0 120.0 340.0 ]

    let ocrCells =
        [ ocr "OCR text" 20.0 122.0 130.0 156.0
          ocr "Other line" 20.0 200.0 130.0 222.0 ]

    let scaledNative =
        DoclingCells.scaleCellsToImage { width = 200.0; height = 400.0 } image native

    let topLeftNative = DoclingGeometry.toTopLeft 800.0 scaledNative.Head.bbox

    Assert.Equal(20.0, topLeftNative.l, 3)
    Assert.Equal(120.0, topLeftNative.t, 3)
    Assert.Equal(240.0, topLeftNative.r, 3)
    Assert.Equal(160.0, topLeftNative.b, 3)

    let merged = DoclingCells.mergePreferPrimary 800.0 0.7 scaledNative ocrCells

    Assert.Equal<string list>([ "Native text"; "Other line" ], merged |> List.map _.text)
    Assert.True(DoclingCells.hasEnoughText 10 merged)

[<Fact>]
let ``standard hybrid assembles native cells without full page ocr`` () =
    async {
        let image = DoclingRgbImage.solid 400 400 255uy 255uy 255uy

        let page =
            { pageNo = 1
              image = image
              ocrCells =
                [ nativeCell "Native" 20.0 270.0 80.0 295.0
                  nativeCell "PDF" 90.0 270.0 135.0 295.0
                  nativeCell "text" 145.0 270.0 190.0 295.0 ] }

        let predictions =
            [ { pageNo = 1
                clusters = [ cluster 0 Text 15.0 95.0 210.0 140.0 ] } ]

        let layout = StaticDoclingLayoutPredictor predictions :> IDoclingLayoutPredictor

        let! result = DoclingStandardHybrid.convertPages "native" (Some "native.pdf") layout None [ page ]

        match result with
        | Error err -> failwith err
        | Ok document ->
            Assert.Single document.texts |> ignore
            Assert.Equal("Native PDF text", document.texts.Head.text)
    }
    |> Async.RunSynchronously

[<Fact>]
let ``standard hybrid honors pre-canceled conversion token before predictor work`` () =
    let image = DoclingRgbImage.solid 400 400 255uy 255uy 255uy

    let page =
        { pageNo = 1
          image = image
          ocrCells = [ nativeCell "Native text" 20.0 270.0 120.0 295.0 ] }

    let layout =
        CountingCancelableDoclingLayoutPredictor
            [ { pageNo = 1
                clusters = [ cluster 0 Text 15.0 95.0 210.0 140.0 ] } ]

    use cts = new CancellationTokenSource()
    cts.Cancel()

    Assert.Throws<OperationCanceledException>(fun () ->
        DoclingStandardHybrid.convertPagesWithOptionsWithCancellation
            DoclingConversionOptions.defaults
            "native"
            (Some "native.pdf")
            (layout :> IDoclingLayoutPredictor)
            None
            [ page ]
            cts.Token
        |> Async.RunSynchronously
        |> ignore)
    |> ignore

    Assert.Equal(0, layout.CancelableCalls)
    Assert.Equal(0, layout.LegacyCalls)

[<Fact>]
let ``standard hybrid uses cancelable predictor overload when available`` () =
    async {
        let image = DoclingRgbImage.solid 400 400 255uy 255uy 255uy

        let page =
            { pageNo = 1
              image = image
              ocrCells = [ nativeCell "Native text" 20.0 270.0 120.0 295.0 ] }

        let layout =
            CountingCancelableDoclingLayoutPredictor
                [ { pageNo = 1
                    clusters = [ cluster 0 Text 15.0 95.0 210.0 140.0 ] } ]

        let! result =
            DoclingStandardHybrid.convertPagesWithOptionsWithCancellation
                DoclingConversionOptions.defaults
                "native"
                (Some "native.pdf")
                (layout :> IDoclingLayoutPredictor)
                None
                [ page ]
                CancellationToken.None

        match result with
        | Error err -> failwith err
        | Ok _ ->
            Assert.Equal(1, layout.CancelableCalls)
            Assert.Equal(0, layout.LegacyCalls)
    }
    |> Async.RunSynchronously

[<Fact>]
let ``index builder honors pre-canceled token before saving index`` () =
    use cts = new CancellationTokenSource()
    cts.Cancel()

    Assert.Throws<OperationCanceledException>(fun () ->
        IndexBuilder.createFromPassagesWithCancellation
            Unchecked.defaultof<OnnxColbertEncoder>
            ChunkOptions.fsKameDefaults
            []
            None
            cts.Token
        |> Async.RunSynchronously
        |> ignore)
    |> ignore

[<Fact>]
let ``standard hybrid builds docling json with reading order tables and pictures`` () =
    async {
        let image = DoclingRgbImage.solid 400 400 255uy 255uy 255uy

        let page =
            { pageNo = 1
              image = image
              ocrCells =
                [ ocr "Confidential" 10.0 8.0 120.0 24.0
                  ocr "Quarterly Results" 20.0 80.0 220.0 110.0
                  ocr "Revenue grew quickly." 20.0 125.0 250.0 150.0
                  ocr "Q1 100" 30.0 205.0 100.0 225.0
                  ocr "Q2 120" 30.0 230.0 100.0 250.0 ] }

        let predictions =
            [ { pageNo = 1
                clusters =
                  [ cluster 0 PageHeader 0.0 0.0 400.0 40.0
                    cluster 1 Text 20.0 75.0 280.0 160.0
                    cluster 2 Table 20.0 195.0 180.0 265.0
                    cluster 3 Picture 250.0 190.0 360.0 300.0 ] } ]

        let layout = StaticDoclingLayoutPredictor predictions :> IDoclingLayoutPredictor

        let classifier =
            StaticDoclingFigureClassifier
                [ { className = "bar_chart"
                    confidence = 0.91f } ]
            :> IDoclingFigureClassifier

        let! result =
            DoclingStandardHybrid.convertPages "quarterly" (Some "quarterly.pdf") layout (Some classifier) [ page ]

        match result with
        | Error err -> failwith err
        | Ok document ->
            Assert.Empty(DoclingJson.validateSubset document)
            Assert.Equal(1, document.pages.Count)
            Assert.Equal(2, document.texts.Length)
            Assert.Single document.tables |> ignore
            Assert.Single document.pictures |> ignore
            Assert.Equal<string list>([ "#/texts/1"; "#/tables/0"; "#/pictures/0" ], document.bodyChildren)
            Assert.Equal<string list>([ "#/texts/0" ], document.furnitureChildren)
            Assert.Equal("Confidential", document.texts[0].text)
            Assert.Equal("Quarterly Results Revenue grew quickly.", document.texts[1].text)
            Assert.Equal(2, document.tables[0].data.numRows)
            Assert.Equal("Q1 100", document.tables[0].data.tableCells[0].text)
            Assert.Equal("bar_chart", document.pictures[0].classifications[0].className)
            Assert.Equal<string list>([ "table" ], document.tables[0].keywords)
            Assert.Equal<string list>([ "picture"; "bar_chart" ], document.pictures[0].keywords)

            let json = DoclingJson.serialize document
            use parsed = System.Text.Json.JsonDocument.Parse json
            let root = parsed.RootElement

            Assert.Equal("DoclingDocument", root.GetProperty("schema_name").GetString())
            Assert.Equal("1.10.0", root.GetProperty("version").GetString())
            let mutable pageElement = Unchecked.defaultof<System.Text.Json.JsonElement>
            Assert.True(root.GetProperty("pages").TryGetProperty("1", &pageElement))
            Assert.Equal("quarterly", root.GetProperty("name").GetString())
            Assert.Equal("table", root.GetProperty("tables").[0].GetProperty("label").GetString())

            Assert.Equal(
                "classification",
                root.GetProperty("pictures").[0].GetProperty("annotations").[0].GetProperty("kind").GetString()
            )

            Assert.Equal(
                "table",
                root
                    .GetProperty("tables")
                    .[0].GetProperty("meta")
                    .GetProperty("fscolbert")
                    .GetProperty("keywords")
                    .[0].GetString()
            )

            Assert.Equal(
                "quarterly",
                root
                    .GetProperty("pictures")
                    .[0].GetProperty("meta")
                    .GetProperty("fscolbert")
                    .GetProperty("source_id")
                    .GetString()
            )

            match DoclingJson.tryDeserialize json with
            | Error err -> failwith err
            | Ok roundTripped ->
                Assert.Equal<string list>([ "picture"; "bar_chart" ], roundTripped.pictures[0].keywords)
                Assert.Equal(Some "quarterly", roundTripped.texts[0].sourceId)
    }
    |> Async.RunSynchronously

[<Fact>]
let ``docling passages render body text table rows and picture metadata`` () =
    let table =
        { selfRef = "#/tables/0"
          parent = "#/body"
          label = Table
          contentLayer = Body
          prov =
            [ { pageNo = 1
                bbox = DoclingGeometry.topLeftBox 0.0 0.0 100.0 100.0
                charSpan = Some(0, 0) } ]
          keywords = [ "revenue table" ]
          sourceId = Some "doc"
          sourceDisplayName = Some "Doc"
          data =
            { numRows = 2
              numCols = 1
              tableCells =
                [ { text = "Alpha"
                    bbox = None
                    startRowOffsetIndex = 0
                    endRowOffsetIndex = 1
                    startColOffsetIndex = 0
                    endColOffsetIndex = 1
                    rowHeader = false
                    columnHeader = false
                    rowSection = false }
                  { text = "Beta"
                    bbox = None
                    startRowOffsetIndex = 1
                    endRowOffsetIndex = 2
                    startColOffsetIndex = 0
                    endColOffsetIndex = 1
                    rowHeader = false
                    columnHeader = false
                    rowSection = false } ] } }

    let document =
        { name = "doc"
          originFileName = None
          originMimeType = None
          pages =
            [ 1,
              { pageNo = 1
                size = { width = 100.0; height = 100.0 } } ]
            |> Map.ofList
          texts =
            [ { selfRef = "#/texts/0"
                parent = "#/body"
                label = Text
                text = "Intro text"
                orig = "Intro text"
                contentLayer = Body
                prov = []
                keywords = [ "executive summary" ]
                sourceId = Some "doc"
                sourceDisplayName = Some "Doc" }
              { selfRef = "#/texts/1"
                parent = "#/body"
                label = Caption
                text = "Table 1: Revenue by quarter."
                orig = "Table 1: Revenue by quarter."
                contentLayer = Body
                prov = []
                keywords = []
                sourceId = Some "doc"
                sourceDisplayName = Some "Doc" } ]
          tables = [ table ]
          pictures =
            [ { selfRef = "#/pictures/0"
                parent = "#/body"
                label = Picture
                contentLayer = Body
                prov = []
                classifications =
                  [ { className = "photograph"
                      confidence = 0.8f } ]
                keywords = [ "product photo" ]
                sourceId = Some "doc"
                sourceDisplayName = Some "Doc" } ]
          bodyChildren = [ "#/texts/0"; "#/texts/1"; "#/tables/0"; "#/pictures/0" ]
          furnitureChildren = [] }

    let blocks = DoclingPassages.toBlocks document

    Assert.Equal<string list>(
        [ "Intro text"
          "Table 1: Revenue by quarter."
          "Alpha Beta"
          "[Picture: photograph (0.800)]" ],
        blocks
    )

    let source = PassageSource.create "doc" "Doc" "/tmp/doc.pdf"

    let passages =
        DoclingPassages.toPassages ChunkOptions.fsKameDefaults source document

    let blocksWithKeywords = DoclingPassages.toBlocksWithKeywords document

    Assert.Equal<string list>(
        [ "executive summary"
          "caption"
          "Table 1: Revenue by quarter."
          "revenue table"
          "table"
          "product photo"
          "picture" ],
        blocksWithKeywords |> List.collect _.keywords
    )

    Assert.Equal("doc", passages.Head.sourceId)
    Assert.Contains("Intro text", passages.Head.text)
    Assert.Contains("executive summary", passages.Head.keywords)
    Assert.Contains("caption", passages.Head.layoutLabels)
    Assert.Contains("table", passages.Head.layoutLabels)
    Assert.Contains("picture", passages.Head.layoutLabels)
    Assert.Contains("Table 1: Revenue by quarter.", passages.Head.captions)
    Assert.Contains("revenue table", passages.Head.keywords)
    Assert.Contains("product photo", passages.Head.keywords)

[<Fact>]
let ``docling passages inherit structured section context`` () =
    let table =
        { selfRef = "#/tables/0"
          parent = "#/body"
          label = Table
          contentLayer = Body
          prov = []
          keywords = []
          sourceId = Some "doc"
          sourceDisplayName = Some "Doc"
          data =
            { numRows = 1
              numCols = 1
              tableCells =
                [ { text = "Result row"
                    bbox = None
                    startRowOffsetIndex = 0
                    endRowOffsetIndex = 1
                    startColOffsetIndex = 0
                    endColOffsetIndex = 1
                    rowHeader = false
                    columnHeader = false
                    rowSection = false } ] } }

    let picture =
        { selfRef = "#/pictures/0"
          parent = "#/body"
          label = Picture
          contentLayer = Body
          prov = []
          classifications = []
          keywords = []
          sourceId = Some "doc"
          sourceDisplayName = Some "Doc" }

    let textItem selfRef label text =
        { selfRef = selfRef
          parent = "#/body"
          label = label
          text = text
          orig = text
          contentLayer = Body
          prov = []
          keywords = []
          sourceId = Some "doc"
          sourceDisplayName = Some "Doc" }

    let document =
        { name = "doc"
          originFileName = None
          originMimeType = None
          pages =
            [ 1,
              { pageNo = 1
                size = { width = 100.0; height = 100.0 } } ]
            |> Map.ofList
          texts =
            [ textItem "#/texts/0" Title "A-Mem: Agentic Memory for LLM Agents"
              textItem "#/texts/1" SectionHeader "4 Experiment"
              textItem "#/texts/2" Text "Dataset description."
              textItem "#/texts/3" SectionHeader "4.3 Empirical Results"
              textItem "#/texts/4" Text "Performance analysis without repeating the heading."
              textItem "#/texts/5" SectionHeader "Limitations"
              textItem "#/texts/6" Text "Model sensitivity discussion." ]
          tables = [ table ]
          pictures = [ picture ]
          bodyChildren =
            [ "#/texts/0"
              "#/texts/1"
              "#/texts/2"
              "#/texts/3"
              "#/texts/4"
              "#/tables/0"
              "#/pictures/0"
              "#/texts/5"
              "#/texts/6" ]
          furnitureChildren = [] }

    let source = PassageSource.create "doc" "Doc" "/tmp/doc.pdf"

    let passages =
        DoclingPassages.toPassages ChunkOptions.fsKameDefaults source document

    let empirical =
        passages
        |> List.find (fun passage -> passage.text.Contains("Performance analysis"))

    Assert.Equal<string list>(
        [ "A-Mem: Agentic Memory for LLM Agents"
          "4 Experiment"
          "4.3 Empirical Results" ],
        empirical.sectionPath
    )

    Assert.Contains("4.3 Empirical Results", empirical.keywords)

    let tablePassage =
        passages |> List.find (fun passage -> passage.text.Contains("Result row"))

    Assert.Equal<string list>(empirical.sectionPath, tablePassage.sectionPath)

    let limitations =
        passages
        |> List.find (fun passage -> passage.text.Contains("Model sensitivity"))

    Assert.Equal<string list>([ "A-Mem: Agentic Memory for LLM Agents"; "Limitations" ], limitations.sectionPath)

[<Fact>]
let ``docling passages infer content roles and page numbers`` () =
    let prov pageNo =
        [ { pageNo = pageNo
            bbox = DoclingGeometry.topLeftBox 0.0 0.0 100.0 20.0
            charSpan = None } ]

    let textItem selfRef label text pageNo =
        { selfRef = selfRef
          parent = "#/body"
          label = label
          text = text
          orig = text
          contentLayer = Body
          prov = prov pageNo
          keywords = []
          sourceId = Some "doc"
          sourceDisplayName = Some "Doc" }

    let document: DoclingDocument =
        { name = "doc"
          originFileName = None
          originMimeType = None
          pages =
            [ for pageNo in 1..6 ->
                  pageNo,
                  { pageNo = pageNo
                    size = { width = 100.0; height = 100.0 } } ]
            |> Map.ofList
          texts =
            [ textItem "#/texts/0" Title "A-Mem: Agentic Memory for LLM Agents" 1
              textItem "#/texts/1" Text "Wujiang Xu, Zujie Liang, Juntao Tan" 1
              textItem "#/texts/2" SectionHeader "Abstract" 1
              textItem "#/texts/3" Text "This paper introduces agentic memory." 1
              textItem "#/texts/4" SectionHeader "1 Introduction" 2
              textItem "#/texts/5" Text "The introduction motivates memory systems." 2
              textItem "#/texts/6" SectionHeader "References" 4
              textItem "#/texts/7" Text "Beltagy et al. Longformer." 4
              textItem "#/texts/8" SectionHeader "Appendix A Extra Details" 5
              textItem "#/texts/9" Text "Supplementary experiment notes." 5
              textItem "#/texts/10" SectionHeader "NeurIPS Paper Checklist" 6
              textItem
                  "#/texts/11"
                  Text
                  "Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions? Answer: [Yes] Guidelines: Authors should answer carefully."
                  6 ]
          tables = []
          pictures = []
          bodyChildren = [ for index in 0..11 -> $"#/texts/{index}" ]
          furnitureChildren = [] }

    let source = PassageSource.create "doc" "Doc" "/tmp/doc.pdf"

    let passages =
        DoclingPassages.toPassages ChunkOptions.fsKameDefaults source document

    let frontMatter =
        passages |> List.find (fun passage -> passage.text.Contains("Wujiang Xu"))

    let abstractPassage =
        passages |> List.find (fun passage -> passage.text.Contains("agentic memory"))

    let body =
        passages |> List.find (fun passage -> passage.text.Contains("motivates memory"))

    let references =
        passages |> List.find (fun passage -> passage.text.Contains("Longformer"))

    let appendix =
        passages |> List.find (fun passage -> passage.text.Contains("Supplementary"))

    let checklist =
        passages
        |> List.find (fun passage -> passage.text.Contains("Question: Do the main claims"))

    Assert.Equal(PassageContentRole.FrontMatter, frontMatter.contentRole)
    Assert.Equal<int list>([ 1 ], frontMatter.pageNumbers)
    Assert.Equal(PassageContentRole.Abstract, abstractPassage.contentRole)
    Assert.Equal<int list>([ 1 ], abstractPassage.pageNumbers)
    Assert.Equal(PassageContentRole.MainBody, body.contentRole)
    Assert.Equal<int list>([ 2 ], body.pageNumbers)
    Assert.Equal(PassageContentRole.References, references.contentRole)
    Assert.Equal(PassageContentRole.Appendix, appendix.contentRole)
    Assert.Equal(PassageContentRole.SubmissionChecklist, checklist.contentRole)
    Assert.Equal<int list>([ 6 ], checklist.pageNumbers)

[<Fact>]
let ``content role inference treats checklist fragments as submission checklist`` () =
    let sectionPath = [ "A-Mem: Agentic Memory for LLM Agents"; "B.4 Examples of Q/A with A-MEM" ]

    let fragments =
        [ "The answer NA means that the paper has no limitation while the answer No means that the paper has limitations."
          "Justification: Both code and datasets are available."
          "At submission time, remember to anonymize your assets if applicable." ]

    for fragment in fragments do
        Assert.Equal(PassageContentRole.SubmissionChecklist, DocumentContentRoles.infer sectionPath fragment)

[<Fact>]
let ``docling numbered heading jumps do not inherit unrelated parents`` () =
    let textItem selfRef label text =
        { selfRef = selfRef
          parent = "#/body"
          label = label
          text = text
          orig = text
          contentLayer = Body
          prov = []
          keywords = []
          sourceId = Some "doc"
          sourceDisplayName = Some "Doc" }

    let document: DoclingDocument =
        { name = "doc"
          originFileName = None
          originMimeType = None
          pages =
            [ 1,
              { pageNo = 1
                size = { width = 100.0; height = 100.0 } } ]
            |> Map.ofList
          texts =
            [ textItem "#/texts/0" Title "AI on the Pulse"
              textItem "#/texts/1" SectionHeader "1 INTRODUCTION"
              textItem "#/texts/2" Text "Introductory body."
              textItem "#/texts/3" SectionHeader "4.3 Experimental Setup"
              textItem "#/texts/4" Text "Experimental body." ]
          tables = []
          pictures = []
          bodyChildren = [ "#/texts/0"; "#/texts/1"; "#/texts/2"; "#/texts/3"; "#/texts/4" ]
          furnitureChildren = [] }

    let source = PassageSource.create "doc" "Doc" "/tmp/doc.pdf"

    let passages =
        DoclingPassages.toPassages ChunkOptions.fsKameDefaults source document

    let experimental =
        passages
        |> List.find (fun passage -> passage.text.Contains("Experimental body."))

    Assert.Equal<string list>([ "AI on the Pulse"; "4.3 Experimental Setup" ], experimental.sectionPath)

[<Fact>]
let ``section path keywords participate in tfidf retrieval`` () =
    let idx =
        index
            [ passageWithSection
                  "doc"
                  0
                  "Performance analysis without repeating the heading."
                  [ "A-Mem: Agentic Memory for LLM Agents"
                    "4 Experiment"
                    "4.3 Empirical Results" ]
                  [ "A-Mem: Agentic Memory for LLM Agents"
                    "4 Experiment"
                    "4.3 Empirical Results" ]
                  (vector [| 1 |] [| 1.0f; 0.0f |]) ]

    let candidates =
        Tfidf.scoreQuery idx.tfidf "empirical results" |> Tfidf.topCandidates 10

    Assert.Single candidates |> ignore
    Assert.Equal(0, fst candidates[0])

[<Fact>]
let ``contextual text includes document role section pages and raw text`` () =
    let passage =
        { sourceId = "doc"
          sourceDisplayName = "A-Mem Paper"
          sourceLocation = "/tmp/amem.pdf"
          index = 3
          text = "LoCoMo and DialSim are the datasets."
          sectionPath = [ "A-Mem"; "4 Experiment"; "4.1 Datasets" ]
          contentRole = PassageContentRole.MainBody
          pageNumbers = [ 6; 7 ]
          layoutLabels = [ "section_header"; "caption" ]
          captions = [ "Table 1: Dataset statistics for LoCoMo and DialSim." ]
          keywords = [] }

    let contextual = PassageContext.contextualText passage

    Assert.Contains("Document: A-Mem Paper", contextual)
    Assert.Contains("Role: Main body", contextual)
    Assert.Contains("Section: A-Mem > 4 Experiment > 4.1 Datasets", contextual)
    Assert.Contains("Pages: 6, 7", contextual)
    Assert.Contains("Captions: Table 1: Dataset statistics for LoCoMo and DialSim.", contextual)
    Assert.Contains("LoCoMo and DialSim are the datasets.", contextual)

[<Fact>]
let ``deterministic context keywords rank paper authors above checklist boilerplate`` () =
    let enrich (passage: IndexedPassage) =
        let reference =
            { passage.reference with
                keywords =
                    passage.reference.keywords
                    @ PassageContext.deterministicKeywords passage.reference }

        { passage with reference = reference }

    let frontMatter =
        passageWithMetadata
            "amem"
            0
            "A-Mem: Agentic Memory for LLM Agents. Wujiang Xu, Zujie Liang, Juntao Tan."
            [ "A-Mem: Agentic Memory for LLM Agents" ]
            PassageContentRole.FrontMatter
            [ 1 ]
            []
            (vector [| 1 |] [| 1.0f; 0.0f |])

    let checklist =
        passageWithMetadata
            "amem"
            1
            "NeurIPS Paper Checklist. Guidelines: Authors should answer each question and ensure the paper satisfies the checklist."
            [ "A-Mem: Agentic Memory for LLM Agents"; "NeurIPS Paper Checklist" ]
            PassageContentRole.SubmissionChecklist
            [ 24 ]
            []
            (vector [| 1 |] [| 0.0f; 1.0f |])

    let idx = index [ enrich frontMatter; enrich checklist ]

    let candidates = Tfidf.scoreQuery idx.tfidf "paper authors" |> Tfidf.topCandidates 2

    Assert.Equal(0, fst candidates[0])

[<Fact>]
let ``docling derived keywords survive index persistence`` () =
    let source = PassageSource.create "doc" "Doc" "/tmp/doc.json"

    let document =
        { name = "doc"
          originFileName = None
          originMimeType = None
          pages =
            [ 1,
              { pageNo = 1
                size = { width = 100.0; height = 100.0 } } ]
            |> Map.ofList
          texts =
            [ { selfRef = "#/texts/0"
                parent = "#/body"
                label = Text
                text = "A passage about benefits and waiting periods."
                orig = "A passage about benefits and waiting periods."
                contentLayer = Body
                prov = []
                keywords = [ "orthodontia"; "waiting period" ]
                sourceId = Some "doc"
                sourceDisplayName = Some "Doc" } ]
          tables = []
          pictures = []
          bodyChildren = [ "#/texts/0" ]
          furnitureChildren = [] }

    let passage =
        DoclingPassages.toPassages ChunkOptions.fsKameDefaults source document
        |> List.head

    let indexed =
        { reference = passage
          embedding = vector [| 1 |] [| 1.0f; 0.0f |]
          terms = Text.terms passage.text }

    let idx = index [ indexed ]
    let path = IO.Path.Combine(IO.Path.GetTempPath(), $"{Guid.NewGuid():N}.fsci")

    try
        IndexPersistence.save path idx
        let loaded = IndexPersistence.load path

        Assert.Equal<string list>([ "orthodontia"; "waiting period" ], loaded.passages.Head.reference.keywords)
        Assert.Empty loaded.passages.Head.reference.sectionPath
        Assert.True(loaded.tfidf.vocabulary.ContainsKey "orthodontia")
    finally
        if IO.File.Exists path then
            IO.File.Delete path

[<Fact>]
let ``index bundle manifest loads compatible indexes and rejects mismatched model`` () =
    let folder =
        IO.Path.Combine(IO.Path.GetTempPath(), $"fscolbert-bundle-{Guid.NewGuid():N}")

    try
        IO.Directory.CreateDirectory folder |> ignore

        let indexPath = IO.Path.Combine(folder, "doc.fsci")
        let manifestPath = IO.Path.Combine(folder, "index-bundle.json")

        let idx =
            index [ passage "doc" 0 "Bundle passage text" (vector [| 1 |] [| 1.0f; 0.0f |]) ]

        IndexPersistence.save indexPath idx

        let source =
            { sourceId = "doc"
              sourceDisplayName = "Doc"
              sourceLocation = Some "doc.json"
              sourceKind = Some "docling-json"
              indexFile = "doc.fsci" }

        let manifest =
            IndexBundle.create
                "bundle"
                "1.0.0"
                ModelCatalog.mxbaiEdgeColbertInt8.id
                ChunkOptions.fsKameDefaults
                TfidfOptions.defaults
                [ source ]

        IndexBundle.writeManifest manifestPath manifest

        match IndexBundle.loadCompatible IndexBundleCompatibility.fsKameDefaults manifestPath with
        | Error errors -> failwith (String.concat "\n" errors)
        | Ok loaded ->
            Assert.Equal("bundle", loaded.manifest.bundleId)
            Assert.Single loaded.indexes |> ignore
            Assert.Equal("doc", loaded.indexes.Head.source.sourceId)
            Assert.Equal("doc.fsci", loaded.indexes.Head.source.indexFile)

        let incompatible =
            { IndexBundleCompatibility.fsKameDefaults with
                modelId = "different/model" }

        match IndexBundle.loadCompatible incompatible manifestPath with
        | Ok _ -> failwith "Expected bundle incompatibility."
        | Error errors -> Assert.Contains(errors, fun error -> error.Contains("model_id"))
    finally
        if IO.Directory.Exists folder then
            IO.Directory.Delete(folder, true)

[<Fact>]
let ``docling model catalog exposes layout and figure manifests`` () =
    Assert.Equal("docling-project/docling-layout-heron-onnx", ModelCatalog.doclingLayoutHeronOnnx.id)
    Assert.EndsWith("/model.onnx", ModelCatalog.doclingLayoutHeronOnnx.modelUrl)
    Assert.Equal("preprocessor_config.json", ModelCatalog.doclingLayoutHeronOnnx.preprocessorConfigFileName)

    Assert.Equal(
        "docling-project/DocumentFigureClassifier-v2.5",
        ModelCatalog.doclingDocumentFigureClassifierV25Onnx.id
    )

    Assert.EndsWith("/config.json", ModelCatalog.doclingDocumentFigureClassifierV25Onnx.configUrl)

let private writeModelManifestFiles folder (manifest: ModelManifest) =
    IO.Directory.CreateDirectory folder |> ignore
    IO.File.WriteAllText(IO.Path.Combine(folder, manifest.modelFileName), "model")
    IO.File.WriteAllText(IO.Path.Combine(folder, manifest.tokenizerFileName), "tokenizer")
    IO.File.WriteAllText(IO.Path.Combine(folder, manifest.configFileName), "{}")

[<Fact>]
let ``model catalog resolves local model files from complete candidate folder`` () =
    let root =
        IO.Path.Combine(IO.Path.GetTempPath(), $"fscolbert-model-local-{Guid.NewGuid():N}")

    try
        let incomplete = IO.Path.Combine(root, "incomplete")
        let complete = IO.Path.Combine(root, "complete")

        IO.Directory.CreateDirectory incomplete |> ignore
        IO.File.WriteAllText(IO.Path.Combine(incomplete, ModelCatalog.mxbaiEdgeColbertInt8.modelFileName), "model")

        writeModelManifestFiles complete ModelCatalog.mxbaiEdgeColbertInt8

        match ModelCatalog.tryResolveLocalFiles [ incomplete; complete ] ModelCatalog.mxbaiEdgeColbertInt8 with
        | None -> failwith "Expected local model files to resolve."
        | Some files ->
            Assert.Equal(IO.Path.Combine(complete, ModelCatalog.mxbaiEdgeColbertInt8.modelFileName), files.modelPath)

            Assert.Equal(
                IO.Path.Combine(complete, ModelCatalog.mxbaiEdgeColbertInt8.tokenizerFileName),
                files.tokenizerPath
            )

            Assert.Equal(
                Some(IO.Path.Combine(complete, ModelCatalog.mxbaiEdgeColbertInt8.configFileName)),
                files.configPath
            )
    finally
        if IO.Directory.Exists root then
            IO.Directory.Delete(root, true)

[<Fact>]
let ``model catalog ignores incomplete local model folder`` () =
    let folder =
        IO.Path.Combine(IO.Path.GetTempPath(), $"fscolbert-model-incomplete-{Guid.NewGuid():N}")

    try
        IO.Directory.CreateDirectory folder |> ignore
        IO.File.WriteAllText(IO.Path.Combine(folder, ModelCatalog.mxbaiEdgeColbertInt8.modelFileName), "model")

        Assert.True((ModelCatalog.tryResolveLocalFiles [ folder ] ModelCatalog.mxbaiEdgeColbertInt8).IsNone)
    finally
        if IO.Directory.Exists folder then
            IO.Directory.Delete(folder, true)

[<Fact>]
let ``model catalog ensure available avoids HTTP when local files exist`` () =
    let root =
        IO.Path.Combine(IO.Path.GetTempPath(), $"fscolbert-model-available-{Guid.NewGuid():N}")

    try
        let localFolder = IO.Path.Combine(root, "local")
        let downloadFolder = IO.Path.Combine(root, "download")
        writeModelManifestFiles localFolder ModelCatalog.mxbaiEdgeColbertInt8

        use client = new HttpClient(new FailingHttpMessageHandler())

        let files =
            ModelCatalog.ensureAvailableAsync client downloadFolder [ localFolder ] ModelCatalog.mxbaiEdgeColbertInt8
            |> Async.RunSynchronously

        Assert.Equal(IO.Path.Combine(localFolder, ModelCatalog.mxbaiEdgeColbertInt8.modelFileName), files.modelPath)
        Assert.False(IO.Directory.Exists downloadFolder)
    finally
        if IO.Directory.Exists root then
            IO.Directory.Delete(root, true)
