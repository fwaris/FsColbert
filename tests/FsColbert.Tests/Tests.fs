module FsColbert.Tests

open System
open System.IO
open System.Numerics
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
          keywords = keywords }
      embedding = embedding
      terms = Text.terms text }

let private passage sourceId index text embedding =
    passageWithKeywords sourceId index text [] embedding

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
                keywords = [] } ]

        let generator =
            StaticKeywordGenerator(
                [ { sourceId = "one"
                    passageIndex = 0
                    keywords = [ "orthodontia"; "dental braces" ] } ]
            )

        let! elaborated =
            IndexBuilder.elaborateKeywords (KeywordElaborationOptions.withGenerator generator) passages

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
        passageWithKeywords
            "pdf-1"
            0
            "The system indexes local PDF passages."
            [ "insurance claims"; "policy support" ]
            (vector [| 1; 2 |] [| 1.0f; 0.0f; 0.0f; 1.0f |])

    let index = index [ passage ]

    let path = IO.Path.Combine(IO.Path.GetTempPath(), $"{Guid.NewGuid():N}.fsci")

    try
        IndexPersistence.save path index
        let loaded = IndexPersistence.load path

        Assert.Equal(1, loaded.passages.Length)
        Assert.Equal("PDF: pdf-1", loaded.passages.Head.reference.sourceDisplayName)
        Assert.Equal<string list>([ "insurance claims"; "policy support" ], loaded.passages.Head.reference.keywords)
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
        Assert.Equal(TfidfOptions.defaults.textWeight, loaded.tfidfOptions.textWeight)
        Assert.Equal(TfidfOptions.defaults.keywordWeight, loaded.tfidfOptions.keywordWeight)
        Assert.True(loaded.tfidf.vocabulary.ContainsKey "prior")
    finally
        if IO.File.Exists path then
            IO.File.Delete path
