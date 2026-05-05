module FsColbert.Tests

open System
open FsColbert
open Xunit

let private vector tokenIds vectors =
    { tokenIds = tokenIds
      vectors = vectors
      tokenCount = tokenIds.Length
      embeddingDim = 2 }

let private passage sourceId index text embedding =
    { reference =
        { sourceId = sourceId
          sourceDisplayName = $"PDF: {sourceId}"
          sourceLocation = $"/tmp/{sourceId}.pdf"
          index = index
          text = text }
      embedding = embedding
      terms = Text.terms text }

let private index passages =
    { config = EncoderConfig.mxbaiEdgeColbert
      chunkOptions = ChunkOptions.fsKameDefaults
      passages = passages
      tfidf = Tfidf.build passages
      createdAt = DateTimeOffset.UtcNow }

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
        passage "pdf-1" 0 "The system indexes local PDF passages." (vector [| 1; 2 |] [| 1.0f; 0.0f; 0.0f; 1.0f |])

    let index = index [ passage ]

    let path = IO.Path.Combine(IO.Path.GetTempPath(), $"{Guid.NewGuid():N}.fsci")

    try
        IndexPersistence.save path index
        let loaded = IndexPersistence.load path

        Assert.Equal(1, loaded.passages.Length)
        Assert.Equal("PDF: pdf-1", loaded.passages.Head.reference.sourceDisplayName)
        Assert.Equal<float32 array>(passage.embedding.vectors, loaded.passages.Head.embedding.vectors)
        Assert.Equal(1, loaded.tfidf.passageCount)
        Assert.True(loaded.tfidf.vocabulary.ContainsKey "system")
        Assert.Equal(1, loaded.tfidf.vocabulary["system"].documentFrequency)
        Assert.Equal(0, loaded.tfidf.vocabulary["system"].postings[0].passageOrdinal)
    finally
        if IO.File.Exists path then
            IO.File.Delete path
