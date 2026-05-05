namespace FsColbert

open System
open System.Collections.Frozen

module Log =
    type FsColbertLog = class end
    let log = FSharp.DI.DI.loggerLazy<FsColbertLog>()

type EncoderMode =
    | Query
    | Document

type ModelFiles =
    { modelPath: string
      tokenizerPath: string
      configPath: string option }

type RuntimeOptions =
    { intraOpThreads: int option
      interOpThreads: int option }

module RuntimeOptions =
    let defaults =
        { intraOpThreads = Some 1
          interOpThreads = Some 1 }

type EncoderConfig =
    { queryLength: int
      documentLength: int
      embeddingDim: int
      padTokenId: int
      maskTokenId: int
      clsTokenId: int
      sepTokenId: int
      queryPrefixId: int
      documentPrefixId: int
      doLowerCase: bool
      normalizeOutput: bool
      skiplistWords: Set<string> }

module EncoderConfig =
    let mxbaiEdgeColbert =
        { queryLength = 48
          documentLength = 512
          embeddingDim = 64
          padTokenId = 50284
          maskTokenId = 50284
          clsTokenId = 50281
          sepTokenId = 50282
          queryPrefixId = 50368
          documentPrefixId = 50369
          doLowerCase = true
          normalizeOutput = true
          skiplistWords =
            [ "!"
              "\""
              "#"
              "$"
              "%"
              "&"
              "'"
              "("
              ")"
              "*"
              "+"
              ","
              "-"
              "."
              "/"
              ":"
              ";"
              "<"
              "="
              ">"
              "?"
              "@"
              "["
              "\\"
              "]"
              "^"
              "_"
              "`"
              "{"
              "|"
              "}"
              "~" ]
            |> Set.ofList }

type TokenizedInput =
    { inputIds: int64 array
      attentionMask: int64 array
      effectiveLength: int }

type MultiVector =
    { tokenIds: int array
      vectors: float32 array
      tokenCount: int
      embeddingDim: int }

type EncodedText =
    { mode: EncoderMode
      tokenized: TokenizedInput
      embedding: MultiVector }

type ChunkOptions =
    { maxChars: int
      overlapChars: int
      minChars: int }

module ChunkOptions =
    let fsKameDefaults =
        { maxChars = 1800
          overlapChars = 250
          minChars = 20 }

type SourceDocument =
    { id: string
      displayName: string
      location: string
      text: string
      enabled: bool }

type PreChunkedDocument =
    { id: string
      displayName: string
      location: string
      chunks: string list
      enabled: bool }

type PassageRef =
    { sourceId: string
      sourceDisplayName: string
      sourceLocation: string
      index: int
      text: string }

type IndexedPassage =
    { reference: PassageRef
      embedding: MultiVector
      terms: Set<string> }

type Posting =
    { passageOrdinal: int
      termFrequency: float32 }

type TermInfo =
    { documentFrequency: int
      inverseDocumentFrequency: float32
      postings: Posting array }

type TfidfIndex =
    { passageCount: int
      averageDocumentLength: float32
      vocabulary: FrozenDictionary<string, TermInfo> }

type ColbertIndex =
    { config: EncoderConfig
      chunkOptions: ChunkOptions
      passages: IndexedPassage list
      tfidf: TfidfIndex
      createdAt: DateTimeOffset }

type SearchOptions =
    { maxResults: int
      candidateLimit: int
      denseWeight: float32
      lexicalWeight: float32 }

module SearchOptions =
    let defaults =
        { maxResults = 6
          candidateLimit = 128
          denseWeight = 1.0f
          lexicalWeight = 0.15f }

type SearchHit =
    { reference: PassageRef
      denseScore: float32
      lexicalScore: float32
      score: float32 }

type IndexProgress =
    { completedPassages: int
      totalPassages: int
      currentSource: string option }
