namespace FsColbert

open System
open System.Collections.Frozen

module Log =
    type FsColbertLog = class end
    let log = FSharp.DI.DI.loggerLazy<FsColbertLog> ()

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
      text: string
      sectionPath: string list
      keywords: string list }

type PassageKeywordResult =
    { sourceId: string
      passageIndex: int
      keywords: string list }

type IPassageKeywordGenerator =
    abstract GenerateKeywordsAsync: PassageRef list -> Async<PassageKeywordResult list>

type KeywordElaborationOptions =
    { generator: IPassageKeywordGenerator option
      batchSize: int
      maxDegreeOfParallelism: int
      replaceExistingKeywords: bool
      maxKeywordsPerPassage: int }

module KeywordElaborationOptions =
    let disabled =
        { generator = None
          batchSize = 4
          maxDegreeOfParallelism = 2
          replaceExistingKeywords = false
          maxKeywordsPerPassage = 16 }

    let withGenerator generator =
        { disabled with
            generator = Some generator }

    let sanitize options =
        { options with
            batchSize = max 1 options.batchSize
            maxDegreeOfParallelism = max 1 options.maxDegreeOfParallelism
            maxKeywordsPerPassage = max 0 options.maxKeywordsPerPassage }

type IndexedPassage =
    { reference: PassageRef
      embedding: MultiVector
      terms: Set<string> }

type IndexingOptions =
    { maxDegreeOfParallelism: int
      batchSize: int }

module IndexingOptions =
    let private environmentInt name =
        match Environment.GetEnvironmentVariable name with
        | value when String.IsNullOrWhiteSpace value -> None
        | value ->
            match Int32.TryParse value with
            | true, parsed when parsed > 0 -> Some parsed
            | _ -> None

    let private environmentParallelism () =
        environmentInt "FSCOLBERT_INDEX_PARALLELISM"
        |> Option.orElse (environmentInt "FSCOLBERT_MODEL_REPLICAS")

    let defaults =
        { maxDegreeOfParallelism =
            environmentParallelism ()
            |> Option.defaultValue (min 3 (max 1 Environment.ProcessorCount))
          batchSize = environmentInt "FSCOLBERT_INDEX_BATCH_SIZE" |> Option.defaultValue 16 }

    let sanitize options =
        { maxDegreeOfParallelism = max 1 options.maxDegreeOfParallelism
          batchSize = max 1 options.batchSize }

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

type TfidfOptions =
    { textWeight: float32
      keywordWeight: float32 }

module TfidfOptions =
    let defaults =
        { textWeight = 1.0f
          keywordWeight = 3.0f }

type ColbertIndex =
    { config: EncoderConfig
      chunkOptions: ChunkOptions
      tfidfOptions: TfidfOptions
      passages: IndexedPassage list
      tfidf: TfidfIndex
      createdAt: DateTimeOffset }

type SearchOptions =
    { maxResults: int
      candidateLimit: int
      denseWeight: float32
      lexicalWeight: float32
      useLexicalFilter: bool
      useRRF: bool
      fusionK: int }

module SearchOptions =
    let defaults =
        { maxResults = 6
          candidateLimit = 128
          denseWeight = 1.0f
          lexicalWeight = 0.15f
          useLexicalFilter = true
          useRRF = true
          fusionK = 60 }

type SearchHit =
    { reference: PassageRef
      denseScore: float32
      lexicalScore: float32
      score: float32 }

type IndexProgress =
    { completedPassages: int
      totalPassages: int
      currentSource: string option }
