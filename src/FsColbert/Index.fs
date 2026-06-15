namespace FsColbert

open System
open System.Threading
open FSharp.Control

module PassageContext =
    let private distinctNonEmpty values =
        values
        |> Seq.choose (fun value ->
            let trimmed = Text.normalizeWhitespace value

            if String.IsNullOrWhiteSpace trimmed then
                None
            else
                Some trimmed)
        |> Seq.distinctBy _.ToLowerInvariant()
        |> Seq.toList

    let private roleKeywords =
        function
        | PassageContentRole.Unknown -> []
        | PassageContentRole.FrontMatter -> [ "front matter"; "title"; "authors"; "paper metadata"; "paper authors" ]
        | PassageContentRole.Abstract -> [ "abstract"; "paper abstract"; "summary" ]
        | PassageContentRole.MainBody -> [ "main body"; "paper body"; "content" ]
        | PassageContentRole.References -> [ "references"; "bibliography"; "citations" ]
        | PassageContentRole.Appendix -> [ "appendix"; "supplementary material" ]
        | PassageContentRole.SubmissionChecklist ->
            [ "submission checklist"
              "paper checklist"
              "compliance"
              "guidelines"
              "disclosure"
              "neurips"
              "question"
              "answer" ]

    let deterministicKeywords (passage: PassageRef) =
        distinctNonEmpty
            [ yield! passage.sectionPath
              yield PassageContentRole.displayName passage.contentRole
              yield PassageContentRole.storageValue passage.contentRole
              yield! roleKeywords passage.contentRole
              if not (List.isEmpty passage.pageNumbers) then
                  yield "page"
                  yield "pages" ]

    let contextualText (passage: PassageRef) =
        let sectionPath = String.concat " > " passage.sectionPath
        let pageNumbers = passage.pageNumbers |> List.map string |> String.concat ", "

        let metadata =
            [ yield $"Document: {passage.sourceDisplayName}"
              match passage.contentRole with
              | PassageContentRole.Unknown -> ()
              | role -> yield $"Role: {PassageContentRole.displayName role}"
              match passage.sectionPath with
              | [] -> ()
              | _ -> yield $"Section: {sectionPath}"
              match passage.pageNumbers with
              | [] -> ()
              | _ -> yield $"Pages: {pageNumbers}" ]

        match metadata with
        | [] -> passage.text
        | _ -> String.concat "\n" metadata + "\n\n" + passage.text

module IndexBuilder =
    let private allPassages options (sources: SourceDocument list) =
        sources |> List.filter _.enabled |> List.collect (Text.splitPassages options)

    let private cleanKeywords maxKeywords values =
        values
        |> List.choose (fun value ->
            let trimmed = Text.normalizeWhitespace value

            if String.IsNullOrWhiteSpace trimmed then
                None
            else
                Some trimmed)
        |> List.distinctBy _.ToLowerInvariant()
        |> List.truncate maxKeywords

    let private keywordKey sourceId passageIndex = sourceId, passageIndex

    let private mergeKeywords options existing generated =
        let values =
            if options.replaceExistingKeywords then
                generated
            else
                existing @ generated

        cleanKeywords options.maxKeywordsPerPassage values

    let private applyKeywordResults
        (options: KeywordElaborationOptions)
        (passages: PassageRef list)
        (results: PassageKeywordResult list)
        =
        let generated =
            results
            |> List.groupBy (fun result -> keywordKey result.sourceId result.passageIndex)
            |> List.map (fun (key, items) -> key, items |> List.collect _.keywords)
            |> Map.ofList

        passages
        |> List.map (fun passage ->
            let key = keywordKey passage.sourceId passage.index

            match Map.tryFind key generated with
            | Some keywords ->
                { passage with
                    keywords = mergeKeywords options passage.keywords keywords }
            | None ->
                { passage with
                    keywords = cleanKeywords options.maxKeywordsPerPassage passage.keywords })

    let elaborateKeywords (keywordOptions: KeywordElaborationOptions) (passages: PassageRef list) =
        async {
            let keywordOptions = KeywordElaborationOptions.sanitize keywordOptions

            match keywordOptions.generator, passages with
            | None, _ -> return passages
            | _, [] -> return []
            | Some generator, _ ->
                let! generated =
                    passages
                    |> List.chunkBySize keywordOptions.batchSize
                    |> AsyncSeq.ofSeq
                    |> AsyncSeq.mapAsyncParallelThrottled
                        keywordOptions.maxDegreeOfParallelism
                        generator.GenerateKeywordsAsync
                    |> AsyncSeq.toListAsync

                return generated |> List.collect id |> applyKeywordResults keywordOptions passages
        }

    let private applyDeterministicContextKeywords (passages: PassageRef list) =
        passages
        |> List.map (fun passage ->
            { passage with
                keywords =
                    seq {
                        yield! passage.keywords
                        yield! PassageContext.deterministicKeywords passage
                    }
                    |> Seq.toList
                    |> cleanKeywords 64 })

    let private indexBatch
        (encoder: OnnxColbertEncoder)
        (cancellationToken: CancellationToken)
        (batch: (int * PassageRef) array)
        =
        async {
            cancellationToken.ThrowIfCancellationRequested()

            let texts =
                batch |> Array.map (fun (_, passage) -> PassageContext.contextualText passage)

            let! encoded = encoder.EncodeDocumentsAsync texts
            cancellationToken.ThrowIfCancellationRequested()

            return
                Array.zip batch encoded
                |> Array.map (fun ((ordinal, passage), encoded) ->
                    let contextualText = PassageContext.contextualText passage

                    ordinal,
                    { reference = passage
                      embedding = encoded.embedding
                      terms = Text.terms contextualText })
        }

    let private reportProgress progress completed total currentSource =
        progress
        |> Option.iter (fun report ->
            report
                { completedPassages = completed
                  totalPassages = total
                  currentSource = currentSource })

    let private indexPassages
        (encoder: OnnxColbertEncoder)
        (indexingOptions: IndexingOptions)
        (passages: PassageRef list)
        (progress: (IndexProgress -> unit) option)
        (cancellationToken: CancellationToken)
        =
        async {
            cancellationToken.ThrowIfCancellationRequested()
            let passages = passages |> List.toArray
            let total = passages.Length
            let results = Array.zeroCreate<IndexedPassage> total
            let indexingOptions = IndexingOptions.sanitize indexingOptions
            let parallelism = min (max 1 total) indexingOptions.maxDegreeOfParallelism
            let progressLock = obj ()
            let mutable completed = 0

            reportProgress progress completed total None

            let storeBatch (batch: (int * IndexedPassage) array) =
                for ordinal, indexed in batch do
                    results[ordinal] <- indexed

                lock progressLock (fun () ->
                    completed <- completed + batch.Length

                    let currentSource =
                        batch
                        |> Array.tryLast
                        |> Option.map (fun (_, indexed) -> indexed.reference.sourceDisplayName)

                    reportProgress progress completed total currentSource)

            if total > 0 then
                do!
                    passages
                    |> Array.mapi (fun ordinal passage -> ordinal, passage)
                    |> AsyncSeq.ofSeq
                    |> AsyncSeq.bufferByCountAndTime indexingOptions.batchSize 1
                    |> AsyncSeq.mapAsyncParallelThrottled parallelism (indexBatch encoder cancellationToken)
                    |> AsyncSeq.iterAsync (fun batch ->
                        async {
                            cancellationToken.ThrowIfCancellationRequested()
                            storeBatch batch
                        })

            cancellationToken.ThrowIfCancellationRequested()
            reportProgress progress completed total None

            return results |> Array.toList
        }

    let createWithOptionsAndTfidfOptionsAndKeywordElaborationOptions
        (encoder: OnnxColbertEncoder)
        (options: ChunkOptions)
        (indexingOptions: IndexingOptions)
        (tfidfOptions: TfidfOptions)
        (keywordOptions: KeywordElaborationOptions)
        (sources: SourceDocument list)
        (progress: (IndexProgress -> unit) option)
        =
        async {
            let! passages = allPassages options sources |> elaborateKeywords keywordOptions
            let passages = applyDeterministicContextKeywords passages
            let! indexed = indexPassages encoder indexingOptions passages progress CancellationToken.None

            return
                { config = encoder.Config
                  chunkOptions = options
                  tfidfOptions = tfidfOptions
                  passages = indexed
                  tfidf = Tfidf.buildWithOptions tfidfOptions indexed
                  createdAt = DateTimeOffset.UtcNow }
        }

    let createWithOptionsAndTfidfOptions
        (encoder: OnnxColbertEncoder)
        (options: ChunkOptions)
        (indexingOptions: IndexingOptions)
        (tfidfOptions: TfidfOptions)
        (sources: SourceDocument list)
        (progress: (IndexProgress -> unit) option)
        =
        createWithOptionsAndTfidfOptionsAndKeywordElaborationOptions
            encoder
            options
            indexingOptions
            tfidfOptions
            KeywordElaborationOptions.disabled
            sources
            progress

    let createWithOptions
        (encoder: OnnxColbertEncoder)
        (options: ChunkOptions)
        (indexingOptions: IndexingOptions)
        (sources: SourceDocument list)
        (progress: (IndexProgress -> unit) option)
        =
        createWithOptionsAndTfidfOptions encoder options indexingOptions TfidfOptions.defaults sources progress

    let create
        (encoder: OnnxColbertEncoder)
        (options: ChunkOptions)
        (sources: SourceDocument list)
        (progress: (IndexProgress -> unit) option)
        =
        createWithOptions encoder options IndexingOptions.defaults sources progress

    let createWithDefaults encoder sources progress =
        create encoder ChunkOptions.fsKameDefaults sources progress

    let createFromPassagesWithOptionsAndTfidfOptionsAndKeywordElaborationOptions
        (encoder: OnnxColbertEncoder)
        (chunkOptions: ChunkOptions)
        (indexingOptions: IndexingOptions)
        (tfidfOptions: TfidfOptions)
        (keywordOptions: KeywordElaborationOptions)
        (passages: PassageRef list)
        (progress: (IndexProgress -> unit) option)
        =
        async {
            let! passages = elaborateKeywords keywordOptions passages
            let passages = applyDeterministicContextKeywords passages
            let! indexed = indexPassages encoder indexingOptions passages progress CancellationToken.None

            return
                { config = encoder.Config
                  chunkOptions = chunkOptions
                  tfidfOptions = tfidfOptions
                  passages = indexed
                  tfidf = Tfidf.buildWithOptions tfidfOptions indexed
                  createdAt = DateTimeOffset.UtcNow }
        }

    let createFromPassagesWithOptionsAndTfidfOptionsAndKeywordElaborationOptionsWithCancellation
        (encoder: OnnxColbertEncoder)
        (chunkOptions: ChunkOptions)
        (indexingOptions: IndexingOptions)
        (tfidfOptions: TfidfOptions)
        (keywordOptions: KeywordElaborationOptions)
        (passages: PassageRef list)
        (progress: (IndexProgress -> unit) option)
        (cancellationToken: CancellationToken)
        =
        async {
            cancellationToken.ThrowIfCancellationRequested()
            let! passages = elaborateKeywords keywordOptions passages
            let passages = applyDeterministicContextKeywords passages
            cancellationToken.ThrowIfCancellationRequested()
            let! indexed = indexPassages encoder indexingOptions passages progress cancellationToken
            cancellationToken.ThrowIfCancellationRequested()

            return
                { config = encoder.Config
                  chunkOptions = chunkOptions
                  tfidfOptions = tfidfOptions
                  passages = indexed
                  tfidf = Tfidf.buildWithOptions tfidfOptions indexed
                  createdAt = DateTimeOffset.UtcNow }
        }

    let createFromPassagesWithOptionsAndTfidfOptions
        (encoder: OnnxColbertEncoder)
        (chunkOptions: ChunkOptions)
        (indexingOptions: IndexingOptions)
        (tfidfOptions: TfidfOptions)
        (passages: PassageRef list)
        (progress: (IndexProgress -> unit) option)
        =
        createFromPassagesWithOptionsAndTfidfOptionsAndKeywordElaborationOptions
            encoder
            chunkOptions
            indexingOptions
            tfidfOptions
            KeywordElaborationOptions.disabled
            passages
            progress

    let createFromPassagesWithOptionsAndTfidfOptionsWithCancellation
        (encoder: OnnxColbertEncoder)
        (chunkOptions: ChunkOptions)
        (indexingOptions: IndexingOptions)
        (tfidfOptions: TfidfOptions)
        (passages: PassageRef list)
        (progress: (IndexProgress -> unit) option)
        (cancellationToken: CancellationToken)
        =
        createFromPassagesWithOptionsAndTfidfOptionsAndKeywordElaborationOptionsWithCancellation
            encoder
            chunkOptions
            indexingOptions
            tfidfOptions
            KeywordElaborationOptions.disabled
            passages
            progress
            cancellationToken

    let createFromPassagesWithOptions
        (encoder: OnnxColbertEncoder)
        (chunkOptions: ChunkOptions)
        (indexingOptions: IndexingOptions)
        (passages: PassageRef list)
        (progress: (IndexProgress -> unit) option)
        =
        createFromPassagesWithOptionsAndTfidfOptions
            encoder
            chunkOptions
            indexingOptions
            TfidfOptions.defaults
            passages
            progress

    let createFromPassagesWithOptionsWithCancellation
        (encoder: OnnxColbertEncoder)
        (chunkOptions: ChunkOptions)
        (indexingOptions: IndexingOptions)
        (passages: PassageRef list)
        (progress: (IndexProgress -> unit) option)
        (cancellationToken: CancellationToken)
        =
        createFromPassagesWithOptionsAndTfidfOptionsWithCancellation
            encoder
            chunkOptions
            indexingOptions
            TfidfOptions.defaults
            passages
            progress
            cancellationToken

    let createFromPassages
        (encoder: OnnxColbertEncoder)
        (chunkOptions: ChunkOptions)
        (passages: PassageRef list)
        (progress: (IndexProgress -> unit) option)
        =
        createFromPassagesWithOptions encoder chunkOptions IndexingOptions.defaults passages progress

    let createFromPassagesWithCancellation
        (encoder: OnnxColbertEncoder)
        (chunkOptions: ChunkOptions)
        (passages: PassageRef list)
        (progress: (IndexProgress -> unit) option)
        (cancellationToken: CancellationToken)
        =
        createFromPassagesWithOptionsWithCancellation
            encoder
            chunkOptions
            IndexingOptions.defaults
            passages
            progress
            cancellationToken

module Search =
    let private candidatePassages options (index: ColbertIndex) queryText searchTerms =
        let passages = index.passages |> List.toArray

        if options.useLexicalFilter then
            Tfidf.scoreQueryWithSearchTerms index.tfidf queryText searchTerms
            |> Tfidf.topCandidates options.candidateLimit
            |> Array.choose (fun (ordinal, lexicalScore) ->
                if ordinal >= 0 && ordinal < passages.Length then
                    Some(passages[ordinal], lexicalScore)
                else
                    None)
            |> Array.toList
        else
            passages |> Array.map (fun p -> p, 0.0f) |> Array.toList

    let private reciprocalRankFusion k denseWeight lexicalWeight (hits: SearchHit list) =
        let denseRank =
            hits
            |> List.sortByDescending (fun h -> h.denseScore)
            |> List.mapi (fun i h -> h.reference, i + 1)
            |> Map.ofList

        let lexicalRank =
            hits
            |> List.sortByDescending (fun h -> h.lexicalScore)
            |> List.mapi (fun i h -> h.reference, i + 1)
            |> Map.ofList

        hits
        |> List.map (fun h ->
            let rDense = Map.find h.reference denseRank
            let rLexical = Map.find h.reference lexicalRank

            let score =
                denseWeight * (1.0f / (float32 k + float32 rDense))
                + lexicalWeight * (1.0f / (float32 k + float32 rLexical))

            { h with score = score })

    let private rankCandidates options queryEmbedding candidates =
        let rawHits =
            candidates
            |> List.map (fun (passage, lexicalScore) ->
                let denseScore = Scoring.maxSim queryEmbedding passage.embedding

                let score =
                    Scoring.combinedScore options.denseWeight options.lexicalWeight denseScore lexicalScore

                { reference = passage.reference
                  denseScore = denseScore
                  lexicalScore = lexicalScore
                  score = score })

        let hits =
            if options.useRRF then
                reciprocalRankFusion options.fusionK options.denseWeight options.lexicalWeight rawHits
            else
                rawHits

        hits
        |> List.sortByDescending (fun hit -> hit.score, hit.denseScore)
        |> List.truncate (max 0 options.maxResults)

    let queryEncodedWithSearchTerms
        (options: SearchOptions)
        (index: ColbertIndex)
        (queryText: string)
        (searchTerms: string seq)
        (queryEmbedding: MultiVector)
        =
        candidatePassages options index queryText searchTerms
        |> rankCandidates options queryEmbedding

    let queryEncoded (options: SearchOptions) (index: ColbertIndex) (queryText: string) (queryEmbedding: MultiVector) =
        queryEncodedWithSearchTerms options index queryText [] queryEmbedding

    let queryWithSearchTerms
        (encoder: OnnxColbertEncoder)
        (options: SearchOptions)
        (index: ColbertIndex)
        (queryText: string)
        (searchTerms: string seq)
        =
        async {
            let candidates = candidatePassages options index queryText searchTerms

            if List.isEmpty candidates then
                return []
            else
                let! encoded = encoder.EncodeQueryAsync queryText

                return candidates |> rankCandidates options encoded.embedding
        }

    let query (encoder: OnnxColbertEncoder) (options: SearchOptions) (index: ColbertIndex) (queryText: string) =
        queryWithSearchTerms encoder options index queryText []

    let queryWithDefaultsAndSearchTerms encoder index queryText searchTerms =
        queryWithSearchTerms encoder SearchOptions.defaults index queryText searchTerms

    let queryWithDefaults encoder index queryText =
        query encoder SearchOptions.defaults index queryText
