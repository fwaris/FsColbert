namespace FsColbert

open System

module IndexBuilder =
    let private allPassages options (sources: SourceDocument list) =
        sources |> List.filter _.enabled |> List.collect (Text.splitPassages options)

    let create
        (encoder: OnnxColbertEncoder)
        (options: ChunkOptions)
        (sources: SourceDocument list)
        (progress: (IndexProgress -> unit) option)
        =
        async {
            let passages = allPassages options sources
            let total = passages.Length
            let mutable completed = 0
            let indexed = ResizeArray<IndexedPassage>()

            for passage in passages do
                progress
                |> Option.iter (fun report ->
                    report
                        { completedPassages = completed
                          totalPassages = total
                          currentSource = Some passage.sourceDisplayName })

                let! encoded = encoder.EncodeDocumentAsync passage.text

                indexed.Add(
                    { reference = passage
                      embedding = encoded.embedding
                      terms = Text.terms passage.text }
                )

                completed <- completed + 1

            progress
            |> Option.iter (fun report ->
                report
                    { completedPassages = completed
                      totalPassages = total
                      currentSource = None })

            let indexed = indexed |> Seq.toList

            return
                { config = encoder.Config
                  chunkOptions = options
                  passages = indexed
                  tfidf = Tfidf.build indexed
                  createdAt = DateTimeOffset.UtcNow }
        }

    let createWithDefaults encoder sources progress =
        create encoder ChunkOptions.fsKameDefaults sources progress

    let createFromPassages
        (encoder: OnnxColbertEncoder)
        (chunkOptions: ChunkOptions)
        (passages: PassageRef list)
        (progress: (IndexProgress -> unit) option)
        =
        async {
            let total = passages.Length
            let mutable completed = 0
            let indexed = ResizeArray<IndexedPassage>()

            for passage in passages do
                progress
                |> Option.iter (fun report ->
                    report
                        { completedPassages = completed
                          totalPassages = total
                          currentSource = Some passage.sourceDisplayName })

                let! encoded = encoder.EncodeDocumentAsync passage.text

                indexed.Add(
                    { reference = passage
                      embedding = encoded.embedding
                      terms = Text.terms passage.text }
                )

                completed <- completed + 1

            progress
            |> Option.iter (fun report ->
                report
                    { completedPassages = completed
                      totalPassages = total
                      currentSource = None })

            let indexed = indexed |> Seq.toList

            return
                { config = encoder.Config
                  chunkOptions = chunkOptions
                  passages = indexed
                  tfidf = Tfidf.build indexed
                  createdAt = DateTimeOffset.UtcNow }
        }

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

        hits |> List.map (fun h ->
            let rDense = Map.find h.reference denseRank
            let rLexical = Map.find h.reference lexicalRank
            let score = 
                denseWeight * (1.0f / (float32 k + float32 rDense)) + 
                lexicalWeight * (1.0f / (float32 k + float32 rLexical))
            { h with score = score }
        )

    let private rankCandidates options queryEmbedding candidates =
        let rawHits = 
            candidates
            |> List.map (fun (passage, lexicalScore) ->
                let denseScore = Scoring.maxSim queryEmbedding passage.embedding
                let score = Scoring.combinedScore options.denseWeight options.lexicalWeight denseScore lexicalScore

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
