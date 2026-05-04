namespace FsColbert

open System

module IndexBuilder =
    let private allPassages options sources =
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

module Search =
    let private candidatePassages options (index: ColbertIndex) queryText searchTerms =
        let passages = index.passages |> List.toArray

        Tfidf.scoreQueryWithSearchTerms index.tfidf queryText searchTerms
        |> Tfidf.topCandidates options.candidateLimit
        |> Array.choose (fun (ordinal, lexicalScore) ->
            if ordinal >= 0 && ordinal < passages.Length then
                Some(passages[ordinal], lexicalScore)
            else
                None)
        |> Array.toList

    let private rankCandidates options queryEmbedding candidates =
        candidates
        |> List.map (fun (passage, lexicalScore) ->
            let denseScore = Scoring.maxSim queryEmbedding passage.embedding

            let score =
                Scoring.combinedScore options.denseWeight options.lexicalWeight denseScore lexicalScore

            { reference = passage.reference
              denseScore = denseScore
              lexicalScore = lexicalScore
              score = score })
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
