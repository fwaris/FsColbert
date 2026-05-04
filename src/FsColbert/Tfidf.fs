namespace FsColbert

open System
open System.Collections.Frozen

module Tfidf =
    let private inverseDocumentFrequency passageCount documentFrequency =
        log ((1.0f + float32 passageCount) / (1.0f + float32 documentFrequency)) + 1.0f

    let private normalizedFrequencies text =
        let frequencies, totalTerms = Text.termFrequencies text

        let normalized =
            if totalTerms <= 0 then
                Map.empty
            else
                frequencies |> Map.map (fun _ count -> float32 count / float32 totalTerms)

        normalized, totalTerms

    let build (passages: IndexedPassage list) =
        let passageCount = passages.Length

        let passageFrequencies =
            passages
            |> List.mapi (fun ordinal passage ->
                let frequencies, totalTerms = normalizedFrequencies passage.reference.text
                ordinal, frequencies, totalTerms)

        let averageDocumentLength =
            if passageCount = 0 then
                0.0f
            else
                passageFrequencies
                |> List.averageBy (fun (_, _, totalTerms) -> float totalTerms)
                |> float32

        let postingsByTerm =
            passageFrequencies
            |> List.collect (fun (ordinal, frequencies, _) ->
                frequencies
                |> Map.toList
                |> List.map (fun (term, frequency) ->
                    term,
                    { passageOrdinal = ordinal
                      termFrequency = frequency }))
            |> List.groupBy fst

        let vocabulary =
            postingsByTerm
            |> Seq.map (fun (term, postings) ->
                let postings =
                    postings |> List.map snd |> List.sortBy _.passageOrdinal |> List.toArray

                let termInfo =
                    { documentFrequency = postings.Length
                      inverseDocumentFrequency = inverseDocumentFrequency passageCount postings.Length
                      postings = postings }

                term, termInfo)
            |> Seq.map (fun (term, termInfo) -> Collections.Generic.KeyValuePair(term, termInfo))
            |> fun pairs -> FrozenDictionary.ToFrozenDictionary(pairs, StringComparer.Ordinal)

        { passageCount = passageCount
          averageDocumentLength = averageDocumentLength
          vocabulary = vocabulary }

    let scoreValues (index: TfidfIndex) (values: string seq) =
        let queryFrequencies, queryLength = Text.termFrequenciesFromValues values

        if queryLength <= 0 || index.passageCount <= 0 then
            [||]
        else
            let scores = Array.zeroCreate<float32> index.passageCount

            for KeyValue(term, queryCount) in queryFrequencies do
                match index.vocabulary.TryGetValue term with
                | true, termInfo ->
                    let queryWeight =
                        (float32 queryCount / float32 queryLength) * termInfo.inverseDocumentFrequency

                    for posting in termInfo.postings do
                        scores[posting.passageOrdinal] <-
                            scores[posting.passageOrdinal]
                            + posting.termFrequency * termInfo.inverseDocumentFrequency * queryWeight
                | false, _ -> ()

            scores
            |> Array.mapi (fun ordinal score -> ordinal, score)
            |> Array.filter (fun (_, score) -> score > 0.0f)

    let scoreQuery (index: TfidfIndex) (queryText: string) = scoreValues index [ queryText ]

    let scoreQueryWithSearchTerms (index: TfidfIndex) (queryText: string) (searchTerms: string seq) =
        scoreValues
            index
            (seq {
                yield queryText
                yield! searchTerms
            })

    let topCandidates candidateLimit (scores: (int * float32) array) =
        let sorted =
            scores |> Array.sortByDescending (fun (ordinal, score) -> score, -ordinal)

        if candidateLimit <= 0 then
            sorted
        else
            sorted |> Array.truncate candidateLimit
