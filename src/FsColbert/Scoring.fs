namespace FsColbert

open System.Numerics

module Scoring =
    let private dot dim (left: float32 array) leftOffset (right: float32 array) rightOffset =
        let vectorWidth = Vector<float32>.Count
        let vectorLimit = dim - (dim % vectorWidth)
        let mutable vectorAcc = Vector<float32>.Zero
        let mutable index = 0

        while index < vectorLimit do
            let leftVector = Vector<float32>(left, leftOffset + index)
            let rightVector = Vector<float32>(right, rightOffset + index)
            vectorAcc <- vectorAcc + (leftVector * rightVector)
            index <- index + vectorWidth

        let mutable acc = 0.0f

        for lane = 0 to vectorWidth - 1 do
            acc <- acc + vectorAcc[lane]

        while index < dim do
            acc <- acc + left[leftOffset + index] * right[rightOffset + index]
            index <- index + 1

        acc

    let maxSim (query: MultiVector) (document: MultiVector) =
        if query.embeddingDim <> document.embeddingDim then
            invalidArg "document" "Query and document embeddings must have the same dimension."

        if query.tokenCount = 0 || document.tokenCount = 0 then
            0.0f
        else
            let dim = query.embeddingDim
            let mutable score = 0.0f

            for queryToken = 0 to query.tokenCount - 1 do
                let queryOffset = queryToken * dim
                let mutable best = System.Single.NegativeInfinity

                for documentToken = 0 to document.tokenCount - 1 do
                    let documentOffset = documentToken * dim
                    let value = dot dim query.vectors queryOffset document.vectors documentOffset

                    if value > best then
                        best <- value

                if best > System.Single.NegativeInfinity then
                    score <- score + best

            score

    let lexicalOverlap (queryTerms: Set<string>) (documentTerms: Set<string>) =
        if Set.isEmpty queryTerms || Set.isEmpty documentTerms then
            0.0f
        else
            queryTerms
            |> Seq.sumBy (fun term -> if documentTerms.Contains term then 1 else 0)
            |> float32

    let combinedScore (denseWeight: float32) (lexicalWeight: float32) (dense: float32) (lexical: float32) =
        denseWeight * dense + lexicalWeight * lexical
