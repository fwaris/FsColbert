namespace FsColbert

open System
open System.Collections.Generic
open System.Numerics
open System.Runtime.CompilerServices
open System.Threading

module internal SemanticCandidateIndex =
    type private CentroidIndex =
        { embeddingDim: int
          passageCount: int
          vectors: float32 array }

    let private normalizeInPlace (values: float32 array) offset length =
        let mutable squaredLength = 0.0f

        for index = offset to offset + length - 1 do
            let value = values[index]
            squaredLength <- squaredLength + value * value

        if squaredLength > 0.0f then
            let scale = 1.0f / sqrt squaredLength

            for index = offset to offset + length - 1 do
                values[index] <- values[index] * scale

    let private meanPoolInto (target: float32 array) targetOffset (embedding: MultiVector) =
        let dimension = embedding.embeddingDim

        if embedding.tokenCount > 0 && dimension > 0 then
            for token = 0 to embedding.tokenCount - 1 do
                let sourceOffset = token * dimension

                for index = 0 to dimension - 1 do
                    target[targetOffset + index] <-
                        target[targetOffset + index] + embedding.vectors[sourceOffset + index]

            normalizeInPlace target targetOffset dimension

    let private centroid (embedding: MultiVector) =
        let values = Array.zeroCreate<float32> (max 0 embedding.embeddingDim)
        meanPoolInto values 0 embedding
        values

    let private build (index: ColbertIndex) =
        let passages = index.passages |> List.toArray

        let dimension =
            passages
            |> Array.tryHead
            |> Option.map (fun passage -> passage.embedding.embeddingDim)
            |> Option.defaultValue index.config.embeddingDim
            |> max 0

        let vectors = Array.zeroCreate<float32> (passages.Length * dimension)

        passages
        |> Array.iteri (fun ordinal passage ->
            if passage.embedding.embeddingDim <> dimension then
                invalidOp
                    $"Passage {ordinal} embedding dimension {passage.embedding.embeddingDim} does not match index dimension {dimension}."

            meanPoolInto vectors (ordinal * dimension) passage.embedding)

        { embeddingDim = dimension
          passageCount = passages.Length
          vectors = vectors }

    let private cache = ConditionalWeakTable<ColbertIndex, Lazy<CentroidIndex>>()

    let private cached index =
        cache
            .GetValue(
                index,
                fun value -> Lazy<CentroidIndex>((fun () -> build value), LazyThreadSafetyMode.ExecutionAndPublication)
            )
            .Value

    let private dot dimension (left: float32 array) (right: float32 array) rightOffset =
        let vectorWidth = Vector<float32>.Count
        let vectorLimit = dimension - (dimension % vectorWidth)
        let mutable vectorAcc = Vector<float32>.Zero
        let mutable index = 0

        while index < vectorLimit do
            let leftVector = Vector<float32>(left, index)
            let rightVector = Vector<float32>(right, rightOffset + index)
            vectorAcc <- vectorAcc + leftVector * rightVector
            index <- index + vectorWidth

        let mutable score = 0.0f

        for lane = 0 to vectorWidth - 1 do
            score <- score + vectorAcc[lane]

        while index < dimension do
            score <- score + left[index] * right[rightOffset + index]
            index <- index + 1

        score

    let prepare index = cached index |> ignore

    let topCandidates limit index queryEmbedding =
        if limit <= 0 || queryEmbedding.tokenCount <= 0 then
            [||]
        else
            let semantic = cached index

            if queryEmbedding.embeddingDim <> semantic.embeddingDim then
                invalidArg
                    "queryEmbedding"
                    $"Query embedding dimension {queryEmbedding.embeddingDim} does not match semantic index dimension {semantic.embeddingDim}."

            let query = centroid queryEmbedding
            let resultLimit = min limit semantic.passageCount
            let queue = PriorityQueue<int * float32, struct (float32 * int)>()

            for ordinal = 0 to semantic.passageCount - 1 do
                let score =
                    dot semantic.embeddingDim query semantic.vectors (ordinal * semantic.embeddingDim)

                queue.Enqueue((ordinal, score), struct (score, -ordinal))

                if queue.Count > resultLimit then
                    queue.Dequeue() |> ignore

            queue.UnorderedItems
            |> Seq.map (fun struct (element, _) -> element)
            |> Seq.sortBy (fun (ordinal, score) -> -score, ordinal)
            |> Seq.toArray
