namespace FsColbert

open System
open System.Collections.Generic
open Microsoft.ML.OnnxRuntime
open Microsoft.ML.OnnxRuntime.Tensors

module internal VectorMath =
    let normalizeInPlace (values: float32 array) offset length =
        let mutable sum = 0.0f

        for index = offset to offset + length - 1 do
            let value = values[index]
            sum <- sum + value * value

        if sum > 0.0f then
            let scale = 1.0f / sqrt sum

            for index = offset to offset + length - 1 do
                values[index] <- values[index] * scale

    let flattenMaskedVectorsAt embeddingDim (mask: bool array) (tensorValues: float32 array) tensorOffset =
        let kept = mask |> Array.filter id |> Array.length
        let vectors = Array.zeroCreate<float32> (kept * embeddingDim)
        let mutable target = 0

        for tokenIndex = 0 to mask.Length - 1 do
            if mask[tokenIndex] then
                Array.Copy(
                    tensorValues,
                    tensorOffset + tokenIndex * embeddingDim,
                    vectors,
                    target * embeddingDim,
                    embeddingDim
                )

                normalizeInPlace vectors (target * embeddingDim) embeddingDim
                target <- target + 1

        vectors

    let flattenMaskedVectors embeddingDim (mask: bool array) (tensorValues: float32 array) =
        flattenMaskedVectorsAt embeddingDim mask tensorValues 0

type private OnnxEncoderRunner(session: InferenceSession, tokenizer: ColbertTokenizer, config: EncoderConfig) =
    let gate = obj ()

    let inputNames =
        session.InputMetadata.Keys
        |> Seq.map (fun name -> name.ToLowerInvariant(), name)
        |> Map.ofSeq

    let outputName =
        session.OutputMetadata.Keys
        |> Seq.tryHead
        |> Option.defaultWith (fun () -> failwith "ONNX model has no outputs.")

    let makeInput name (data: int64 array) batchSize length =
        let dimensions = ReadOnlySpan<int>([| batchSize; length |])
        let tensor = DenseTensor<int64>(Memory<int64>(data), dimensions)
        NamedOnnxValue.CreateFromTensor(name, tensor)

    let tryAddInput (inputs: ResizeArray<NamedOnnxValue>) logicalName data batchSize length =
        inputNames
        |> Map.tryFind logicalName
        |> Option.iter (fun name -> inputs.Add(makeInput name data batchSize length))

    let tensorValuesToVectors mode (tokenized: TokenizedInput) sequenceLength embeddingDim tensorOffset tensorValues =
        if sequenceLength > tokenized.inputIds.Length then
            failwith
                $"ONNX output sequence length {sequenceLength} exceeds tokenizer length {tokenized.inputIds.Length}."

        let tokenMask =
            tokenizer.OutputTokenMask(mode, tokenized) |> Array.truncate sequenceLength

        let tokenIds =
            tokenized.inputIds
            |> Array.truncate sequenceLength
            |> Array.mapi (fun index tokenId -> index, int tokenId)
            |> Array.choose (fun (index, tokenId) -> if tokenMask[index] then Some tokenId else None)

        let vectors =
            if config.normalizeOutput then
                VectorMath.flattenMaskedVectorsAt embeddingDim tokenMask tensorValues tensorOffset
            else
                let kept = tokenMask |> Array.filter id |> Array.length
                let vectors = Array.zeroCreate<float32> (kept * embeddingDim)
                let mutable target = 0

                for tokenIndex = 0 to tokenMask.Length - 1 do
                    if tokenMask[tokenIndex] then
                        Array.Copy(
                            tensorValues,
                            tensorOffset + tokenIndex * embeddingDim,
                            vectors,
                            target * embeddingDim,
                            embeddingDim
                        )

                        target <- target + 1

                vectors

        { tokenIds = tokenIds
          vectors = vectors
          tokenCount = tokenIds.Length
          embeddingDim = embeddingDim }

    let tensorBatchToVectors mode (tokenizedBatch: TokenizedInput array) (tensor: Tensor<float32>) =
        let dimensions = tensor.Dimensions.ToArray()

        let batchSize, sequenceLength, embeddingDim =
            match dimensions with
            | [| batch; sequence; dim |] -> batch, sequence, dim
            | [| sequence; dim |] when tokenizedBatch.Length = 1 -> 1, sequence, dim
            | _ ->
                let shape = String.Join("x", dimensions)
                failwith $"Unexpected ONNX output shape: {shape}"

        if batchSize <> tokenizedBatch.Length then
            failwith $"Expected ONNX batch size {tokenizedBatch.Length}, got {batchSize}."

        if embeddingDim <> config.embeddingDim then
            failwith $"Expected embedding dimension {config.embeddingDim}, got {embeddingDim}."

        let values = tensor.ToDenseTensor().Buffer.ToArray()
        let vectorStride = sequenceLength * embeddingDim

        tokenizedBatch
        |> Array.mapi (fun batchIndex tokenized ->
            tensorValuesToVectors mode tokenized sequenceLength embeddingDim (batchIndex * vectorStride) values)

    let runBatch mode (tokenizedBatch: TokenizedInput array) =
        let batchSize = tokenizedBatch.Length
        let length = tokenizedBatch[0].inputIds.Length
        let inputIds = Array.zeroCreate<int64> (batchSize * length)
        let attentionMask = Array.zeroCreate<int64> (batchSize * length)

        for batchIndex = 0 to batchSize - 1 do
            let offset = batchIndex * length
            Array.Copy(tokenizedBatch[batchIndex].inputIds, 0, inputIds, offset, length)
            Array.Copy(tokenizedBatch[batchIndex].attentionMask, 0, attentionMask, offset, length)

        let inputs = ResizeArray<NamedOnnxValue>()

        tryAddInput inputs "input_ids" inputIds batchSize length
        tryAddInput inputs "attention_mask" attentionMask batchSize length

        if inputs.Count = 0 then
            failwith "ONNX model does not expose supported input names. Expected input_ids and attention_mask."

        use results = session.Run(inputs)

        let output =
            results
            |> Seq.tryFind (fun value -> value.Name = outputName)
            |> Option.defaultValue (results |> Seq.head)

        output.AsTensor<float32>() |> tensorBatchToVectors mode tokenizedBatch

    member _.Config = config

    member _.Tokenizer = tokenizer

    member _.EncodeBatch(mode: EncoderMode, texts: string array) : EncodedText array =
        lock gate (fun () ->
            if Array.isEmpty texts then
                [||]
            else
                let tokenizedBatch = texts |> Array.map (fun text -> tokenizer.Tokenize(mode, text))

                let embeddings =
                    if texts.Length = 1 then
                        runBatch mode tokenizedBatch
                    else
                        try
                            runBatch mode tokenizedBatch
                        with _ ->
                            tokenizedBatch |> Array.collect (fun tokenized -> runBatch mode [| tokenized |])

                Array.map2
                    (fun tokenized embedding ->
                        { mode = mode
                          tokenized = tokenized
                          embedding = embedding })
                    tokenizedBatch
                    embeddings)

    member this.Encode(mode: EncoderMode, text: string) : EncodedText = this.EncodeBatch(mode, [| text |])[0]

    interface IDisposable with
        member _.Dispose() = session.Dispose()

type OnnxColbertEncoder private (runners: OnnxEncoderRunner array, config: EncoderConfig) =
    let runnerLock = obj ()
    let mutable nextRunnerIndex = 0

    let nextRunner () =
        lock runnerLock (fun () ->
            if runners.Length = 0 then
                failwith "ONNX encoder has no model runners."

            let index = nextRunnerIndex
            nextRunnerIndex <- (nextRunnerIndex + 1) % runners.Length
            runners[index])

    static let tryEnvironmentInt name =
        match Environment.GetEnvironmentVariable name with
        | value when String.IsNullOrWhiteSpace value -> None
        | value ->
            match Int32.TryParse value with
            | true, parsed when parsed > 0 -> Some parsed
            | _ -> None

    static let createSessionOptions runtimeOptions =
        let options = new SessionOptions()
        options.GraphOptimizationLevel <- GraphOptimizationLevel.ORT_ENABLE_ALL

        runtimeOptions.intraOpThreads
        |> Option.iter (fun threads -> options.IntraOpNumThreads <- max 1 threads)

        runtimeOptions.interOpThreads
        |> Option.iter (fun threads -> options.InterOpNumThreads <- max 1 threads)

        options

    static let createRunner files config runtimeOptions =
        let tokenizer = ColbertTokenizer.Load(files.tokenizerPath, config)

        let session =
            new InferenceSession(files.modelPath, createSessionOptions runtimeOptions)

        new OnnxEncoderRunner(session, tokenizer, config)

    member _.Config = config

    member _.Tokenizer = runners[0].Tokenizer

    member _.ReplicaCount = runners.Length

    member this.EncodeBatch(mode: EncoderMode, texts: string array) : EncodedText array =
        try
            (nextRunner ()).EncodeBatch(mode, texts)
        with ex ->
            Log.log.Value.exn (ex, nameof (this.EncodeBatch))
            raise ex

    member this.Encode(mode: EncoderMode, text: string) : EncodedText =
        try
            (nextRunner ()).Encode(mode, text)
        with ex ->
            Log.log.Value.exn (ex, nameof (this.Encode))
            raise ex

    member this.EncodeQuery(text: string) = this.Encode(Query, text)

    member this.EncodeDocument(text: string) = this.Encode(Document, text)

    member this.EncodeDocuments(texts: string array) = this.EncodeBatch(Document, texts)

    member this.EncodeQueryAsync(text: string) = async { return this.EncodeQuery text }

    member this.EncodeDocumentAsync(text: string) =
        async { return this.EncodeDocument text }

    member this.EncodeDocumentsAsync(texts: string array) =
        async { return this.EncodeDocuments texts }

    interface IDisposable with
        member _.Dispose() =
            for runner in runners do
                (runner :> IDisposable).Dispose()

    static member Load(files: ModelFiles, ?runtimeOptions: RuntimeOptions, ?replicaCount: int) =
        let config = ModelConfig.loadOptional files.configPath
        let runtimeOptions = defaultArg runtimeOptions RuntimeOptions.defaults

        let replicaCount =
            replicaCount
            |> Option.orElse (tryEnvironmentInt "FSCOLBERT_MODEL_REPLICAS")
            |> Option.defaultValue 1
            |> max 1

        let runners =
            Array.init replicaCount (fun _ -> createRunner files config runtimeOptions)

        new OnnxColbertEncoder(runners, config)
