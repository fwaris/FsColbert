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

    let flattenMaskedVectors embeddingDim (mask: bool array) (tensorValues: float32 array) =
        let kept = mask |> Array.filter id |> Array.length
        let vectors = Array.zeroCreate<float32> (kept * embeddingDim)
        let mutable target = 0

        for tokenIndex = 0 to mask.Length - 1 do
            if mask[tokenIndex] then
                Array.Copy(tensorValues, tokenIndex * embeddingDim, vectors, target * embeddingDim, embeddingDim)
                normalizeInPlace vectors (target * embeddingDim) embeddingDim
                target <- target + 1

        vectors

type OnnxColbertEncoder private (session: InferenceSession, tokenizer: ColbertTokenizer, config: EncoderConfig) =
    let inputNames =
        session.InputMetadata.Keys
        |> Seq.map (fun name -> name.ToLowerInvariant(), name)
        |> Map.ofSeq

    let outputName =
        session.OutputMetadata.Keys
        |> Seq.tryHead
        |> Option.defaultWith (fun () -> failwith "ONNX model has no outputs.")

    let makeInput name (data: int64 array) length =
        let tensor = DenseTensor<int64>(data, [| 1; length |])
        NamedOnnxValue.CreateFromTensor(name, tensor)

    let tryAddInput (inputs: ResizeArray<NamedOnnxValue>) logicalName data length =
        inputNames
        |> Map.tryFind logicalName
        |> Option.iter (fun name -> inputs.Add(makeInput name data length))

    let tensorToVectors mode (tokenized: TokenizedInput) (tensor: Tensor<float32>) =
        let dimensions = tensor.Dimensions.ToArray()

        let sequenceLength, embeddingDim =
            match dimensions with
            | [| 1; sequence; dim |] -> sequence, dim
            | [| sequence; dim |] -> sequence, dim
            | _ ->
                let shape = String.Join("x", dimensions)
                failwith $"Unexpected ONNX output shape: {shape}"

        if embeddingDim <> config.embeddingDim then
            failwith $"Expected embedding dimension {config.embeddingDim}, got {embeddingDim}."

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

        let values = tensor.ToDenseTensor().Buffer.ToArray()

        let vectors =
            if config.normalizeOutput then
                VectorMath.flattenMaskedVectors embeddingDim tokenMask values
            else
                let kept = tokenMask |> Array.filter id |> Array.length
                let vectors = Array.zeroCreate<float32> (kept * embeddingDim)
                let mutable target = 0

                for tokenIndex = 0 to tokenMask.Length - 1 do
                    if tokenMask[tokenIndex] then
                        Array.Copy(values, tokenIndex * embeddingDim, vectors, target * embeddingDim, embeddingDim)
                        target <- target + 1

                vectors

        { tokenIds = tokenIds
          vectors = vectors
          tokenCount = tokenIds.Length
          embeddingDim = embeddingDim }

    member _.Config = config

    member _.Tokenizer = tokenizer

    member this.Encode(mode: EncoderMode, text: string) : EncodedText =
        try
            let tokenized = tokenizer.Tokenize(mode, text)
            let inputs = ResizeArray<NamedOnnxValue>()

            tryAddInput inputs "input_ids" tokenized.inputIds tokenized.inputIds.Length
            tryAddInput inputs "attention_mask" tokenized.attentionMask tokenized.inputIds.Length

            if inputs.Count = 0 then
                failwith "ONNX model does not expose supported input names. Expected input_ids and attention_mask."

            use results = session.Run(inputs)

            let output =
                results
                |> Seq.tryFind (fun value -> value.Name = outputName)
                |> Option.defaultValue (results |> Seq.head)

            let tensor = output.AsTensor<float32>()

            { mode = mode
              tokenized = tokenized
              embedding = tensorToVectors mode tokenized tensor }
        with ex ->
            Log.log.Value.exn(ex,nameof(this.Encode))
            raise ex

    member this.EncodeQuery(text: string) = this.Encode(Query, text)

    member this.EncodeDocument(text: string) = this.Encode(Document, text)

    member this.EncodeQueryAsync(text: string) = async { return this.EncodeQuery text }

    member this.EncodeDocumentAsync(text: string) =
        async { return this.EncodeDocument text }

    interface IDisposable with
        member _.Dispose() = session.Dispose()

    static member Load(files: ModelFiles, ?runtimeOptions: RuntimeOptions) =
        let config = ModelConfig.loadOptional files.configPath
        let tokenizer = ColbertTokenizer.Load(files.tokenizerPath, config)
        let runtimeOptions = defaultArg runtimeOptions RuntimeOptions.defaults
        let options = new SessionOptions()
        options.GraphOptimizationLevel <- GraphOptimizationLevel.ORT_ENABLE_ALL

        runtimeOptions.intraOpThreads
        |> Option.iter (fun threads -> options.IntraOpNumThreads <- max 1 threads)

        runtimeOptions.interOpThreads
        |> Option.iter (fun threads -> options.InterOpNumThreads <- max 1 threads)

        let session = new InferenceSession(files.modelPath, options)
        new OnnxColbertEncoder(session, tokenizer, config)
