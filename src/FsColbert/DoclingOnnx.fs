namespace FsColbert

open System
open System.IO
open System.Linq
open System.Text.Json
open System.Threading
open Microsoft.ML.OnnxRuntime
open Microsoft.ML.OnnxRuntime.Tensors

type private DoclingImagePreprocessorConfig =
    { width: int
      height: int
      doRescale: bool
      rescaleFactor: float32
      doNormalize: bool
      imageMean: float32 array
      imageStd: float32 array }

module private DoclingOnnxConfig =
    let private tryProperty (name: string) (element: JsonElement) =
        let mutable value = Unchecked.defaultof<JsonElement>

        if element.TryGetProperty(name, &value) then
            Some value
        else
            None

    let private tryGetBool name (element: JsonElement) =
        match tryProperty name element with
        | Some value when value.ValueKind = JsonValueKind.True -> Some true
        | Some value when value.ValueKind = JsonValueKind.False -> Some false
        | _ -> None

    let private tryGetFloat32 name (element: JsonElement) =
        match tryProperty name element with
        | Some value when value.ValueKind = JsonValueKind.Number -> Some(float32 (value.GetDouble()))
        | _ -> None

    let private tryGetInt name (element: JsonElement) =
        match tryProperty name element with
        | Some value when value.ValueKind = JsonValueKind.Number -> Some(value.GetInt32())
        | _ -> None

    let private tryGetFloat32Array name (element: JsonElement) =
        match tryProperty name element with
        | Some value when value.ValueKind = JsonValueKind.Array ->
            value.EnumerateArray()
            |> Seq.choose (fun item ->
                if item.ValueKind = JsonValueKind.Number then
                    Some(float32 (item.GetDouble()))
                else
                    None)
            |> Seq.toArray
            |> Some
        | _ -> None

    let loadPreprocessor path =
        use document = JsonDocument.Parse(File.ReadAllText path)
        let root = document.RootElement

        let width, height =
            match tryProperty "size" root with
            | Some size -> defaultArg (tryGetInt "width" size) 224, defaultArg (tryGetInt "height" size) 224
            | None -> 224, 224

        { width = width
          height = height
          doRescale = defaultArg (tryGetBool "do_rescale" root) true
          rescaleFactor = defaultArg (tryGetFloat32 "rescale_factor" root) (1.0f / 255.0f)
          doNormalize = defaultArg (tryGetBool "do_normalize" root) true
          imageMean = defaultArg (tryGetFloat32Array "image_mean" root) [| 0.485f; 0.456f; 0.406f |]
          imageStd = defaultArg (tryGetFloat32Array "image_std" root) [| 0.229f; 0.224f; 0.225f |] }

    let loadLabels path fallback =
        use document = JsonDocument.Parse(File.ReadAllText path)
        let root = document.RootElement

        match tryProperty "id2label" root with
        | Some id2Label when id2Label.ValueKind = JsonValueKind.Object ->
            id2Label.EnumerateObject()
            |> Seq.choose (fun property ->
                match Int32.TryParse property.Name, property.Value.ValueKind with
                | (true, id), JsonValueKind.String -> Some(id, property.Value.GetString())
                | _ -> None)
            |> Seq.choose (fun (id, label) -> label |> Option.ofObj |> Option.map (fun value -> id, value))
            |> Map.ofSeq
        | _ -> fallback

module private DoclingOnnxTensor =
    let createSessionOptions runtimeOptions =
        let options = new SessionOptions()
        options.GraphOptimizationLevel <- GraphOptimizationLevel.ORT_ENABLE_ALL

        runtimeOptions.intraOpThreads
        |> Option.iter (fun threads -> options.IntraOpNumThreads <- max 1 threads)

        runtimeOptions.interOpThreads
        |> Option.iter (fun threads -> options.InterOpNumThreads <- max 1 threads)

        options

    let firstInputName (session: InferenceSession) =
        session.InputMetadata.Keys
        |> Seq.tryHead
        |> Option.defaultWith (fun () -> failwith "ONNX model has no inputs.")

    let private channelValue (config: DoclingImagePreprocessorConfig) channel value =
        let mutable value = float32 value

        if config.doRescale then
            value <- value * config.rescaleFactor

        if config.doNormalize then
            let mean =
                if channel < config.imageMean.Length then
                    config.imageMean[channel]
                else
                    0.0f

            let std =
                if channel < config.imageStd.Length && config.imageStd[channel] <> 0.0f then
                    config.imageStd[channel]
                else
                    1.0f

            value <- (value - mean) / std

        value

    let imageToNchw (config: DoclingImagePreprocessorConfig) image =
        let resized = DoclingRgbImage.resizeBilinear config.width config.height image
        let planeSize = config.width * config.height
        let values = Array.zeroCreate<float32> (3 * planeSize)

        for y = 0 to config.height - 1 do
            for x = 0 to config.width - 1 do
                let pixelOffset = (y * config.width + x) * 3
                let planeOffset = y * config.width + x

                for channel = 0 to 2 do
                    values[channel * planeSize + planeOffset] <-
                        channelValue config channel resized.pixels[pixelOffset + channel]

        values

    let makeImageInput inputName config image =
        let data = imageToNchw config image
        let dimensions = ReadOnlySpan<int>([| 1; 3; config.height; config.width |])
        let tensor = DenseTensor<float32>(Memory<float32>(data), dimensions)
        NamedOnnxValue.CreateFromTensor(inputName, tensor)

    let tensorValues (tensor: Tensor<float32>) =
        tensor.Dimensions.ToArray(), tensor.ToDenseTensor().Buffer.ToArray()

    let tryAsFloatTensor (value: DisposableNamedOnnxValue) =
        try
            Some(value.Name, value.AsTensor<float32>())
        with _ ->
            None

    let sigmoid value = 1.0f / (1.0f + exp (-value))

    let softmax (values: float32 array) =
        if Array.isEmpty values then
            [||]
        else
            let maxValue = values |> Array.max

            let exps = values |> Array.map (fun value -> exp (value - maxValue))

            let sum = exps |> Array.sum

            if sum = 0.0f then
                Array.zeroCreate values.Length
            else
                exps |> Array.map (fun value -> value / sum)

    let get3 (dims: int array) (values: float32 array) batch row col =
        values[(batch * dims[1] + row) * dims[2] + col]

type DoclingLayoutOnnx(files: DoclingOnnxModelFiles, ?runtimeOptions: RuntimeOptions, ?confidenceThreshold: float32) =
    let runtimeOptions = defaultArg runtimeOptions RuntimeOptions.defaults
    let threshold = defaultArg confidenceThreshold 0.3f
    let preprocessor = DoclingOnnxConfig.loadPreprocessor files.preprocessorConfigPath

    let labels =
        DoclingOnnxConfig.loadLabels
            files.configPath
            ([ 0, "caption"
               1, "footnote"
               2, "formula"
               3, "list_item"
               4, "page_footer"
               5, "page_header"
               6, "picture"
               7, "section_header"
               8, "table"
               9, "text"
               10, "title"
               11, "document_index"
               12, "code"
               13, "checkbox_selected"
               14, "checkbox_unselected"
               15, "form"
               16, "key_value_region" ]
             |> Map.ofList)

    let session =
        new InferenceSession(files.modelPath, DoclingOnnxTensor.createSessionOptions runtimeOptions)

    let inputName = DoclingOnnxTensor.firstInputName session
    let gate = obj ()

    let findTensorByNamePart (part: string) outputs =
        outputs
        |> List.tryFind (fun (name: string, tensor: Tensor<float32>) ->
            name.Contains(part, StringComparison.OrdinalIgnoreCase))

    let findRank3WithLast expectedLast outputs =
        outputs
        |> List.tryFind (fun (_, tensor: Tensor<float32>) ->
            let dims = tensor.Dimensions.ToArray()
            dims.Length = 3 && dims[2] = expectedLast)

    let findRank3NotLast expectedLast outputs =
        outputs
        |> List.tryFind (fun (_, tensor: Tensor<float32>) ->
            let dims = tensor.Dimensions.ToArray()
            dims.Length = 3 && dims[2] <> expectedLast)

    let labelFor labelId =
        labels
        |> Map.tryFind labelId
        |> Option.defaultValue $"label_{labelId}"
        |> DoclingLabels.ofJsonValue

    let boxToBbox originalWidth originalHeight (box: float32 array) =
        let maxValue = box |> Array.max

        if maxValue <= 1.5f then
            let cx = float box[0]
            let cy = float box[1]
            let width = float box[2]
            let height = float box[3]

            DoclingGeometry.topLeftBox
                ((cx - width / 2.0) * float originalWidth)
                ((cy - height / 2.0) * float originalHeight)
                ((cx + width / 2.0) * float originalWidth)
                ((cy + height / 2.0) * float originalHeight)
        else
            let scaleX = float originalWidth / float preprocessor.width
            let scaleY = float originalHeight / float preprocessor.height

            DoclingGeometry.topLeftBox
                (float box[0] * scaleX)
                (float box[1] * scaleY)
                (float box[2] * scaleX)
                (float box[3] * scaleY)

    let predictOne pageNo image =
        let input = DoclingOnnxTensor.makeImageInput inputName preprocessor image
        use results = session.Run([ input ])

        let outputs = results |> Seq.choose DoclingOnnxTensor.tryAsFloatTensor |> Seq.toList

        let logits =
            findTensorByNamePart "logit" outputs
            |> Option.orElse (findRank3NotLast 4 outputs)
            |> Option.defaultWith (fun () -> failwith "Layout ONNX output did not include rank-3 logits.")

        let boxes =
            findTensorByNamePart "box" outputs
            |> Option.orElse (findRank3WithLast 4 outputs)
            |> Option.defaultWith (fun () -> failwith "Layout ONNX output did not include rank-3 boxes.")

        let logitsDims, logitsValues = DoclingOnnxTensor.tensorValues (snd logits)
        let boxesDims, boxesValues = DoclingOnnxTensor.tensorValues (snd boxes)

        if logitsDims[0] <> 1 || boxesDims[0] <> 1 || logitsDims[1] <> boxesDims[1] then
            failwith "Unexpected layout ONNX batch/query dimensions."

        let mutable clusters = []

        for queryIndex = 0 to logitsDims[1] - 1 do
            let mutable bestLabel = 0
            let mutable bestScore = Single.MinValue

            for labelIndex = 0 to logitsDims[2] - 1 do
                let score =
                    DoclingOnnxTensor.sigmoid (DoclingOnnxTensor.get3 logitsDims logitsValues 0 queryIndex labelIndex)

                if score > bestScore then
                    bestScore <- score
                    bestLabel <- labelIndex

            if bestScore >= threshold then
                let box =
                    [| for col = 0 to 3 do
                           DoclingOnnxTensor.get3 boxesDims boxesValues 0 queryIndex col |]

                let bbox =
                    boxToBbox image.width image.height box
                    |> DoclingGeometry.clampToSize
                        { width = float image.width
                          height = float image.height }

                clusters <-
                    { id = queryIndex
                      label = labelFor bestLabel
                      confidence = bestScore
                      bbox = bbox
                      cells = [] }
                    :: clusters

        { pageNo = pageNo
          clusters = List.rev clusters }

    member _.PredictPage(page: DoclingPageInput) =
        lock gate (fun () -> predictOne page.pageNo page.image)

    interface IDoclingLayoutPredictor with
        member this.PredictLayoutAsync pages =
            async {
                try
                    return Ok(pages |> List.map this.PredictPage)
                with ex ->
                    return Error $"Layout ONNX inference failed: {ex.Message}"
            }

    interface ICancelableDoclingLayoutPredictor with
        member this.PredictLayoutAsync(pages, cancellationToken) =
            async {
                try
                    let predictions =
                        pages
                        |> List.map (fun page ->
                            cancellationToken.ThrowIfCancellationRequested()
                            let prediction = this.PredictPage page
                            cancellationToken.ThrowIfCancellationRequested()
                            prediction)

                    return Ok predictions
                with
                | :? OperationCanceledException -> return raise (OperationCanceledException cancellationToken)
                | ex -> return Error $"Layout ONNX inference failed: {ex.Message}"
            }

    interface IDisposable with
        member _.Dispose() = session.Dispose()

type DoclingFigureClassifierOnnx(files: DoclingOnnxModelFiles, ?runtimeOptions: RuntimeOptions, ?topK: int) =
    let runtimeOptions = defaultArg runtimeOptions RuntimeOptions.defaults
    let topK = defaultArg topK 5 |> max 1
    let preprocessor = DoclingOnnxConfig.loadPreprocessor files.preprocessorConfigPath
    let labels = DoclingOnnxConfig.loadLabels files.configPath Map.empty

    let session =
        new InferenceSession(files.modelPath, DoclingOnnxTensor.createSessionOptions runtimeOptions)

    let inputName = DoclingOnnxTensor.firstInputName session

    let outputName =
        session.OutputMetadata.Keys
        |> Seq.tryHead
        |> Option.defaultWith (fun () -> failwith "Figure classifier ONNX model has no outputs.")

    let gate = obj ()

    let classifyOne image =
        let input = DoclingOnnxTensor.makeImageInput inputName preprocessor image
        use results = session.Run([ input ])

        let output =
            results
            |> Seq.tryFind (fun value -> value.Name = outputName)
            |> Option.defaultValue (results |> Seq.head)

        let dims, values = output.AsTensor<float32>() |> DoclingOnnxTensor.tensorValues

        let logits =
            match dims with
            | [| count |] -> values |> Array.take count
            | [| 1; count |] -> values |> Array.take count
            | _ ->
                let shape = String.Join("x", dims)
                failwith $"Unexpected figure classifier output shape: {shape}"

        let probabilities = DoclingOnnxTensor.softmax logits

        probabilities
        |> Array.mapi (fun index confidence ->
            { className = labels |> Map.tryFind index |> Option.defaultValue $"class_{index}"
              confidence = confidence })
        |> Array.sortByDescending _.confidence
        |> Array.truncate topK
        |> Array.toList

    member _.Classify(image: DoclingRgbImage) = lock gate (fun () -> classifyOne image)

    interface IDoclingFigureClassifier with
        member this.ClassifyAsync image =
            async {
                try
                    return Ok(this.Classify image)
                with ex ->
                    return Error $"Figure classifier ONNX inference failed: {ex.Message}"
            }

    interface IDisposable with
        member _.Dispose() = session.Dispose()
