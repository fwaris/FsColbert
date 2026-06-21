namespace FsColbert

open System
open System.IO
open System.Net.Http

type ModelManifest =
    { id: string
      modelUrl: string
      tokenizerUrl: string
      configUrl: string
      modelFileName: string
      tokenizerFileName: string
      configFileName: string }

type DoclingOnnxModelManifest =
    { id: string
      modelUrl: string
      configUrl: string
      preprocessorConfigUrl: string
      modelFileName: string
      configFileName: string
      preprocessorConfigFileName: string }

module ModelCatalog =
    let mxbaiEdgeColbertInt8 =
        { id = "lightonai/mxbai-edge-colbert-v0-32m-onnx:int8"
          modelUrl = "https://huggingface.co/lightonai/mxbai-edge-colbert-v0-32m-onnx/resolve/main/model_int8.onnx"
          tokenizerUrl = "https://huggingface.co/lightonai/mxbai-edge-colbert-v0-32m-onnx/resolve/main/tokenizer.json"
          configUrl = "https://huggingface.co/lightonai/mxbai-edge-colbert-v0-32m-onnx/resolve/main/onnx_config.json"
          modelFileName = "model_int8.onnx"
          tokenizerFileName = "tokenizer.json"
          configFileName = "onnx_config.json" }

    let doclingLayoutHeronOnnx =
        { id = "docling-project/docling-layout-heron-onnx"
          modelUrl = "https://huggingface.co/docling-project/docling-layout-heron-onnx/resolve/main/model.onnx"
          configUrl = "https://huggingface.co/docling-project/docling-layout-heron-onnx/resolve/main/config.json"
          preprocessorConfigUrl =
            "https://huggingface.co/docling-project/docling-layout-heron-onnx/resolve/main/preprocessor_config.json"
          modelFileName = "model.onnx"
          configFileName = "config.json"
          preprocessorConfigFileName = "preprocessor_config.json" }

    let doclingDocumentFigureClassifierV25Onnx =
        { id = "docling-project/DocumentFigureClassifier-v2.5"
          modelUrl = "https://huggingface.co/docling-project/DocumentFigureClassifier-v2.5/resolve/main/model.onnx"
          configUrl = "https://huggingface.co/docling-project/DocumentFigureClassifier-v2.5/resolve/main/config.json"
          preprocessorConfigUrl =
            "https://huggingface.co/docling-project/DocumentFigureClassifier-v2.5/resolve/main/preprocessor_config.json"
          modelFileName = "model.onnx"
          configFileName = "config.json"
          preprocessorConfigFileName = "preprocessor_config.json" }

    let private downloadFileAsync (client: HttpClient) url path =
        async {
            if not (File.Exists path) then
                use! response = client.GetAsync(Uri url) |> Async.AwaitTask
                response.EnsureSuccessStatusCode() |> ignore
                use! source = response.Content.ReadAsStreamAsync() |> Async.AwaitTask
                use target = File.Create path
                do! source.CopyToAsync target |> Async.AwaitTask
        }

    let tryResolveLocalFiles (candidateFolders: string seq) (manifest: ModelManifest) =
        let tryResolveFromFolder folder =
            if String.IsNullOrWhiteSpace folder then
                None
            else
                let modelPath = Path.Combine(folder, manifest.modelFileName)
                let tokenizerPath = Path.Combine(folder, manifest.tokenizerFileName)
                let configPath = Path.Combine(folder, manifest.configFileName)

                if File.Exists modelPath && File.Exists tokenizerPath && File.Exists configPath then
                    Some
                        { modelPath = modelPath
                          tokenizerPath = tokenizerPath
                          configPath = Some configPath }
                else
                    None

        candidateFolders |> Seq.tryPick tryResolveFromFolder

    let ensureDownloadedAsync (client: HttpClient) (folder: string) (manifest: ModelManifest) =
        async {
            Directory.CreateDirectory folder |> ignore

            let modelPath = Path.Combine(folder, manifest.modelFileName)
            let tokenizerPath = Path.Combine(folder, manifest.tokenizerFileName)
            let configPath = Path.Combine(folder, manifest.configFileName)

            do! downloadFileAsync client manifest.modelUrl modelPath
            do! downloadFileAsync client manifest.tokenizerUrl tokenizerPath
            do! downloadFileAsync client manifest.configUrl configPath

            return
                { modelPath = modelPath
                  tokenizerPath = tokenizerPath
                  configPath = Some configPath }
        }

    let ensureAvailableAsync
        (client: HttpClient)
        (downloadFolder: string)
        (candidateFolders: string seq)
        (manifest: ModelManifest)
        =
        async {
            match tryResolveLocalFiles candidateFolders manifest with
            | Some files -> return files
            | None -> return! ensureDownloadedAsync client downloadFolder manifest
        }

    let ensureDoclingOnnxDownloadedAsync (client: HttpClient) (folder: string) (manifest: DoclingOnnxModelManifest) =
        async {
            Directory.CreateDirectory folder |> ignore

            let modelPath = Path.Combine(folder, manifest.modelFileName)
            let configPath = Path.Combine(folder, manifest.configFileName)

            let preprocessorConfigPath =
                Path.Combine(folder, manifest.preprocessorConfigFileName)

            do! downloadFileAsync client manifest.modelUrl modelPath
            do! downloadFileAsync client manifest.configUrl configPath
            do! downloadFileAsync client manifest.preprocessorConfigUrl preprocessorConfigPath

            return
                { modelPath = modelPath
                  configPath = configPath
                  preprocessorConfigPath = preprocessorConfigPath }
        }
