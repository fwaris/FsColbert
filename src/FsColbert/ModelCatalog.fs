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

module ModelCatalog =
    let mxbaiEdgeColbertInt8 =
        { id = "lightonai/mxbai-edge-colbert-v0-32m-onnx:int8"
          modelUrl = "https://huggingface.co/lightonai/mxbai-edge-colbert-v0-32m-onnx/resolve/main/model_int8.onnx"
          tokenizerUrl = "https://huggingface.co/lightonai/mxbai-edge-colbert-v0-32m-onnx/resolve/main/tokenizer.json"
          configUrl = "https://huggingface.co/lightonai/mxbai-edge-colbert-v0-32m-onnx/resolve/main/onnx_config.json"
          modelFileName = "model_int8.onnx"
          tokenizerFileName = "tokenizer.json"
          configFileName = "onnx_config.json" }

    let private downloadFileAsync (client: HttpClient) url path =
        async {
            if not (File.Exists path) then
                use! response = client.GetAsync(Uri url) |> Async.AwaitTask
                response.EnsureSuccessStatusCode() |> ignore
                use! source = response.Content.ReadAsStreamAsync() |> Async.AwaitTask
                use target = File.Create path
                do! source.CopyToAsync target |> Async.AwaitTask
        }

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
