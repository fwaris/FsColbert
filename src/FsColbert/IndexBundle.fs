namespace FsColbert

open System
open System.Globalization
open System.IO
open System.Text.Json
open System.Text.Json.Nodes

type IndexBundleSource =
    { sourceId: string
      sourceDisplayName: string
      sourceLocation: string option
      sourceKind: string option
      indexFile: string }

type IndexBundleManifest =
    { manifestVersion: int
      bundleId: string
      bundleVersion: string
      modelId: string
      chunkOptions: ChunkOptions
      tfidfOptions: TfidfOptions
      createdAt: DateTimeOffset
      sources: IndexBundleSource list }

type IndexBundleCompatibility =
    { modelId: string
      chunkOptions: ChunkOptions option
      tfidfOptions: TfidfOptions option }

module IndexBundleCompatibility =
    let fsKameDefaults =
        { modelId = ModelCatalog.mxbaiEdgeColbertInt8.id
          chunkOptions = Some ChunkOptions.fsKameDefaults
          tfidfOptions = Some TfidfOptions.defaults }

type LoadedIndexBundleEntry =
    { source: IndexBundleSource
      indexPath: string
      index: ColbertIndex }

type LoadedIndexBundle =
    { manifest: IndexBundleManifest
      indexes: LoadedIndexBundleEntry list }

module IndexBundle =
    let schemaName = "FsColbertIndexBundle"
    let currentManifestVersion = 1

    let create bundleId bundleVersion modelId chunkOptions tfidfOptions sources =
        { manifestVersion = currentManifestVersion
          bundleId = bundleId
          bundleVersion = bundleVersion
          modelId = modelId
          chunkOptions = chunkOptions
          tfidfOptions = tfidfOptions
          createdAt = DateTimeOffset.UtcNow
          sources = sources }

    let private optionStringNode (value: string option) : JsonNode =
        match value with
        | Some value -> JsonValue.Create(value)
        | None -> null

    let private chunkOptionsNode (options: ChunkOptions) =
        let node = JsonObject()
        node["max_chars"] <- JsonValue.Create(options.maxChars)
        node["overlap_chars"] <- JsonValue.Create(options.overlapChars)
        node["min_chars"] <- JsonValue.Create(options.minChars)
        node

    let private tfidfOptionsNode (options: TfidfOptions) =
        let node = JsonObject()
        node["text_weight"] <- JsonValue.Create(options.textWeight)
        node["keyword_weight"] <- JsonValue.Create(options.keywordWeight)
        node

    let private sourceNode (source: IndexBundleSource) =
        let node = JsonObject()
        node["source_id"] <- JsonValue.Create(source.sourceId)
        node["source_display_name"] <- JsonValue.Create(source.sourceDisplayName)
        node["source_location"] <- optionStringNode source.sourceLocation
        node["source_kind"] <- optionStringNode source.sourceKind
        node["index_file"] <- JsonValue.Create(source.indexFile)
        node

    let toJsonNode (manifest: IndexBundleManifest) =
        let sources = JsonArray()

        for source in manifest.sources do
            sources.Add(sourceNode source)

        let node = JsonObject()
        node["schema_name"] <- JsonValue.Create(schemaName)
        node["manifest_version"] <- JsonValue.Create(manifest.manifestVersion)
        node["bundle_id"] <- JsonValue.Create(manifest.bundleId)
        node["bundle_version"] <- JsonValue.Create(manifest.bundleVersion)
        node["model_id"] <- JsonValue.Create(manifest.modelId)
        node["created_at"] <- JsonValue.Create(manifest.createdAt.ToString("O", CultureInfo.InvariantCulture))
        node["chunk_options"] <- chunkOptionsNode manifest.chunkOptions
        node["tfidf_options"] <- tfidfOptionsNode manifest.tfidfOptions
        node["sources"] <- sources
        node

    let serializeIndented manifest =
        let options = JsonSerializerOptions(WriteIndented = true)
        (toJsonNode manifest).ToJsonString(options)

    let serialize manifest = (toJsonNode manifest).ToJsonString()

    let writeManifest path manifest =
        Path.GetDirectoryName(Path.GetFullPath path)
        |> Option.ofObj
        |> Option.iter (fun folder -> Directory.CreateDirectory folder |> ignore)

        File.WriteAllText(path, serializeIndented manifest)

    let private tryProperty (name: string) (element: JsonElement) =
        if element.ValueKind <> JsonValueKind.Object then
            None
        else
            let mutable value = Unchecked.defaultof<JsonElement>

            if element.TryGetProperty(name, &value) then
                Some value
            else
                None

    let private firstProperty names element =
        names |> List.tryPick (fun name -> tryProperty name element)

    let private stringValue (element: JsonElement) =
        if element.ValueKind = JsonValueKind.String then
            element.GetString() |> Option.ofObj
        else
            None

    let private firstString names element =
        firstProperty names element |> Option.bind stringValue

    let private intProperty name fallback element =
        match tryProperty name element with
        | Some value when value.ValueKind = JsonValueKind.Number ->
            match value.TryGetInt32() with
            | true, result -> result
            | _ -> fallback
        | _ -> fallback

    let private float32Property name fallback element =
        match tryProperty name element with
        | Some value when value.ValueKind = JsonValueKind.Number ->
            match value.TryGetSingle() with
            | true, result -> result
            | _ -> fallback
        | _ -> fallback

    let private parseChunkOptions element =
        { maxChars = intProperty "max_chars" ChunkOptions.fsKameDefaults.maxChars element
          overlapChars = intProperty "overlap_chars" ChunkOptions.fsKameDefaults.overlapChars element
          minChars = intProperty "min_chars" ChunkOptions.fsKameDefaults.minChars element }

    let private parseTfidfOptions element =
        { textWeight = float32Property "text_weight" TfidfOptions.defaults.textWeight element
          keywordWeight = float32Property "keyword_weight" TfidfOptions.defaults.keywordWeight element }

    let private parseSource index element =
        { sourceId =
            firstString [ "source_id"; "id" ] element
            |> Option.defaultValue $"source-{index}"
          sourceDisplayName =
            firstString [ "source_display_name"; "display_name"; "displayName" ] element
            |> Option.defaultValue $"Source {index}"
          sourceLocation = firstString [ "source_location"; "location"; "document_file"; "documentAsset" ] element
          sourceKind = firstString [ "source_kind"; "kind" ] element
          indexFile = firstString [ "index_file"; "indexFile" ] element |> Option.defaultValue "" }

    let private parseSources root =
        match tryProperty "sources" root with
        | Some sources when sources.ValueKind = JsonValueKind.Array ->
            sources.EnumerateArray() |> Seq.mapi parseSource |> Seq.toList
        | _ -> []

    let private parseCreatedAt root =
        match firstString [ "created_at"; "createdAt" ] root with
        | Some value ->
            match DateTimeOffset.TryParse(value, CultureInfo.InvariantCulture, DateTimeStyles.RoundtripKind) with
            | true, parsed -> parsed
            | _ -> DateTimeOffset.MinValue
        | None -> DateTimeOffset.MinValue

    let tryDeserialize (json: string) =
        try
            use parsed = JsonDocument.Parse json
            let root = parsed.RootElement

            match firstString [ "schema_name"; "schemaName" ] root with
            | Some schema when String.Equals(schema, schemaName, StringComparison.Ordinal) ->
                let chunkOptions =
                    tryProperty "chunk_options" root
                    |> Option.orElseWith (fun () -> tryProperty "chunkOptions" root)
                    |> Option.map parseChunkOptions
                    |> Option.defaultValue ChunkOptions.fsKameDefaults

                let tfidfOptions =
                    tryProperty "tfidf_options" root
                    |> Option.orElseWith (fun () -> tryProperty "tfidfOptions" root)
                    |> Option.map parseTfidfOptions
                    |> Option.defaultValue TfidfOptions.defaults

                Ok
                    { manifestVersion = intProperty "manifest_version" currentManifestVersion root
                      bundleId = firstString [ "bundle_id"; "bundleId" ] root |> Option.defaultValue ""
                      bundleVersion = firstString [ "bundle_version"; "bundleVersion" ] root |> Option.defaultValue ""
                      modelId = firstString [ "model_id"; "modelId" ] root |> Option.defaultValue ""
                      chunkOptions = chunkOptions
                      tfidfOptions = tfidfOptions
                      createdAt = parseCreatedAt root
                      sources = parseSources root }
            | Some schema -> Error $"Expected schema_name '{schemaName}', got '{schema}'."
            | None -> Error "Index bundle manifest schema_name is missing."
        with ex ->
            Error $"Unable to parse index bundle manifest: {ex.Message}"

    let readManifest path =
        if File.Exists path then
            File.ReadAllText path |> tryDeserialize
        else
            Error $"Index bundle manifest '{path}' does not exist."

    let private resolveFile baseFolder file =
        if String.IsNullOrWhiteSpace file then
            file
        elif Path.IsPathRooted file then
            file
        else
            Path.Combine(baseFolder, file.Replace('/', Path.DirectorySeparatorChar))

    let private chunkOptionsText options =
        $"max={options.maxChars}, overlap={options.overlapChars}, min={options.minChars}"

    let private tfidfOptionsText options =
        $"text={options.textWeight}, keyword={options.keywordWeight}"

    let private validateManifest baseFolder (compatibility: IndexBundleCompatibility) (manifest: IndexBundleManifest) =
        [ if manifest.manifestVersion <> currentManifestVersion then
              yield
                  $"Unsupported index bundle manifest version {manifest.manifestVersion}; expected {currentManifestVersion}."

          if String.IsNullOrWhiteSpace manifest.bundleId then
              yield "Index bundle manifest bundle_id is required."

          if String.IsNullOrWhiteSpace manifest.modelId then
              yield "Index bundle manifest model_id is required."

          if manifest.modelId <> compatibility.modelId then
              yield $"Index bundle model_id '{manifest.modelId}' does not match expected '{compatibility.modelId}'."

          match compatibility.chunkOptions with
          | Some expected when manifest.chunkOptions <> expected ->
              yield
                  $"Index bundle chunk_options ({chunkOptionsText manifest.chunkOptions}) do not match expected ({chunkOptionsText expected})."
          | _ -> ()

          match compatibility.tfidfOptions with
          | Some expected when manifest.tfidfOptions <> expected ->
              yield
                  $"Index bundle tfidf_options ({tfidfOptionsText manifest.tfidfOptions}) do not match expected ({tfidfOptionsText expected})."
          | _ -> ()

          if List.isEmpty manifest.sources then
              yield "Index bundle manifest must contain at least one source."

          for source in manifest.sources do
              if String.IsNullOrWhiteSpace source.sourceId then
                  yield "Index bundle source_id is required."

              if String.IsNullOrWhiteSpace source.sourceDisplayName then
                  yield $"Index bundle source '{source.sourceId}' source_display_name is required."

              if String.IsNullOrWhiteSpace source.indexFile then
                  yield $"Index bundle source '{source.sourceId}' index_file is required."
              else
                  let indexPath = resolveFile baseFolder source.indexFile

                  if not (File.Exists indexPath) then
                      yield $"Index bundle source '{source.sourceId}' index file '{source.indexFile}' was not found." ]

    let private loadSourceIndex baseFolder (manifest: IndexBundleManifest) source =
        let indexPath = resolveFile baseFolder source.indexFile

        try
            let index = IndexPersistence.load indexPath

            let errors =
                [ if index.chunkOptions <> manifest.chunkOptions then
                      yield
                          $"Index '{source.indexFile}' chunk options ({chunkOptionsText index.chunkOptions}) do not match manifest ({chunkOptionsText manifest.chunkOptions})."

                  if index.tfidfOptions <> manifest.tfidfOptions then
                      yield
                          $"Index '{source.indexFile}' TF-IDF options ({tfidfOptionsText index.tfidfOptions}) do not match manifest ({tfidfOptionsText manifest.tfidfOptions})." ]

            if List.isEmpty errors then
                Ok
                    { source = source
                      indexPath = indexPath
                      index = index }
            else
                Error errors
        with ex ->
            Error [ $"Unable to load index '{source.indexFile}' for source '{source.sourceId}': {ex.Message}" ]

    let loadCompatible (compatibility: IndexBundleCompatibility) manifestPath =
        match readManifest manifestPath with
        | Error err -> Error [ err ]
        | Ok manifest ->
            let baseFolder =
                Path.GetDirectoryName(Path.GetFullPath manifestPath)
                |> Option.ofObj
                |> Option.defaultValue "."

            let manifestErrors = validateManifest baseFolder compatibility manifest

            if not (List.isEmpty manifestErrors) then
                Error manifestErrors
            else
                let loaded = manifest.sources |> List.map (loadSourceIndex baseFolder manifest)

                let errors =
                    loaded
                    |> List.choose (function
                        | Ok _ -> None
                        | Error errs -> Some errs)
                    |> List.collect id

                if not (List.isEmpty errors) then
                    Error errors
                else
                    Ok
                        { manifest = manifest
                          indexes =
                            loaded
                            |> List.choose (function
                                | Ok entry -> Some entry
                                | Error _ -> None) }
