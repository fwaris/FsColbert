namespace FsColbert

open System.IO
open System.Text.Json

module ModelConfig =
    let private tryProperty (name: string) (element: JsonElement) =
        let mutable value = Unchecked.defaultof<JsonElement>

        if element.TryGetProperty(name, &value) then
            Some value
        else
            None

    let private tryGetInt name (element: JsonElement) =
        match tryProperty name element with
        | Some value when value.ValueKind = JsonValueKind.Number -> Some(value.GetInt32())
        | _ -> None

    let private tryGetBool name (element: JsonElement) =
        match tryProperty name element with
        | Some value when value.ValueKind = JsonValueKind.True -> Some true
        | Some value when value.ValueKind = JsonValueKind.False -> Some false
        | _ -> None

    let private tryGetStringSet name (element: JsonElement) =
        match tryProperty name element with
        | Some value when value.ValueKind = JsonValueKind.Array ->
            value.EnumerateArray()
            |> Seq.choose (fun item ->
                if item.ValueKind = JsonValueKind.String then
                    Some(item.GetString())
                else
                    None)
            |> Seq.choose Option.ofObj
            |> Set.ofSeq
            |> Some
        | _ -> None

    let load (path: string) =
        use document = JsonDocument.Parse(File.ReadAllText path)
        let root = document.RootElement
        let defaults = EncoderConfig.mxbaiEdgeColbert

        { defaults with
            queryLength = defaultArg (tryGetInt "query_length" root) defaults.queryLength
            documentLength = defaultArg (tryGetInt "document_length" root) defaults.documentLength
            embeddingDim = defaultArg (tryGetInt "embedding_dim" root) defaults.embeddingDim
            padTokenId = defaultArg (tryGetInt "pad_token_id" root) defaults.padTokenId
            maskTokenId = defaultArg (tryGetInt "mask_token_id" root) defaults.maskTokenId
            queryPrefixId = defaultArg (tryGetInt "query_prefix_id" root) defaults.queryPrefixId
            documentPrefixId = defaultArg (tryGetInt "document_prefix_id" root) defaults.documentPrefixId
            doLowerCase = defaultArg (tryGetBool "do_lower_case" root) defaults.doLowerCase
            skiplistWords = defaultArg (tryGetStringSet "skiplist_words" root) defaults.skiplistWords }

    let loadOptional (path: string option) =
        path
        |> Option.filter File.Exists
        |> Option.map load
        |> Option.defaultValue EncoderConfig.mxbaiEdgeColbert
