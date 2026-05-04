namespace FsColbert

open System
open System.Collections.Generic
open System.IO
open System.Text
open System.Text.Json
open Microsoft.ML.Tokenizers

module internal TokenizerJson =
    type Spec =
        { vocabulary: KeyValuePair<string, int> list
          merges: string list
          addedTokens: Map<string, int>
          clsTokenId: int option
          sepTokenId: int option }

    let private readMerges (model: JsonElement) =
        match model.TryGetProperty "merges" with
        | false, _ -> []
        | true, merges when merges.ValueKind <> JsonValueKind.Array -> []
        | true, merges ->
            merges.EnumerateArray()
            |> Seq.choose (fun item ->
                match item.ValueKind with
                | JsonValueKind.String -> item.GetString() |> Option.ofObj
                | JsonValueKind.Array ->
                    let pieces =
                        item.EnumerateArray()
                        |> Seq.choose (fun part ->
                            if part.ValueKind = JsonValueKind.String then
                                Some(part.GetString())
                            else
                                None)
                        |> Seq.choose Option.ofObj
                        |> Seq.toList

                    match pieces with
                    | left :: right :: _ -> Some $"{left} {right}"
                    | _ -> None
                | _ -> None)
            |> Seq.toList

    let private readVocabulary (model: JsonElement) =
        model.GetProperty("vocab").EnumerateObject()
        |> Seq.map (fun prop -> KeyValuePair(prop.Name, prop.Value.GetInt32()))
        |> Seq.toList

    let private readAddedTokens (root: JsonElement) =
        match root.TryGetProperty "added_tokens" with
        | false, _ -> Map.empty
        | true, tokens when tokens.ValueKind <> JsonValueKind.Array -> Map.empty
        | true, tokens ->
            tokens.EnumerateArray()
            |> Seq.choose (fun token ->
                match token.TryGetProperty "content", token.TryGetProperty "id" with
                | (true, content), (true, id) when
                    content.ValueKind = JsonValueKind.String && id.ValueKind = JsonValueKind.Number
                    ->
                    content.GetString()
                    |> Option.ofObj
                    |> Option.map (fun value -> value, id.GetInt32())
                | _ -> None)
            |> Map.ofSeq

    let load (path: string) =
        use document = JsonDocument.Parse(File.ReadAllText path)
        let root = document.RootElement
        let model = root.GetProperty "model"
        let addedTokens = readAddedTokens root

        let vocabulary =
            readVocabulary model
            |> fun vocab ->
                let existing = vocab |> Seq.map _.Key |> Set.ofSeq

                addedTokens
                |> Map.toSeq
                |> Seq.filter (fun (token, _) -> not (existing.Contains token))
                |> Seq.fold (fun acc (token, id) -> KeyValuePair(token, id) :: acc) vocab

        { vocabulary = vocabulary
          merges = readMerges model
          addedTokens = addedTokens
          clsTokenId = addedTokens |> Map.tryFind "[CLS]"
          sepTokenId = addedTokens |> Map.tryFind "[SEP]" }

type ColbertTokenizer private (tokenizer: BpeTokenizer, config: EncoderConfig, skipTokenIds: Set<int>) =
    let normalizeInput (value: string) =
        let normalized =
            (defaultArg (Option.ofObj value) "").Normalize(NormalizationForm.FormC)

        if config.doLowerCase then
            normalized.ToLowerInvariant()
        else
            normalized

    let prefixId mode =
        match mode with
        | Query -> config.queryPrefixId
        | Document -> config.documentPrefixId

    let maxLength mode =
        match mode with
        | Query -> config.queryLength
        | Document -> config.documentLength

    let shouldKeepToken mode tokenId =
        tokenId <> config.padTokenId
        && tokenId <> config.maskTokenId
        && (mode = Query || not (skipTokenIds.Contains tokenId))

    member _.Config = config

    member _.SkipTokenIds = skipTokenIds

    member _.Tokenize(mode: EncoderMode, text: string) : TokenizedInput =
        let length = maxLength mode
        let inputIds = Array.create length (int64 config.padTokenId)
        let attentionMask = Array.zeroCreate<int64> length

        let bodyCapacity = max 0 (length - 3)

        let bodyIds =
            tokenizer.EncodeToIds(normalizeInput text)
            |> Seq.truncate bodyCapacity
            |> Seq.toArray

        let ids =
            [| yield config.clsTokenId
               yield prefixId mode
               yield! bodyIds
               yield config.sepTokenId |]
            |> Array.truncate length

        for index = 0 to ids.Length - 1 do
            inputIds[index] <- int64 ids[index]
            attentionMask[index] <- 1L

        { inputIds = inputIds
          attentionMask = attentionMask
          effectiveLength = ids.Length }

    member _.OutputTokenMask(mode: EncoderMode, tokenized: TokenizedInput) =
        tokenized.inputIds
        |> Array.mapi (fun index tokenId -> tokenized.attentionMask[index] = 1L && shouldKeepToken mode (int tokenId))

    static member Load(tokenizerPath: string, config: EncoderConfig) =
        let spec = TokenizerJson.load tokenizerPath

        let config =
            { config with
                clsTokenId = defaultArg spec.clsTokenId config.clsTokenId
                sepTokenId = defaultArg spec.sepTokenId config.sepTokenId }

        let specialTokens =
            spec.addedTokens |> Map.toSeq |> dict :?> IReadOnlyDictionary<string, int>

        let options = BpeOptions(spec.vocabulary)
        options.Merges <- spec.merges
        options.ByteLevel <- true
        options.PreTokenizer <- RobertaPreTokenizer.Instance
        options.SpecialTokens <- specialTokens

        match spec.addedTokens |> Map.tryFind "[UNK]" with
        | Some _ -> options.UnknownToken <- "[UNK]"
        | None -> ()

        let tokenizer = BpeTokenizer.Create options

        let skipTokenIds =
            config.skiplistWords
            |> Seq.collect (fun word -> tokenizer.EncodeToIds(word) |> Seq.map int)
            |> Set.ofSeq
            |> Set.add config.padTokenId
            |> Set.add config.maskTokenId

        ColbertTokenizer(tokenizer, config, skipTokenIds)
