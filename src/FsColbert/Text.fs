namespace FsColbert

open System
open System.Text.RegularExpressions

module Text =
    let private termMatches (value: string) =
        let value = defaultArg (Option.ofObj value) ""

        Regex.Matches(value.ToLowerInvariant(), "[a-z0-9][a-z0-9_-]{2,}")
        |> Seq.cast<Match>
        |> Seq.map _.Value

    let normalizeWhitespace (value: string) =
        Regex.Replace(defaultArg (Option.ofObj value) "", @"\s+", " ").Trim()

    let terms (value: string) =
        termMatches value |> Seq.distinct |> Set.ofSeq

    let termFrequencies (value: string) =
        let tokens = termMatches value |> Seq.toList
        let total = tokens.Length

        let frequencies =
            tokens
            |> List.fold
                (fun counts term ->
                    let count = counts |> Map.tryFind term |> Option.defaultValue 0
                    counts |> Map.add term (count + 1))
                Map.empty

        frequencies, total

    let termFrequenciesFromValues (values: string seq) =
        let tokens = values |> Seq.collect termMatches |> Seq.toList

        let frequencies =
            tokens
            |> List.fold
                (fun counts term ->
                    let count = counts |> Map.tryFind term |> Option.defaultValue 0
                    counts |> Map.add term (count + 1))
                Map.empty

        frequencies, tokens.Length

    let truncate maxChars (value: string) =
        if String.IsNullOrEmpty value || value.Length <= maxChars then
            value
        else
            value.Substring(0, maxChars).TrimEnd() + "..."

    let chunkText (options: ChunkOptions) (text: string) =
        let text = normalizeWhitespace text

        if String.IsNullOrWhiteSpace text then
            []
        else
            let maxChars = max 1 options.maxChars
            let overlap = min (max 0 options.overlapChars) (maxChars - 1)
            let step = max 1 (maxChars - overlap)

            let rec loop index offset acc =
                if offset >= text.Length then
                    List.rev acc
                else
                    let length = min maxChars (text.Length - offset)
                    let snip = text.Substring(offset, length).Trim()

                    let acc =
                        if snip.Length < options.minChars then
                            acc
                        else
                            (index, snip) :: acc

                    loop (index + 1) (offset + step) acc

            loop 0 0 []

    let splitPassages (options: ChunkOptions) (source: SourceDocument) =
        source.text
        |> chunkText options
        |> List.map (fun (index, text) ->
            { sourceId = source.id
              sourceDisplayName = source.displayName
              sourceLocation = source.location
              index = index
              text = text })
