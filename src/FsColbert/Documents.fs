namespace FsColbert

open System
open System.IO
open System.Text.RegularExpressions
open UglyToad.PdfPig
open UglyToad.PdfPig.Content
open UglyToad.PdfPig.DocumentLayoutAnalysis.PageSegmenter
open UglyToad.PdfPig.DocumentLayoutAnalysis.WordExtractor

type PassageSource =
    { id: string
      displayName: string
      location: string }

module PassageSource =
    let create id displayName location =
        { id = id
          displayName = displayName
          location = location }

module DocumentSections =
    let sectionPrefix = "Section:"

    let private headingTerms =
        [ "abstract"
          "acknowledgments"
          "acknowledgements"
          "appendix"
          "background"
          "conclusion"
          "conclusions"
          "discussion"
          "evaluation"
          "experiments"
          "introduction"
          "limitations"
          "method"
          "methods"
          "overview"
          "problem"
          "references"
          "related"
          "results" ]
        |> Set.ofList

    let stripNumbering (text: string) =
        Regex
            .Replace(
                text.Trim().TrimEnd(':'),
                @"^\s*(?:[0-9]+(?:\.[0-9]+)*|[IVXLCDM]+(?:\.[0-9]+)*)\.?\s+",
                "",
                RegexOptions.IgnoreCase
            )
            .Trim()

    let isLikelyHeading (text: string) =
        let heading = stripNumbering text
        let terms = Text.terms heading |> Set.toList
        let letters = heading |> Seq.filter Char.IsLetter |> Seq.toArray

        let uppercaseRatio =
            if letters.Length = 0 then
                0.0
            else
                let upper = letters |> Array.filter Char.IsUpper |> Array.length
                float upper / float letters.Length

        let hasHeadingKeyword =
            terms |> List.exists (fun term -> headingTerms.Contains term)

        let hasSentencePunctuation = Regex.IsMatch(heading, @"[.!?]\s+\p{Lu}")

        heading.Length >= 4
        && heading.Length <= 100
        && terms.Length <= 8
        && not hasSentencePunctuation
        && (uppercaseRatio >= 0.65 || (hasHeadingKeyword && terms.Length <= 5))

    let formatChunk heading chunk =
        match heading with
        | Some heading -> $"{sectionPrefix} {heading}\n{chunk}"
        | None -> chunk

    let tryGetHeading (text: string) =
        if
            String.IsNullOrWhiteSpace text
            || not (text.StartsWith(sectionPrefix, StringComparison.OrdinalIgnoreCase))
        then
            None
        else
            let firstLine =
                text.Replace("\r\n", "\n").Split('\n', 2, StringSplitOptions.None)[0]

            firstLine.Substring(sectionPrefix.Length).Trim()
            |> Option.ofObj
            |> Option.filter (String.IsNullOrWhiteSpace >> not)

    let normalizedName value =
        stripNumbering value
        |> Text.normalizeWhitespace
        |> fun text -> text.ToLowerInvariant()

    let matches requested heading =
        String.Equals(normalizedName requested, normalizedName heading, StringComparison.Ordinal)

module DocumentChunking =
    let representationVersion = "pdf-section-aware-v2"

    type private DocumentSection =
        { heading: string option
          blocks: string list }

    let chunkBlocks (options: ChunkOptions) (blocks: string list) =
        let maxChars = max 1 options.maxChars
        let overlapChars = min (max 0 options.overlapChars) (maxChars - 1)

        let rec loop remaining currentChunk currentLen chunks =
            match remaining with
            | [] ->
                if List.isEmpty currentChunk then
                    chunks
                else
                    (List.rev currentChunk |> String.concat "\n\n") :: chunks
            | (block: string) :: rest ->
                let blockLen = block.Length

                if blockLen > maxChars && List.isEmpty currentChunk then
                    loop rest [] 0 (block :: chunks)
                elif currentLen + blockLen + 2 > maxChars && not (List.isEmpty currentChunk) then
                    let chunkText = List.rev currentChunk |> String.concat "\n\n"

                    let rec getOverlap acc accLen revChunk =
                        match revChunk with
                        | [] -> acc
                        | (b: string) :: bs ->
                            if List.length acc = List.length currentChunk - 1 then
                                acc
                            elif accLen + b.Length + 2 > overlapChars && not (List.isEmpty acc) then
                                acc
                            else
                                getOverlap (b :: acc) (accLen + b.Length + 2) bs

                    let overlapBlocks = getOverlap [] 0 currentChunk

                    let overlapLen =
                        overlapBlocks
                        |> List.sumBy (fun block -> block.Length)
                        |> (+) (
                            if overlapBlocks.Length > 0 then
                                (overlapBlocks.Length - 1) * 2
                            else
                                0
                        )

                    loop remaining overlapBlocks overlapLen (chunkText :: chunks)
                else
                    loop rest (block :: currentChunk) (currentLen + blockLen + 2) chunks

        loop blocks [] 0 [] |> List.rev

    let private splitIntoSections (blocks: string list) =
        let addSection heading current sections =
            if List.isEmpty current then
                sections
            else
                { heading = heading
                  blocks = List.rev current }
                :: sections

        let rec loop remaining heading current sections =
            match remaining with
            | [] -> addSection heading current sections |> List.rev
            | block :: rest when DocumentSections.isLikelyHeading block ->
                let sections = addSection heading current sections
                loop rest (Some(DocumentSections.stripNumbering block)) [] sections
            | block :: rest -> loop rest heading (block :: current) sections

        loop blocks None [] []

    let private enforceChunkBounds (options: ChunkOptions) (chunk: string) =
        if chunk.Length > max 1 options.maxChars then
            Text.chunkText options chunk |> List.map snd
        else
            [ chunk ]

    let chunkSectionedBlocks (options: ChunkOptions) (blocks: string list) =
        blocks
        |> splitIntoSections
        |> List.collect (fun section ->
            section.blocks
            |> chunkBlocks options
            |> List.collect (fun chunk ->
                chunk
                |> DocumentSections.formatChunk section.heading
                |> enforceChunkBounds options))

    let passagesFromBlocks (options: ChunkOptions) (source: PassageSource) (blocks: string list) : PassageRef list =
        blocks
        |> chunkSectionedBlocks options
        |> List.mapi (fun index text ->
            { sourceId = source.id
              sourceDisplayName = source.displayName
              sourceLocation = source.location
              index = index
              text = text })

type PdfReadOptions =
    { includeImageNotes: bool
      filterNumericBlocks: bool }

module PdfReadOptions =
    let defaults =
        { includeImageNotes = true
          filterNumericBlocks = true }

module PdfDocuments =
    let private pageImageNotes (page: Page) =
        let images = page.GetImages() |> Seq.toList

        if List.isEmpty images then
            ""
        else
            images
            |> List.mapi (fun index image ->
                $"[PDF image note: page {page.Number}, image {index + 1}, {image.WidthInSamples}x{image.HeightInSamples} samples. Embedded image detected during PDF processing.]")
            |> String.concat "\n"

    let private shouldKeepBlock options (text: string) =
        not (String.IsNullOrWhiteSpace text)
        && (not options.filterNumericBlocks || not (Regex.IsMatch(text, @"^\d+$")))

    let readBlocksWithOptions (options: PdfReadOptions) path =
        async {
            if not (File.Exists path) then
                return Error $"PDF not found: {path}"
            else
                try
                    use document = PdfDocument.Open(path)
                    let wordExtractor = NearestNeighbourWordExtractor.Instance
                    let pageSegmenter = DocstrumBoundingBoxes.Instance

                    let blocks =
                        document.GetPages()
                        |> Seq.collect (fun page ->
                            let words = wordExtractor.GetWords(page.Letters)
                            let pageBlocks = pageSegmenter.GetBlocks(words)
                            let blockTexts = pageBlocks |> Seq.map (fun block -> block.Text)

                            if options.includeImageNotes then
                                let imageNote = pageImageNotes page

                                if String.IsNullOrWhiteSpace imageNote then
                                    blockTexts
                                else
                                    Seq.append blockTexts [ imageNote ]
                            else
                                blockTexts)
                        |> Seq.map Text.normalizeWhitespace
                        |> Seq.filter (shouldKeepBlock options)
                        |> Seq.toList

                    return Ok blocks
                with ex ->
                    return Error $"Unable to read PDF '{path}': {ex.Message}"
        }

    let readBlocks path =
        readBlocksWithOptions PdfReadOptions.defaults path

    let readTextWithOptions options path =
        async {
            let! result = readBlocksWithOptions options path
            return result |> Result.map (String.concat "\n\n")
        }

    let readText path =
        readTextWithOptions PdfReadOptions.defaults path

    let readPassagesWithOptions readOptions chunkOptions source path =
        async {
            let! result = readBlocksWithOptions readOptions path
            return result |> Result.map (DocumentChunking.passagesFromBlocks chunkOptions source)
        }

    let readPassages chunkOptions source path =
        readPassagesWithOptions PdfReadOptions.defaults chunkOptions source path
