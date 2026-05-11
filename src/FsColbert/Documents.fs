namespace FsColbert

open System
open System.IO
open System.Text.RegularExpressions
open F23.StringSimilarity
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

    let private levenshtein = Levenshtein()

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

    let private maxEditDistance length =
        if length <= 6 then 1
        elif length <= 12 then 2
        else 3

    let private nearlyMatches requested heading =
        let requestedName = normalizedName requested
        let headingName = normalizedName heading

        not (String.IsNullOrWhiteSpace requestedName)
        && not (String.IsNullOrWhiteSpace headingName)
        && levenshtein.Distance(requestedName, headingName) <= maxEditDistance (max requestedName.Length headingName.Length)

    let matches requested heading =
        String.Equals(normalizedName requested, normalizedName heading, StringComparison.Ordinal)
        || nearlyMatches requested heading

module DocumentChunking =
    let representationVersion = "pdf-section-aware-v2"

    type SectionBlocks =
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

    let chunkSections (options: ChunkOptions) (sections: SectionBlocks list) =
        sections
        |> List.collect (fun section ->
            section.blocks
            |> chunkBlocks options
            |> List.collect (fun chunk ->
                chunk
                |> DocumentSections.formatChunk section.heading
                |> enforceChunkBounds options))

    let chunkSectionedBlocks (options: ChunkOptions) (blocks: string list) =
        blocks |> splitIntoSections |> chunkSections options

    let passagesFromBlocks (options: ChunkOptions) (source: PassageSource) (blocks: string list) : PassageRef list =
        blocks
        |> chunkSectionedBlocks options
        |> List.mapi (fun index text ->
            { sourceId = source.id
              sourceDisplayName = source.displayName
              sourceLocation = source.location
              index = index
              text = text
              keywords = [] })

    let passagesFromSections
        (options: ChunkOptions)
        (source: PassageSource)
        (sections: SectionBlocks list)
        : PassageRef list =
        sections
        |> chunkSections options
        |> List.mapi (fun index text ->
            { sourceId = source.id
              sourceDisplayName = source.displayName
              sourceLocation = source.location
              index = index
              text = text
              keywords = [] })

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

module MarkdownDocuments =
    let private headingPattern = Regex(@"^(#{1,6})\s+(.+?)\s*#*\s*$", RegexOptions.Compiled)
    let private unorderedListPattern = Regex(@"^\s*[-+*]\s+", RegexOptions.Compiled)
    let private orderedListPattern = Regex(@"^\s*\d+[.)]\s+", RegexOptions.Compiled)

    let private normalizeLine (line: string) =
        line.Replace("\t", "    ").TrimEnd()

    let private stripInlineMarkup (text: string) =
        text
        |> fun value -> Regex.Replace(value, @"`([^`]+)`", "$1")
        |> fun value -> Regex.Replace(value, @"\*\*([^*]+)\*\*", "$1")
        |> fun value -> Regex.Replace(value, @"__([^_]+)__", "$1")
        |> fun value -> Regex.Replace(value, @"\*([^*]+)\*", "$1")
        |> fun value -> Regex.Replace(value, @"_([^_]+)_", "$1")
        |> fun value -> Regex.Replace(value, @"\[(.*?)\]\([^)]+\)", "$1")
        |> Text.normalizeWhitespace

    let private tryHeading (line: string) =
        let m = headingPattern.Match(line.Trim())

        if m.Success then
            let level = m.Groups[1].Value.Length
            let heading = stripInlineMarkup m.Groups[2].Value

            if String.IsNullOrWhiteSpace heading then
                None
            else
                Some(level, heading)
        else
            None

    let private isFence (line: string) =
        let trimmed = line.TrimStart()
        trimmed.StartsWith("```", StringComparison.Ordinal) || trimmed.StartsWith("~~~", StringComparison.Ordinal)

    let private isListLine (line: string) =
        unorderedListPattern.IsMatch(line) || orderedListPattern.IsMatch(line)

    let private isTableLine (line: string) =
        let trimmed = line.Trim()
        trimmed.StartsWith("|", StringComparison.Ordinal) && trimmed.EndsWith("|", StringComparison.Ordinal)

    let private addBlock lines blocks =
        match lines with
        | [] -> blocks
        | _ ->
            let block =
                lines
                |> List.rev
                |> String.concat "\n"
                |> Text.normalizeWhitespace

            if String.IsNullOrWhiteSpace block then
                blocks
            else
                block :: blocks

    let private headingPath (stack: (int * string) list) =
        let path =
            stack
            |> List.sortBy fst
            |> List.map snd
            |> String.concat " > "

        if String.IsNullOrWhiteSpace path then
            None
        else
            Some path

    let readSections path : Async<Result<DocumentChunking.SectionBlocks list, string>> =
        async {
            if not (File.Exists path) then
                return Error $"Markdown file not found: {path}"
            else
                try
                    let! text = File.ReadAllTextAsync(path) |> Async.AwaitTask

                    let lines =
                        text.Replace("\r\n", "\n").Replace('\r', '\n').Split('\n')
                        |> Array.toList
                        |> List.map normalizeLine

                    let rec skipFrontMatter (remaining: string list) =
                        match remaining with
                        | first :: rest when first.Trim() = "---" ->
                            rest
                            |> List.skipWhile (fun line -> line.Trim() <> "---")
                            |> function
                                | _closing :: tail -> tail
                                | [] -> remaining
                        | _ -> remaining

                    let addSection
                        (stack: (int * string) list)
                        blockLines
                        (sections: DocumentChunking.SectionBlocks list)
                        =
                        let blocks = addBlock blockLines []

                        if List.isEmpty blocks then
                            sections
                        else
                            { heading = headingPath stack
                              blocks = List.rev blocks }
                            :: sections

                    let rec loop
                        (remaining: string list)
                        (stack: (int * string) list)
                        blockLines
                        (sections: DocumentChunking.SectionBlocks list)
                        inFence
                        =
                        match remaining with
                        | [] -> addSection stack blockLines sections |> List.rev
                        | line :: rest when inFence ->
                            let blockLines = line :: blockLines
                            loop rest stack blockLines sections (not (isFence line))
                        | line :: rest when isFence line ->
                            loop rest stack (line :: blockLines) sections true
                        | line :: rest ->
                            match tryHeading line with
                            | Some(level, heading) ->
                                let sections = addSection stack blockLines sections

                                let stack =
                                    stack
                                    |> List.filter (fun (existingLevel, _) -> existingLevel < level)
                                    |> fun parents -> (level, heading) :: parents

                                loop rest stack [] sections false
                            | None when String.IsNullOrWhiteSpace line ->
                                let sections = addSection stack blockLines sections
                                loop rest stack [] sections false
                            | None when isListLine line || isTableLine line ->
                                let rec collectRelated collected remaining =
                                    match remaining with
                                    | next :: tail when
                                        String.IsNullOrWhiteSpace next
                                        || isListLine next
                                        || isTableLine next
                                        || next.StartsWith("  ", StringComparison.Ordinal)
                                        ->
                                        collectRelated (next :: collected) tail
                                    | _ -> List.rev collected, remaining

                                let related, remaining = collectRelated [ line ] rest
                                let sections = addSection stack (List.rev related) sections
                                loop remaining stack [] sections false
                            | None -> loop rest stack (line :: blockLines) sections false

                    let sections = loop (skipFrontMatter lines) [] [] [] false

                    return Ok sections
                with ex ->
                    return Error $"Unable to read Markdown file '{path}': {ex.Message}"
        }

    let readBlocks path =
        async {
            let! result = readSections path

            return
                result
                |> Result.map (fun sections ->
                    sections
                    |> List.collect (fun section ->
                        match section.heading with
                        | Some heading -> heading :: section.blocks
                        | None -> section.blocks))
        }

    let readText path =
        async {
            let! result = readBlocks path
            return result |> Result.map (String.concat "\n\n")
        }

    let readPassages chunkOptions source path =
        async {
            let! result = readSections path
            return result |> Result.map (DocumentChunking.passagesFromSections chunkOptions source)
        }
