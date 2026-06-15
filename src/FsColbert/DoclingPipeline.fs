namespace FsColbert

open System
open System.Text.RegularExpressions
open System.Threading

type DoclingConversionOptions =
    { ocrCellOverlapThreshold: float
      includeEmptyTextClusters: bool
      maxFigureClassifications: int }

module DoclingConversionOptions =
    let defaults =
        { ocrCellOverlapThreshold = 0.3
          includeEmptyTextClusters = false
          maxFigureClassifications = 5 }

    let sanitize options =
        { options with
            ocrCellOverlapThreshold = min 1.0 (max 0.0 options.ocrCellOverlapThreshold)
            maxFigureClassifications = max 0 options.maxFigureClassifications }

module DoclingReadingOrder =
    let private groupOrder label =
        match label with
        | PageHeader -> 0
        | PageFooter -> 2
        | _ -> 1

    let private compareFloat left right = compare left right

    let compareClusters (leftPageNo, left: DoclingLayoutCluster) (rightPageNo, right: DoclingLayoutCluster) =
        let pageCompare = compare leftPageNo rightPageNo

        if pageCompare <> 0 then
            pageCompare
        else
            let groupCompare = compare (groupOrder left.label) (groupOrder right.label)

            if groupCompare <> 0 then
                groupCompare
            elif DoclingGeometry.horizontallyOverlaps left.bbox right.bbox then
                let topCompare = compareFloat left.bbox.t right.bbox.t

                if topCompare <> 0 then
                    topCompare
                else
                    compareFloat left.bbox.l right.bbox.l
            else
                let leftCompare = compareFloat left.bbox.l right.bbox.l

                if leftCompare <> 0 then
                    leftCompare
                else
                    compareFloat left.bbox.t right.bbox.t

    let sortPageClusters (clusters: (int * DoclingLayoutCluster) list) =
        clusters |> List.sortWith compareClusters

module DoclingAssembly =
    type private PageClusters =
        { page: DoclingPageInput
          clusters: DoclingLayoutCluster list }

    type private AssembledElement =
        | TextElement of int * DoclingLayoutCluster * string
        | TableElement of int * DoclingLayoutCluster * DoclingTableData
        | PictureElement of int * DoclingLayoutCluster

    let private normalizeCell pageHeight (cell: DoclingOcrCell) : DoclingOcrCell =
        { cell with
            bbox = DoclingGeometry.toTopLeft pageHeight cell.bbox }

    let private normalizeCluster pageHeight (cluster: DoclingLayoutCluster) : DoclingLayoutCluster =
        { cluster with
            bbox = DoclingGeometry.toTopLeft pageHeight cluster.bbox
            cells = cluster.cells |> List.map (normalizeCell pageHeight) }

    let private shouldAssignCell threshold (cluster: DoclingLayoutCluster) (cell: DoclingOcrCell) =
        let cellBox = cell.bbox
        let clusterBox = cluster.bbox
        let centerX, centerY = DoclingGeometry.center cellBox

        DoclingGeometry.intersectionOverSelf cellBox clusterBox >= threshold
        || DoclingGeometry.containsPoint centerX centerY clusterBox

    let private attachCells threshold (page: DoclingPageInput) (prediction: DoclingLayoutPrediction) =
        let pageHeight = float page.image.height
        let cells = page.ocrCells |> List.map (normalizeCell pageHeight)

        let clusters =
            prediction.clusters
            |> List.map (normalizeCluster pageHeight)
            |> List.mapi (fun index cluster ->
                let assigned =
                    cells
                    |> List.filter (shouldAssignCell threshold cluster)
                    |> List.sortBy (fun cell -> cell.bbox.t, cell.bbox.l)

                { cluster with
                    id = index
                    cells = assigned })

        { page = page; clusters = clusters }

    let private sanitizeText values =
        values |> String.concat " " |> Text.normalizeWhitespace

    let private cellHeight (cell: DoclingOcrCell) = max 1.0 (cell.bbox.b - cell.bbox.t)

    let private sameLine (line: DoclingOcrCell list) (cell: DoclingOcrCell) =
        let lineTop = line |> List.averageBy (fun item -> item.bbox.t)
        let lineHeight = line |> List.averageBy cellHeight
        abs (cell.bbox.t - lineTop) <= max 3.0 (lineHeight * 0.6)

    let private orderCellsForReading (cells: DoclingOcrCell list) =
        let rec addCell (cell: DoclingOcrCell) (lines: DoclingOcrCell list list) =
            match lines with
            | [] -> [ [ cell ] ]
            | line :: rest when sameLine line cell -> (cell :: line) :: rest
            | line :: rest -> line :: addCell cell rest

        cells
        |> List.sortBy (fun cell -> cell.bbox.t)
        |> List.fold (fun lines cell -> addCell cell lines) []
        |> List.collect (List.sortBy (fun cell -> cell.bbox.l))

    let private clusterText (cluster: DoclingLayoutCluster) =
        cluster.cells |> orderCellsForReading |> List.map _.text |> sanitizeText

    let private toTextLabel label =
        if DoclingLabels.isTextLike label then label else Text

    let private toTableLabel label =
        match label with
        | DocumentIndex -> DocumentIndex
        | _ -> Table

    let private toPictureLabel label =
        match label with
        | Chart -> Chart
        | _ -> Picture

    let private tableDataFromCluster cluster =
        let cells =
            cluster.cells
            |> orderCellsForReading
            |> List.mapi (fun index cell ->
                { text = Text.normalizeWhitespace cell.text
                  bbox = Some cell.bbox
                  startRowOffsetIndex = index
                  endRowOffsetIndex = index + 1
                  startColOffsetIndex = 0
                  endColOffsetIndex = 1
                  rowHeader = false
                  columnHeader = false
                  rowSection = false })

        { numRows = cells.Length
          numCols = if List.isEmpty cells then 0 else 1
          tableCells = cells }

    let private elementFromCluster includeEmptyText pageNo cluster =
        if DoclingLabels.isTableLike cluster.label then
            Some(TableElement(pageNo, cluster, tableDataFromCluster cluster))
        elif DoclingLabels.isPictureLike cluster.label then
            Some(PictureElement(pageNo, cluster))
        else
            let text = clusterText cluster

            if String.IsNullOrWhiteSpace text && not includeEmptyText then
                None
            else
                Some(TextElement(pageNo, cluster, text))

    let private provenance pageNo textLength bbox =
        { pageNo = pageNo
          bbox = bbox
          charSpan = Some(0, max 0 textLength) }

    let private bodyParent layer =
        match layer with
        | Furniture -> "#/furniture"
        | _ -> "#/body"

    let private appendChild layer cref bodyChildren furnitureChildren =
        match layer with
        | Furniture -> bodyChildren, cref :: furnitureChildren
        | _ -> cref :: bodyChildren, furnitureChildren

    let private classifyPicture
        (options: DoclingConversionOptions)
        (pageByNumber: Map<int, DoclingPageInput>)
        (figureClassifier: IDoclingFigureClassifier option)
        pageNo
        (cluster: DoclingLayoutCluster)
        =
        async {
            match figureClassifier, Map.tryFind pageNo pageByNumber with
            | None, _ -> return Ok []
            | Some _, None -> return Error $"Unable to classify picture on missing page {pageNo}."
            | Some classifier, Some page ->
                let crop = DoclingRgbImage.crop cluster.bbox page.image
                let! result = classifier.ClassifyAsync crop

                return
                    result
                    |> Result.map (
                        List.sortByDescending _.confidence
                        >> List.truncate options.maxFigureClassifications
                    )
        }

    let fromPredictionsWithOptions
        (options: DoclingConversionOptions)
        (documentName: string)
        (originFileName: string option)
        (pages: DoclingPageInput list)
        (predictions: DoclingLayoutPrediction list)
        (figureClassifier: IDoclingFigureClassifier option)
        =
        async {
            let options = DoclingConversionOptions.sanitize options

            if List.isEmpty pages then
                return Error "At least one page is required."
            else
                let predictionByPage =
                    predictions
                    |> List.map (fun prediction -> prediction.pageNo, prediction)
                    |> Map.ofList

                let missingPages =
                    pages
                    |> List.choose (fun page ->
                        if Map.containsKey page.pageNo predictionByPage then
                            None
                        else
                            Some page.pageNo)

                if not (List.isEmpty missingPages) then
                    let missingPageText = String.Join(", ", missingPages)
                    return Error $"Missing layout predictions for pages: {missingPageText}."
                else
                    let pageClusters =
                        pages
                        |> List.map (fun page ->
                            attachCells options.ocrCellOverlapThreshold page predictionByPage[page.pageNo])

                    let sortedElements =
                        pageClusters
                        |> List.collect (fun pageClusters ->
                            pageClusters.clusters
                            |> List.choose (
                                elementFromCluster options.includeEmptyTextClusters pageClusters.page.pageNo
                            ))
                        |> List.sortWith (fun left right ->
                            let leftCluster =
                                match left with
                                | TextElement(pageNo, cluster, _) -> pageNo, cluster
                                | TableElement(pageNo, cluster, _) -> pageNo, cluster
                                | PictureElement(pageNo, cluster) -> pageNo, cluster

                            let rightCluster =
                                match right with
                                | TextElement(pageNo, cluster, _) -> pageNo, cluster
                                | TableElement(pageNo, cluster, _) -> pageNo, cluster
                                | PictureElement(pageNo, cluster) -> pageNo, cluster

                            DoclingReadingOrder.compareClusters leftCluster rightCluster)

                    let pageByNumber = pages |> List.map (fun page -> page.pageNo, page) |> Map.ofList
                    let textItems = ResizeArray<DoclingTextItem>()
                    let tableItems = ResizeArray<DoclingTableItem>()
                    let pictureItems = ResizeArray<DoclingPictureItem>()
                    let sourceId = Some documentName
                    let sourceDisplayName = originFileName |> Option.orElse (Some documentName)
                    let mutable bodyChildren = []
                    let mutable furnitureChildren = []
                    let mutable error = None

                    for element in sortedElements do
                        match error, element with
                        | Some _, _ -> ()
                        | None, TextElement(pageNo, cluster, text) ->
                            let layer = DoclingLabels.contentLayer cluster.label
                            let label = toTextLabel cluster.label
                            let selfRef = $"#/texts/{textItems.Count}"

                            let item =
                                { selfRef = selfRef
                                  parent = bodyParent layer
                                  label = label
                                  text = text
                                  orig = text
                                  contentLayer = layer
                                  prov = [ provenance pageNo text.Length cluster.bbox ]
                                  keywords = DoclingKeywords.weakDefaults label [] []
                                  sourceId = sourceId
                                  sourceDisplayName = sourceDisplayName }

                            textItems.Add item
                            let body, furniture = appendChild layer selfRef bodyChildren furnitureChildren
                            bodyChildren <- body
                            furnitureChildren <- furniture
                        | None, TableElement(pageNo, cluster, data) ->
                            let label = toTableLabel cluster.label
                            let selfRef = $"#/tables/{tableItems.Count}"

                            let item =
                                { selfRef = selfRef
                                  parent = "#/body"
                                  label = label
                                  contentLayer = Body
                                  prov = [ provenance pageNo 0 cluster.bbox ]
                                  data = data
                                  keywords = DoclingKeywords.weakDefaults label [] []
                                  sourceId = sourceId
                                  sourceDisplayName = sourceDisplayName }

                            tableItems.Add item
                            bodyChildren <- selfRef :: bodyChildren
                        | None, PictureElement(pageNo, cluster) ->
                            let! classified = classifyPicture options pageByNumber figureClassifier pageNo cluster

                            match classified with
                            | Error err -> error <- Some err
                            | Ok classifications ->
                                let label = toPictureLabel cluster.label
                                let selfRef = $"#/pictures/{pictureItems.Count}"

                                let item =
                                    { selfRef = selfRef
                                      parent = "#/body"
                                      label = label
                                      contentLayer = Body
                                      prov = [ provenance pageNo 0 cluster.bbox ]
                                      classifications = classifications
                                      keywords = DoclingKeywords.weakDefaults label classifications []
                                      sourceId = sourceId
                                      sourceDisplayName = sourceDisplayName }

                                pictureItems.Add item
                                bodyChildren <- selfRef :: bodyChildren

                    match error with
                    | Some err -> return Error err
                    | None ->
                        let pages =
                            pages
                            |> List.map (fun page ->
                                page.pageNo,
                                { pageNo = page.pageNo
                                  size =
                                    { width = float page.image.width
                                      height = float page.image.height } })
                            |> Map.ofList

                        let document =
                            { name = documentName
                              originFileName = originFileName
                              originMimeType = Some "application/pdf"
                              pages = pages
                              texts = textItems |> Seq.toList
                              tables = tableItems |> Seq.toList
                              pictures = pictureItems |> Seq.toList
                              bodyChildren = List.rev bodyChildren
                              furnitureChildren = List.rev furnitureChildren }

                        return Ok document
        }

    let fromPredictions documentName originFileName pages predictions figureClassifier =
        fromPredictionsWithOptions
            DoclingConversionOptions.defaults
            documentName
            originFileName
            pages
            predictions
            figureClassifier

module DoclingStandardHybrid =
    let private combineResults results =
        results
        |> List.fold
            (fun state result ->
                match state, result with
                | Error err, _ -> Error err
                | _, Error err -> Error err
                | Ok values, Ok value -> Ok(value :: values))
            (Ok [])
        |> Result.map List.rev

    let private recognizePages (ocr: IDoclingOcrProvider) pages =
        let rec loop remaining acc =
            async {
                match remaining with
                | [] -> return Ok(List.rev acc)
                | page :: rest ->
                    let! result = ocr.RecognizeAsync page

                    match result with
                    | Error err -> return Error err
                    | Ok cells ->
                        let input =
                            { pageNo = page.pageNo
                              image = page.image
                              ocrCells = cells }

                        return! loop rest (input :: acc)
            }

        loop pages []

    let rec convertPagesWithOptions
        (options: DoclingConversionOptions)
        (documentName: string)
        (originFileName: string option)
        (layoutPredictor: IDoclingLayoutPredictor)
        (figureClassifier: IDoclingFigureClassifier option)
        (pages: DoclingPageInput list)
        =
        convertPagesWithOptionsWithCancellation
            options
            documentName
            originFileName
            layoutPredictor
            figureClassifier
            pages
            CancellationToken.None

    and convertPagesWithOptionsWithCancellation
        (options: DoclingConversionOptions)
        (documentName: string)
        (originFileName: string option)
        (layoutPredictor: IDoclingLayoutPredictor)
        (figureClassifier: IDoclingFigureClassifier option)
        (pages: DoclingPageInput list)
        (cancellationToken: CancellationToken)
        =
        async {
            cancellationToken.ThrowIfCancellationRequested()

            let! predictions =
                match layoutPredictor with
                | :? ICancelableDoclingLayoutPredictor as cancelable ->
                    cancelable.PredictLayoutAsync(pages, cancellationToken)
                | _ -> layoutPredictor.PredictLayoutAsync pages

            cancellationToken.ThrowIfCancellationRequested()

            match predictions with
            | Error err -> return Error err
            | Ok predictions ->
                cancellationToken.ThrowIfCancellationRequested()

                return!
                    DoclingAssembly.fromPredictionsWithOptions
                        options
                        documentName
                        originFileName
                        pages
                        predictions
                        figureClassifier
        }

    let convertPages documentName originFileName layoutPredictor figureClassifier pages =
        convertPagesWithOptions
            DoclingConversionOptions.defaults
            documentName
            originFileName
            layoutPredictor
            figureClassifier
            pages

    let convertPdfWithProviders
        (options: DoclingConversionOptions)
        (path: string)
        (rasterizer: IDoclingPageRasterizer)
        (ocr: IDoclingOcrProvider)
        (layoutPredictor: IDoclingLayoutPredictor)
        (figureClassifier: IDoclingFigureClassifier option)
        =
        async {
            let! rasterized = rasterizer.RasterizeAsync path

            match rasterized with
            | Error err -> return Error err
            | Ok pages ->
                let! pageInputs = recognizePages ocr pages

                match pageInputs with
                | Error err -> return Error err
                | Ok pageInputs ->
                    let documentName = IO.Path.GetFileNameWithoutExtension path

                    return!
                        convertPagesWithOptions
                            options
                            documentName
                            (Some(IO.Path.GetFileName path))
                            layoutPredictor
                            figureClassifier
                            pageInputs
        }

type DoclingPassageBlock =
    { text: string
      sectionPath: string list
      contentRole: PassageContentRole
      pageNumbers: int list
      layoutLabels: string list
      captions: string list
      keywords: string list }

module DoclingPassages =
    type private SectionState =
        { title: string option
          sections: string list }

    let private emptySectionState = { title = None; sections = [] }

    let private numberedHeadingPattern =
        Regex(@"^\s*(\d+(?:\.\d+)*)\b", RegexOptions.Compiled)

    let private normalizeSectionText text =
        text
        |> Text.normalizeWhitespace
        |> Option.ofObj
        |> Option.filter (String.IsNullOrWhiteSpace >> not)

    let private currentSectionPath state =
        [ yield! state.title |> Option.toList; yield! state.sections ]

    let private numberedHeadingParts text =
        let m = numberedHeadingPattern.Match(defaultArg (Option.ofObj text) "")

        if m.Success then
            m.Groups[1].Value.Split('.')
            |> Array.toList
            |> List.map Int32.TryParse
            |> List.fold
                (fun acc parsed ->
                    match acc, parsed with
                    | Some values, (true, value) -> Some(values @ [ value ])
                    | _ -> None)
                (Some [])
            |> Option.filter (List.isEmpty >> not)
        else
            None

    let private matchingNumberedParents headingParts sections =
        let parentParts = headingParts |> List.truncate (max 0 (headingParts.Length - 1))

        if List.isEmpty parentParts then
            []
        else
            let currentParents = sections |> List.truncate parentParts.Length

            let currentParentParts = currentParents |> List.choose numberedHeadingParts

            if
                currentParentParts.Length = parentParts.Length
                && currentParentParts = (parentParts
                                         |> List.mapi (fun index _ -> parentParts |> List.truncate (index + 1)))
            then
                currentParents
            else
                []

    let private updateTitle state text =
        match normalizeSectionText text with
        | Some title -> { title = Some title; sections = [] }
        | None -> state

    let private updateSection state text =
        match normalizeSectionText text with
        | None -> state
        | Some heading ->
            match numberedHeadingParts heading with
            | Some parts ->
                { state with
                    sections = matchingNumberedParents parts state.sections @ [ heading ] }
            | None -> { state with sections = [ heading ] }

    let private sectionKeywords sectionPath = sectionPath |> DoclingKeywords.sanitize

    let private sortedPageNumbers (prov: DoclingProvenance list) =
        prov
        |> List.map _.pageNo
        |> List.filter (fun page -> page > 0)
        |> List.distinct
        |> List.sort

    let private roleKeywords role =
        match role with
        | PassageContentRole.Unknown -> []
        | _ -> [ PassageContentRole.displayName role; PassageContentRole.storageValue role ]

    let private layoutLabelValue label =
        match label with
        | Text
        | Paragraph
        | PageHeader
        | PageFooter -> []
        | _ -> [ DoclingLabels.toJsonValue label ]

    let private layoutLabelKeywords label =
        layoutLabelValue label |> List.map (fun value -> value.Replace("_", " "))

    let private captionText label text =
        match label with
        | Caption ->
            text
            |> Text.normalizeWhitespace
            |> Option.ofObj
            |> Option.filter (String.IsNullOrWhiteSpace >> not)
            |> Option.toList
        | _ -> []

    let private roleForText sectionPath (item: DoclingTextItem) =
        match item.label with
        | Title -> PassageContentRole.FrontMatter
        | _ ->
            match sectionPath with
            | _ :: [] -> PassageContentRole.FrontMatter
            | _ -> DocumentContentRoles.infer sectionPath item.text

    let private roleForStructuredBlock sectionPath text =
        match sectionPath with
        | _ :: [] -> PassageContentRole.FrontMatter
        | _ -> DocumentContentRoles.infer sectionPath text

    let private tableToText (table: DoclingTableItem) =
        if table.data.numRows = 0 || List.isEmpty table.data.tableCells then
            "[Table]"
        else
            table.data.tableCells
            |> List.groupBy _.startRowOffsetIndex
            |> List.sortBy fst
            |> List.map (fun (_, cells) ->
                cells
                |> List.sortBy _.startColOffsetIndex
                |> List.map _.text
                |> String.concat " | "
                |> Text.normalizeWhitespace)
            |> List.filter (String.IsNullOrWhiteSpace >> not)
            |> String.concat "\n"

    let private pictureToText (picture: DoclingPictureItem) =
        if List.isEmpty picture.classifications then
            "[Picture]"
        else
            let classes =
                picture.classifications
                |> List.map (fun cls -> sprintf "%s (%.3f)" cls.className cls.confidence)
                |> String.concat ", "

            $"[Picture: {classes}]"

    let private normalizeBlock block =
        { text = Text.normalizeWhitespace block.text
          sectionPath =
            block.sectionPath
            |> List.choose normalizeSectionText
            |> DoclingKeywords.sanitize
          contentRole = block.contentRole
          pageNumbers =
            block.pageNumbers
            |> List.filter (fun page -> page > 0)
            |> List.distinct
            |> List.sort
          layoutLabels = block.layoutLabels |> DoclingKeywords.sanitize
          captions = block.captions |> DoclingKeywords.sanitizeWithLimit 8
          keywords = DoclingKeywords.sanitize block.keywords }

    let private textBlock sectionPath (item: DoclingTextItem) =
        let role = roleForText sectionPath item

        { text = item.text
          sectionPath = sectionPath
          contentRole = role
          pageNumbers = sortedPageNumbers item.prov
          layoutLabels = layoutLabelValue item.label
          captions = captionText item.label item.text
          keywords =
            DoclingKeywords.weakDefaults item.label [] item.keywords
            @ layoutLabelKeywords item.label
            @ captionText item.label item.text
            @ sectionKeywords sectionPath
            @ roleKeywords role }
        |> normalizeBlock

    let private tableBlock sectionPath (item: DoclingTableItem) =
        let text = tableToText item
        let role = roleForStructuredBlock sectionPath text

        { text = text
          sectionPath = sectionPath
          contentRole = role
          pageNumbers = sortedPageNumbers item.prov
          layoutLabels = layoutLabelValue item.label
          captions = []
          keywords =
            DoclingKeywords.weakDefaults item.label [] item.keywords
            @ layoutLabelKeywords item.label
            @ sectionKeywords sectionPath
            @ roleKeywords role }
        |> normalizeBlock

    let private pictureBlock sectionPath (item: DoclingPictureItem) =
        let text = pictureToText item
        let role = roleForStructuredBlock sectionPath text

        { text = text
          sectionPath = sectionPath
          contentRole = role
          pageNumbers = sortedPageNumbers item.prov
          layoutLabels = layoutLabelValue item.label
          captions = []
          keywords =
            DoclingKeywords.weakDefaults item.label item.classifications item.keywords
            @ layoutLabelKeywords item.label
            @ sectionKeywords sectionPath
            @ roleKeywords role }
        |> normalizeBlock

    let toBlocksWithKeywords document =
        let texts =
            document.texts |> List.map (fun item -> item.selfRef, item) |> Map.ofList

        let tables =
            document.tables |> List.map (fun item -> item.selfRef, item) |> Map.ofList

        let pictures =
            document.pictures |> List.map (fun item -> item.selfRef, item) |> Map.ofList

        let step (state, blocks) ref =
            match Map.tryFind ref texts with
            | Some item ->
                let state =
                    match item.label with
                    | Title -> updateTitle state item.text
                    | SectionHeader -> updateSection state item.text
                    | _ -> state

                let sectionPath = currentSectionPath state
                let block = textBlock sectionPath item
                state, block :: blocks
            | None ->
                let sectionPath = currentSectionPath state

                match Map.tryFind ref tables with
                | Some item -> state, tableBlock sectionPath item :: blocks
                | None ->
                    match Map.tryFind ref pictures with
                    | Some item -> state, pictureBlock sectionPath item :: blocks
                    | None -> state, blocks

        document.bodyChildren
        |> List.fold step (emptySectionState, [])
        |> snd
        |> List.rev
        |> List.filter (fun block -> not (String.IsNullOrWhiteSpace block.text))

    let toBlocks document =
        document |> toBlocksWithKeywords |> List.map _.text

    let private flushChunk
        currentTexts
        currentSectionPath
        currentRole
        currentPageNumbers
        currentLayoutLabels
        currentCaptions
        currentKeywords
        chunks
        =
        match currentTexts with
        | [] -> chunks
        | _ ->
            let text =
                currentTexts |> List.rev |> String.concat "\n\n" |> Text.normalizeWhitespace

            if String.IsNullOrWhiteSpace text then
                chunks
            else
                { text = text
                  sectionPath = currentSectionPath
                  contentRole = currentRole
                  pageNumbers = currentPageNumbers |> List.distinct |> List.sort
                  layoutLabels = DoclingKeywords.sanitize currentLayoutLabels
                  captions = DoclingKeywords.sanitizeWithLimit 8 currentCaptions
                  keywords = DoclingKeywords.sanitize currentKeywords }
                :: chunks

    let private splitLongBlock options block =
        block.text
        |> Text.chunkText options
        |> List.map (fun (_, text) ->
            { text = text
              sectionPath = block.sectionPath
              contentRole = block.contentRole
              pageNumbers = block.pageNumbers
              layoutLabels = block.layoutLabels
              captions = block.captions
              keywords = block.keywords })

    let private chunkBlocksWithKeywords options blocks =
        let maxChars = max 1 options.maxChars

        let rec loop
            remaining
            currentTexts
            currentLen
            currentSectionPath
            currentRole
            currentPageNumbers
            currentLayoutLabels
            currentCaptions
            currentKeywords
            chunks
            =
            match remaining with
            | [] ->
                flushChunk
                    currentTexts
                    currentSectionPath
                    currentRole
                    currentPageNumbers
                    currentLayoutLabels
                    currentCaptions
                    currentKeywords
                    chunks
                |> List.rev
            | block :: rest ->
                let block = normalizeBlock block

                if String.IsNullOrWhiteSpace block.text then
                    loop
                        rest
                        currentTexts
                        currentLen
                        currentSectionPath
                        currentRole
                        currentPageNumbers
                        currentLayoutLabels
                        currentCaptions
                        currentKeywords
                        chunks
                elif
                    currentLen > 0
                    && (block.sectionPath <> currentSectionPath || block.contentRole <> currentRole)
                then
                    let chunks =
                        flushChunk
                            currentTexts
                            currentSectionPath
                            currentRole
                            currentPageNumbers
                            currentLayoutLabels
                            currentCaptions
                            currentKeywords
                            chunks

                    loop remaining [] 0 [] PassageContentRole.Unknown [] [] [] [] chunks
                elif block.text.Length > maxChars then
                    let chunks =
                        flushChunk
                            currentTexts
                            currentSectionPath
                            currentRole
                            currentPageNumbers
                            currentLayoutLabels
                            currentCaptions
                            currentKeywords
                            chunks

                    let splitChunks = splitLongBlock options block
                    loop rest [] 0 [] PassageContentRole.Unknown [] [] [] [] (List.rev splitChunks @ chunks)
                elif currentLen = 0 then
                    loop
                        rest
                        [ block.text ]
                        block.text.Length
                        block.sectionPath
                        block.contentRole
                        block.pageNumbers
                        block.layoutLabels
                        block.captions
                        block.keywords
                        chunks
                elif currentLen + block.text.Length + 2 > maxChars then
                    let chunks =
                        flushChunk
                            currentTexts
                            currentSectionPath
                            currentRole
                            currentPageNumbers
                            currentLayoutLabels
                            currentCaptions
                            currentKeywords
                            chunks

                    loop remaining [] 0 [] PassageContentRole.Unknown [] [] [] [] chunks
                else
                    loop
                        rest
                        (block.text :: currentTexts)
                        (currentLen + block.text.Length + 2)
                        currentSectionPath
                        currentRole
                        (currentPageNumbers @ block.pageNumbers)
                        (currentLayoutLabels @ block.layoutLabels)
                        (currentCaptions @ block.captions)
                        (currentKeywords @ block.keywords)
                        chunks

        loop blocks [] 0 [] PassageContentRole.Unknown [] [] [] [] []

    let toPassages (chunkOptions: ChunkOptions) (source: PassageSource) document =
        document
        |> toBlocksWithKeywords
        |> chunkBlocksWithKeywords chunkOptions
        |> List.mapi (fun index block ->
            { sourceId = source.id
              sourceDisplayName = source.displayName
              sourceLocation = source.location
              index = index
              text = block.text
              sectionPath = block.sectionPath
              contentRole = block.contentRole
              pageNumbers = block.pageNumbers
              layoutLabels = block.layoutLabels
              captions = block.captions
              keywords = block.keywords })
