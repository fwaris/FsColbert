namespace FsColbert

open System
open System.Threading
open System.Text.Json
open System.Text.Json.Nodes

type DoclingCoordinateOrigin =
    | TopLeft
    | BottomLeft

type DoclingContentLayer =
    | Body
    | Furniture
    | Background
    | Invisible
    | Notes

type DoclingLabel =
    | Caption
    | CheckboxSelected
    | CheckboxUnselected
    | Code
    | DocumentIndex
    | Footnote
    | Formula
    | ListItem
    | PageFooter
    | PageHeader
    | Paragraph
    | Picture
    | Chart
    | SectionHeader
    | Table
    | Text
    | Title
    | Form
    | KeyValueRegion
    | Other of string

type DoclingBoundingBox =
    { l: float
      t: float
      r: float
      b: float
      coordOrigin: DoclingCoordinateOrigin }

type DoclingSize = { width: float; height: float }

type DoclingProvenance =
    { pageNo: int
      bbox: DoclingBoundingBox
      charSpan: (int * int) option }

type DoclingOcrCell =
    { text: string
      bbox: DoclingBoundingBox
      confidence: float option }

type DoclingFigureClass =
    { className: string
      confidence: float32 }

type DoclingTableCell =
    { text: string
      bbox: DoclingBoundingBox option
      startRowOffsetIndex: int
      endRowOffsetIndex: int
      startColOffsetIndex: int
      endColOffsetIndex: int
      rowHeader: bool
      columnHeader: bool
      rowSection: bool }

type DoclingTableData =
    { numRows: int
      numCols: int
      tableCells: DoclingTableCell list }

type DoclingTextItem =
    { selfRef: string
      parent: string
      label: DoclingLabel
      text: string
      orig: string
      contentLayer: DoclingContentLayer
      prov: DoclingProvenance list
      keywords: string list
      sourceId: string option
      sourceDisplayName: string option }

type DoclingTableItem =
    { selfRef: string
      parent: string
      label: DoclingLabel
      contentLayer: DoclingContentLayer
      prov: DoclingProvenance list
      data: DoclingTableData
      keywords: string list
      sourceId: string option
      sourceDisplayName: string option }

type DoclingPictureItem =
    { selfRef: string
      parent: string
      label: DoclingLabel
      contentLayer: DoclingContentLayer
      prov: DoclingProvenance list
      classifications: DoclingFigureClass list
      keywords: string list
      sourceId: string option
      sourceDisplayName: string option }

type DoclingPageItem = { pageNo: int; size: DoclingSize }

type DoclingDocument =
    { name: string
      originFileName: string option
      originMimeType: string option
      pages: Map<int, DoclingPageItem>
      texts: DoclingTextItem list
      tables: DoclingTableItem list
      pictures: DoclingPictureItem list
      bodyChildren: string list
      furnitureChildren: string list }

type DoclingRgbImage =
    { width: int
      height: int
      pixels: byte array }

type DoclingRasterPage = { pageNo: int; image: DoclingRgbImage }

type DoclingPageInput =
    { pageNo: int
      image: DoclingRgbImage
      ocrCells: DoclingOcrCell list }

type DoclingNativePageText =
    { pageNo: int
      size: DoclingSize
      cells: DoclingOcrCell list }

type DoclingLayoutCluster =
    { id: int
      label: DoclingLabel
      confidence: float32
      bbox: DoclingBoundingBox
      cells: DoclingOcrCell list }

type DoclingLayoutPrediction =
    { pageNo: int
      clusters: DoclingLayoutCluster list }

type DoclingOnnxModelFiles =
    { modelPath: string
      configPath: string
      preprocessorConfigPath: string }

type IDoclingPageRasterizer =
    abstract RasterizeAsync: path: string -> Async<Result<DoclingRasterPage list, string>>

type ICancelableDoclingPageRasterizer =
    abstract RasterizeAsync:
        path: string * cancellationToken: CancellationToken -> Async<Result<DoclingRasterPage list, string>>

type IDoclingOcrProvider =
    abstract RecognizeAsync: page: DoclingRasterPage -> Async<Result<DoclingOcrCell list, string>>

type ICancelableDoclingOcrProvider =
    abstract RecognizeAsync:
        page: DoclingRasterPage * cancellationToken: CancellationToken -> Async<Result<DoclingOcrCell list, string>>

type IDoclingLayoutPredictor =
    abstract PredictLayoutAsync: pages: DoclingPageInput list -> Async<Result<DoclingLayoutPrediction list, string>>

type ICancelableDoclingLayoutPredictor =
    abstract PredictLayoutAsync:
        pages: DoclingPageInput list * cancellationToken: CancellationToken ->
            Async<Result<DoclingLayoutPrediction list, string>>

type IDoclingFigureClassifier =
    abstract ClassifyAsync: image: DoclingRgbImage -> Async<Result<DoclingFigureClass list, string>>

type ICancelableDoclingFigureClassifier =
    abstract ClassifyAsync:
        image: DoclingRgbImage * cancellationToken: CancellationToken -> Async<Result<DoclingFigureClass list, string>>

module DoclingLabels =
    let toJsonValue label =
        match label with
        | Caption -> "caption"
        | CheckboxSelected -> "checkbox_selected"
        | CheckboxUnselected -> "checkbox_unselected"
        | Code -> "code"
        | DocumentIndex -> "document_index"
        | Footnote -> "footnote"
        | Formula -> "formula"
        | ListItem -> "list_item"
        | PageFooter -> "page_footer"
        | PageHeader -> "page_header"
        | Paragraph -> "paragraph"
        | Picture -> "picture"
        | Chart -> "chart"
        | SectionHeader -> "section_header"
        | Table -> "table"
        | Text -> "text"
        | Title -> "title"
        | Form -> "form"
        | KeyValueRegion -> "key_value_region"
        | Other value -> value

    let ofJsonValue value =
        match value with
        | "caption" -> Caption
        | "checkbox_selected" -> CheckboxSelected
        | "checkbox_unselected" -> CheckboxUnselected
        | "code" -> Code
        | "document_index" -> DocumentIndex
        | "footnote" -> Footnote
        | "formula" -> Formula
        | "list_item" -> ListItem
        | "page_footer" -> PageFooter
        | "page_header" -> PageHeader
        | "paragraph" -> Paragraph
        | "picture" -> Picture
        | "chart" -> Chart
        | "section_header" -> SectionHeader
        | "table" -> Table
        | "text" -> Text
        | "title" -> Title
        | "form" -> Form
        | "key_value_region" -> KeyValueRegion
        | other -> Other other

    let isTextLike label =
        match label with
        | Caption
        | CheckboxSelected
        | CheckboxUnselected
        | Code
        | Footnote
        | Formula
        | ListItem
        | PageFooter
        | PageHeader
        | Paragraph
        | SectionHeader
        | Text
        | Title -> true
        | _ -> false

    let isTableLike label =
        match label with
        | Table
        | DocumentIndex -> true
        | _ -> false

    let isPictureLike label =
        match label with
        | Picture
        | Chart -> true
        | _ -> false

    let contentLayer label =
        match label with
        | PageHeader
        | PageFooter -> Furniture
        | _ -> Body

module DoclingKeywords =
    let maxKeywords = 32

    let sanitizeWithLimit maxKeywords values =
        values
        |> List.choose (fun value ->
            let trimmed = Text.normalizeWhitespace value

            if String.IsNullOrWhiteSpace trimmed then
                None
            else
                Some trimmed)
        |> List.distinctBy _.ToLowerInvariant()
        |> List.truncate (max 0 maxKeywords)

    let sanitize values = sanitizeWithLimit maxKeywords values

    let private labelKeywords label =
        match label with
        | Caption -> [ "caption" ]
        | CheckboxSelected
        | CheckboxUnselected -> [ "checkbox" ]
        | Code -> [ "code" ]
        | DocumentIndex -> [ "document index" ]
        | Footnote -> [ "footnote" ]
        | Formula -> [ "formula" ]
        | ListItem -> [ "list item" ]
        | PageFooter -> [ "page footer" ]
        | PageHeader -> [ "page header" ]
        | Picture -> [ "picture" ]
        | Chart -> [ "chart" ]
        | SectionHeader -> [ "section header" ]
        | Table -> [ "table" ]
        | Title -> [ "title" ]
        | Form -> [ "form" ]
        | KeyValueRegion -> [ "key value region" ]
        | Other value -> [ value.Replace("_", " ") ]
        | Paragraph
        | Text -> []

    let weakDefaults label (figureClasses: DoclingFigureClass list) explicitKeywords =
        let explicitKeywords = sanitize explicitKeywords

        if not (List.isEmpty explicitKeywords) then
            explicitKeywords
        else
            let figureKeywords =
                figureClasses |> List.sortByDescending _.confidence |> List.map _.className

            sanitize (labelKeywords label @ figureKeywords)

module DoclingGeometry =
    let topLeftBox l t r b =
        { l = l
          t = t
          r = r
          b = b
          coordOrigin = TopLeft }

    let bottomLeftBox l t r b =
        { l = l
          t = t
          r = r
          b = b
          coordOrigin = BottomLeft }

    let width bbox = max 0.0 (bbox.r - bbox.l)

    let height bbox = max 0.0 (bbox.b - bbox.t)

    let area bbox = width bbox * height bbox

    let clampToSize (size: DoclingSize) bbox =
        { bbox with
            l = min size.width (max 0.0 bbox.l)
            t = min size.height (max 0.0 bbox.t)
            r = min size.width (max 0.0 bbox.r)
            b = min size.height (max 0.0 bbox.b) }

    let toTopLeft pageHeight bbox =
        match bbox.coordOrigin with
        | TopLeft -> bbox
        | BottomLeft ->
            { bbox with
                t = pageHeight - bbox.b
                b = pageHeight - bbox.t
                coordOrigin = TopLeft }

    let toBottomLeft pageHeight bbox =
        match bbox.coordOrigin with
        | BottomLeft -> bbox
        | TopLeft ->
            { bbox with
                t = pageHeight - bbox.b
                b = pageHeight - bbox.t
                coordOrigin = BottomLeft }

    let intersection a b =
        let l = max a.l b.l
        let t = max a.t b.t
        let r = min a.r b.r
        let bottom = min a.b b.b

        if r <= l || bottom <= t then
            None
        else
            Some
                { l = l
                  t = t
                  r = r
                  b = bottom
                  coordOrigin = a.coordOrigin }

    let intersectionArea a b =
        intersection a b |> Option.map area |> Option.defaultValue 0.0

    let intersectionOverSelf self other =
        let selfArea = area self

        if selfArea <= 0.0 then
            0.0
        else
            intersectionArea self other / selfArea

    let center bbox =
        ((bbox.l + bbox.r) / 2.0, (bbox.t + bbox.b) / 2.0)

    let containsPoint x y bbox =
        x >= bbox.l && x <= bbox.r && y >= bbox.t && y <= bbox.b

    let horizontallyOverlaps a b = min a.r b.r > max a.l b.l

module DoclingCells =
    let private normalizedText (cell: DoclingOcrCell) = Text.normalizeWhitespace cell.text

    let textLength (cells: DoclingOcrCell list) =
        cells |> List.sumBy (fun cell -> (normalizedText cell).Length)

    let hasEnoughText minChars cells = textLength cells >= max 0 minChars

    let scaleCells
        (sourceSize: DoclingSize)
        (targetSize: DoclingSize)
        (cells: DoclingOcrCell list)
        : DoclingOcrCell list =
        let xScale =
            if sourceSize.width <= 0.0 then
                1.0
            else
                targetSize.width / sourceSize.width

        let yScale =
            if sourceSize.height <= 0.0 then
                1.0
            else
                targetSize.height / sourceSize.height

        cells
        |> List.map (fun cell ->
            { cell with
                bbox =
                    { cell.bbox with
                        l = cell.bbox.l * xScale
                        t = cell.bbox.t * yScale
                        r = cell.bbox.r * xScale
                        b = cell.bbox.b * yScale } })

    let scaleCellsToImage sourceSize (image: DoclingRgbImage) cells =
        scaleCells
            sourceSize
            { width = float image.width
              height = float image.height }
            cells

    let private cellForComparison pageHeight (cell: DoclingOcrCell) : DoclingOcrCell =
        { cell with
            text = normalizedText cell
            bbox = DoclingGeometry.toTopLeft pageHeight cell.bbox }

    let private overlaps threshold (existing: DoclingOcrCell) (candidate: DoclingOcrCell) =
        DoclingGeometry.intersectionOverSelf candidate.bbox existing.bbox >= threshold
        || DoclingGeometry.intersectionOverSelf existing.bbox candidate.bbox >= threshold

    let mergePreferPrimary
        pageHeight
        overlapThreshold
        (primary: DoclingOcrCell list)
        (secondary: DoclingOcrCell list)
        : DoclingOcrCell list =
        let threshold = min 1.0 (max 0.0 overlapThreshold)
        let primaryForComparison = primary |> List.map (cellForComparison pageHeight)

        let keepSecondary cell =
            let comparison = cellForComparison pageHeight cell

            not (String.IsNullOrWhiteSpace comparison.text)
            && not (primaryForComparison |> List.exists (overlaps threshold comparison))

        primary @ (secondary |> List.filter keepSecondary)

module DoclingRgbImage =
    let validate image =
        if image.width <= 0 || image.height <= 0 then
            invalidArg (nameof image) "Image dimensions must be positive."

        let expected = image.width * image.height * 3

        if image.pixels.Length <> expected then
            invalidArg (nameof image) $"Expected {expected} RGB bytes, got {image.pixels.Length}."

    let create width height pixels =
        let image =
            { width = width
              height = height
              pixels = pixels }

        validate image
        image

    let solid width height r g b =
        let pixels = Array.zeroCreate<byte> (width * height * 3)

        for index = 0 to width * height - 1 do
            let offset = index * 3
            pixels[offset] <- r
            pixels[offset + 1] <- g
            pixels[offset + 2] <- b

        create width height pixels

    let private clampByteIndex value upper = min (upper - 1) (max 0 value)

    let crop bbox image =
        validate image

        let size =
            { width = float image.width
              height = float image.height }

        let bbox =
            bbox
            |> DoclingGeometry.toTopLeft size.height
            |> DoclingGeometry.clampToSize size

        let left = int (floor bbox.l)
        let top = int (floor bbox.t)
        let right = int (ceil bbox.r)
        let bottom = int (ceil bbox.b)
        let width = max 1 (right - left)
        let height = max 1 (bottom - top)
        let pixels = Array.zeroCreate<byte> (width * height * 3)

        for y = 0 to height - 1 do
            let srcY = clampByteIndex (top + y) image.height

            for x = 0 to width - 1 do
                let srcX = clampByteIndex (left + x) image.width
                let source = (srcY * image.width + srcX) * 3
                let target = (y * width + x) * 3
                pixels[target] <- image.pixels[source]
                pixels[target + 1] <- image.pixels[source + 1]
                pixels[target + 2] <- image.pixels[source + 2]

        create width height pixels

    let resizeBilinear targetWidth targetHeight image =
        validate image

        if targetWidth <= 0 || targetHeight <= 0 then
            invalidArg (nameof targetWidth) "Target dimensions must be positive."

        if targetWidth = image.width && targetHeight = image.height then
            image
        else
            let pixels = Array.zeroCreate<byte> (targetWidth * targetHeight * 3)
            let xScale = float image.width / float targetWidth
            let yScale = float image.height / float targetHeight

            for y = 0 to targetHeight - 1 do
                let sourceY = (float y + 0.5) * yScale - 0.5
                let y0 = int (floor sourceY) |> clampByteIndex <| image.height
                let y1 = clampByteIndex (y0 + 1) image.height
                let wy = sourceY - floor sourceY

                for x = 0 to targetWidth - 1 do
                    let sourceX = (float x + 0.5) * xScale - 0.5
                    let x0 = int (floor sourceX) |> clampByteIndex <| image.width
                    let x1 = clampByteIndex (x0 + 1) image.width
                    let wx = sourceX - floor sourceX

                    for channel = 0 to 2 do
                        let p00 = float image.pixels[(y0 * image.width + x0) * 3 + channel]
                        let p10 = float image.pixels[(y0 * image.width + x1) * 3 + channel]
                        let p01 = float image.pixels[(y1 * image.width + x0) * 3 + channel]
                        let p11 = float image.pixels[(y1 * image.width + x1) * 3 + channel]
                        let top = p00 * (1.0 - wx) + p10 * wx
                        let bottom = p01 * (1.0 - wx) + p11 * wx
                        let value = top * (1.0 - wy) + bottom * wy
                        pixels[(y * targetWidth + x) * 3 + channel] <- byte (min 255.0 (max 0.0 (Math.Round value)))

            create targetWidth targetHeight pixels

module DoclingJson =
    let schemaName = "DoclingDocument"
    let schemaVersion = "1.10.0"

    let private coordOriginValue origin =
        match origin with
        | TopLeft -> "TOPLEFT"
        | BottomLeft -> "BOTTOMLEFT"

    let private contentLayerValue layer =
        match layer with
        | Body -> "body"
        | Furniture -> "furniture"
        | Background -> "background"
        | Invisible -> "invisible"
        | Notes -> "notes"

    let private refNode (cref: string) =
        let node = JsonObject()
        node["$ref"] <- JsonValue.Create(cref)
        node

    let private refArray (refs: string list) =
        let array = JsonArray()

        for ref in refs do
            array.Add(refNode ref)

        array

    let private bboxNode (bbox: DoclingBoundingBox) =
        let node = JsonObject()
        node["l"] <- JsonValue.Create(bbox.l)
        node["t"] <- JsonValue.Create(bbox.t)
        node["r"] <- JsonValue.Create(bbox.r)
        node["b"] <- JsonValue.Create(bbox.b)
        node["coord_origin"] <- JsonValue.Create(coordOriginValue bbox.coordOrigin)
        node

    let private provNode (prov: DoclingProvenance) =
        let node = JsonObject()
        node["page_no"] <- JsonValue.Create(prov.pageNo)
        node["bbox"] <- bboxNode prov.bbox

        prov.charSpan
        |> Option.iter (fun (startIndex, endIndex) ->
            let charSpan = JsonArray()
            charSpan.Add(JsonValue.Create(startIndex))
            charSpan.Add(JsonValue.Create(endIndex))
            node["charspan"] <- charSpan)

        node

    let private provArray (provs: DoclingProvenance list) =
        let array = JsonArray()

        for prov in provs do
            array.Add(provNode prov)

        array

    let private groupNode (selfRef: string) (contentLayer: DoclingContentLayer) (children: string list) =
        let node = JsonObject()
        node["self_ref"] <- JsonValue.Create(selfRef)
        node["parent"] <- null
        node["children"] <- refArray children
        node["content_layer"] <- JsonValue.Create(contentLayerValue contentLayer)
        node["meta"] <- null
        node["name"] <- JsonValue.Create("_root_")
        node["label"] <- JsonValue.Create("unspecified")
        node

    let private pageNode (page: DoclingPageItem) =
        let size = JsonObject()
        size["width"] <- JsonValue.Create(page.size.width)
        size["height"] <- JsonValue.Create(page.size.height)

        let node = JsonObject()
        node["size"] <- size
        node["page_no"] <- JsonValue.Create(page.pageNo)
        node

    let private fscolbertMetaNode sourceId sourceDisplayName keywords : JsonNode =
        let keywords = DoclingKeywords.sanitize keywords

        let sourceId =
            sourceId
            |> Option.map Text.normalizeWhitespace
            |> Option.filter (String.IsNullOrWhiteSpace >> not)

        let sourceDisplayName =
            sourceDisplayName
            |> Option.map Text.normalizeWhitespace
            |> Option.filter (String.IsNullOrWhiteSpace >> not)

        if
            List.isEmpty keywords
            && Option.isNone sourceId
            && Option.isNone sourceDisplayName
        then
            null
        else
            let fscolbert = JsonObject()

            if not (List.isEmpty keywords) then
                let keywordArray = JsonArray()

                for keyword in keywords do
                    keywordArray.Add(JsonValue.Create(keyword))

                fscolbert["keywords"] <- keywordArray

            sourceId
            |> Option.iter (fun value -> fscolbert["source_id"] <- JsonValue.Create(value))

            sourceDisplayName
            |> Option.iter (fun value -> fscolbert["source_display_name"] <- JsonValue.Create(value))

            let meta = JsonObject()
            meta["fscolbert"] <- fscolbert
            meta

    let private baseItemNode
        (selfRef: string)
        (parent: string)
        (label: DoclingLabel)
        (contentLayer: DoclingContentLayer)
        (provs: DoclingProvenance list)
        (keywords: string list)
        (sourceId: string option)
        (sourceDisplayName: string option)
        =
        let node = JsonObject()
        node["self_ref"] <- JsonValue.Create(selfRef)
        node["parent"] <- refNode parent
        node["children"] <- JsonArray()
        node["label"] <- JsonValue.Create(DoclingLabels.toJsonValue label)
        node["content_layer"] <- JsonValue.Create(contentLayerValue contentLayer)
        node["prov"] <- provArray provs
        node["meta"] <- fscolbertMetaNode sourceId sourceDisplayName keywords
        node

    let private textNode (item: DoclingTextItem) =
        let node =
            baseItemNode
                item.selfRef
                item.parent
                item.label
                item.contentLayer
                item.prov
                item.keywords
                item.sourceId
                item.sourceDisplayName

        node["orig"] <- JsonValue.Create(item.orig)
        node["text"] <- JsonValue.Create(item.text)
        node

    let private tableCellNode (cell: DoclingTableCell) =
        let node = JsonObject()
        node["text"] <- JsonValue.Create(cell.text)
        cell.bbox |> Option.iter (fun bbox -> node["bbox"] <- bboxNode bbox)
        node["start_row_offset_idx"] <- JsonValue.Create(cell.startRowOffsetIndex)
        node["end_row_offset_idx"] <- JsonValue.Create(cell.endRowOffsetIndex)
        node["start_col_offset_idx"] <- JsonValue.Create(cell.startColOffsetIndex)
        node["end_col_offset_idx"] <- JsonValue.Create(cell.endColOffsetIndex)
        node["row_header"] <- JsonValue.Create(cell.rowHeader)
        node["column_header"] <- JsonValue.Create(cell.columnHeader)
        node["row_section"] <- JsonValue.Create(cell.rowSection)
        node

    let private tableDataNode (data: DoclingTableData) =
        let cells = JsonArray()

        for cell in data.tableCells do
            cells.Add(tableCellNode cell)

        let node = JsonObject()
        node["num_rows"] <- JsonValue.Create(data.numRows)
        node["num_cols"] <- JsonValue.Create(data.numCols)
        node["table_cells"] <- cells
        node

    let private tableNode (item: DoclingTableItem) =
        let node =
            baseItemNode
                item.selfRef
                item.parent
                item.label
                item.contentLayer
                item.prov
                item.keywords
                item.sourceId
                item.sourceDisplayName

        node["data"] <- tableDataNode item.data
        node

    let private classificationNode (predictions: DoclingFigureClass list) =
        let predictedClasses = JsonArray()

        for prediction in predictions do
            let cls = JsonObject()
            cls["class_name"] <- JsonValue.Create(prediction.className)
            cls["confidence"] <- JsonValue.Create(float prediction.confidence)
            predictedClasses.Add(cls)

        let node = JsonObject()
        node["kind"] <- JsonValue.Create("classification")
        node["provenance"] <- JsonValue.Create("FsColbert")
        node["predicted_classes"] <- predictedClasses
        node

    let private pictureNode (item: DoclingPictureItem) =
        let node =
            baseItemNode
                item.selfRef
                item.parent
                item.label
                item.contentLayer
                item.prov
                item.keywords
                item.sourceId
                item.sourceDisplayName

        if not (List.isEmpty item.classifications) then
            let annotations = JsonArray()
            annotations.Add(classificationNode item.classifications)
            node["annotations"] <- annotations

        node

    let private originNode (fileName: string) (mimeType: string) =
        let node = JsonObject()
        node["filename"] <- JsonValue.Create(fileName)
        node["mimetype"] <- JsonValue.Create(mimeType)
        node["binary_hash"] <- JsonValue.Create(0UL)
        node

    let toJsonNode (document: DoclingDocument) =
        let root = JsonObject()
        root["schema_name"] <- JsonValue.Create(schemaName)
        root["version"] <- JsonValue.Create(schemaVersion)
        root["name"] <- JsonValue.Create(document.name)

        match document.originFileName, document.originMimeType with
        | Some fileName, Some mimeType -> root["origin"] <- originNode fileName mimeType
        | Some fileName, None -> root["origin"] <- originNode fileName "application/pdf"
        | _ -> ()

        let pages = JsonObject()

        for KeyValue(pageNo, page) in document.pages do
            pages[string pageNo] <- pageNode page

        root["pages"] <- pages

        let texts = JsonArray()

        for item in document.texts do
            texts.Add(textNode item)

        root["texts"] <- texts

        let tables = JsonArray()

        for item in document.tables do
            tables.Add(tableNode item)

        root["tables"] <- tables

        let pictures = JsonArray()

        for item in document.pictures do
            pictures.Add(pictureNode item)

        root["pictures"] <- pictures
        root["body"] <- groupNode "#/body" Body document.bodyChildren
        root["furniture"] <- groupNode "#/furniture" Furniture document.furnitureChildren
        root

    let serializeIndented (document: DoclingDocument) =
        let options = JsonSerializerOptions(WriteIndented = true)
        (toJsonNode document).ToJsonString(options)

    let serialize (document: DoclingDocument) = (toJsonNode document).ToJsonString()

    type private ParsedFscolbertMeta =
        { keywords: string list
          sourceId: string option
          sourceDisplayName: string option }

    let private emptyFscolbertMeta =
        { keywords = []
          sourceId = None
          sourceDisplayName = None }

    let private tryProperty (name: string) (element: JsonElement) =
        if element.ValueKind <> JsonValueKind.Object then
            None
        else
            let mutable value = Unchecked.defaultof<JsonElement>

            if element.TryGetProperty(name, &value) then
                Some value
            else
                None

    let private stringValue (element: JsonElement) =
        if element.ValueKind = JsonValueKind.String then
            element.GetString() |> Option.ofObj
        else
            None

    let private stringProperty name element =
        tryProperty name element |> Option.bind stringValue

    let private intProperty name fallback element =
        match tryProperty name element with
        | Some value when value.ValueKind = JsonValueKind.Number ->
            match value.TryGetInt32() with
            | true, result -> result
            | _ -> fallback
        | _ -> fallback

    let private floatProperty name fallback element =
        match tryProperty name element with
        | Some value when value.ValueKind = JsonValueKind.Number ->
            match value.TryGetDouble() with
            | true, result -> result
            | _ -> fallback
        | _ -> fallback

    let private boolProperty name fallback element =
        match tryProperty name element with
        | Some value when value.ValueKind = JsonValueKind.True -> true
        | Some value when value.ValueKind = JsonValueKind.False -> false
        | _ -> fallback

    let private parseRef (fallback: string) (element: JsonElement) =
        match element.ValueKind with
        | JsonValueKind.Object -> stringProperty "$ref" element |> Option.defaultValue fallback
        | JsonValueKind.String -> stringValue element |> Option.defaultValue fallback
        | _ -> fallback

    let private parseParent fallback element =
        tryProperty "parent" element
        |> Option.map (parseRef fallback)
        |> Option.defaultValue fallback

    let private parseCoordOrigin value =
        match value with
        | Some origin when String.Equals(origin, "BOTTOMLEFT", StringComparison.OrdinalIgnoreCase) -> BottomLeft
        | _ -> TopLeft

    let private parseContentLayer value =
        match value with
        | Some layer when String.Equals(layer, "furniture", StringComparison.OrdinalIgnoreCase) -> Furniture
        | Some layer when String.Equals(layer, "background", StringComparison.OrdinalIgnoreCase) -> Background
        | Some layer when String.Equals(layer, "invisible", StringComparison.OrdinalIgnoreCase) -> Invisible
        | Some layer when String.Equals(layer, "notes", StringComparison.OrdinalIgnoreCase) -> Notes
        | _ -> Body

    let private parseBBox element =
        { l = floatProperty "l" 0.0 element
          t = floatProperty "t" 0.0 element
          r = floatProperty "r" 0.0 element
          b = floatProperty "b" 0.0 element
          coordOrigin = parseCoordOrigin (stringProperty "coord_origin" element) }

    let private emptyBBox = DoclingGeometry.topLeftBox 0.0 0.0 0.0 0.0

    let private parseCharSpan element =
        match tryProperty "charspan" element with
        | Some charSpan when charSpan.ValueKind = JsonValueKind.Array ->
            let values = charSpan.EnumerateArray() |> Seq.toArray

            if
                values.Length >= 2
                && values[0].ValueKind = JsonValueKind.Number
                && values[1].ValueKind = JsonValueKind.Number
            then
                match values[0].TryGetInt32(), values[1].TryGetInt32() with
                | (true, startIndex), (true, endIndex) -> Some(startIndex, endIndex)
                | _ -> None
            else
                None
        | _ -> None

    let private parseProvenance element =
        let bbox =
            tryProperty "bbox" element
            |> Option.map parseBBox
            |> Option.defaultValue emptyBBox

        { pageNo = intProperty "page_no" 1 element
          bbox = bbox
          charSpan = parseCharSpan element }

    let private parseProvenanceArray element =
        match tryProperty "prov" element with
        | Some provs when provs.ValueKind = JsonValueKind.Array ->
            provs.EnumerateArray() |> Seq.map parseProvenance |> Seq.toList
        | _ -> []

    let private parseStringArray (element: JsonElement) =
        if element.ValueKind = JsonValueKind.Array then
            element.EnumerateArray() |> Seq.choose stringValue |> Seq.toList
        else
            []

    let private parseFscolbertMeta element =
        match tryProperty "meta" element |> Option.bind (tryProperty "fscolbert") with
        | None -> emptyFscolbertMeta
        | Some fscolbert ->
            { keywords =
                tryProperty "keywords" fscolbert
                |> Option.map parseStringArray
                |> Option.defaultValue []
                |> DoclingKeywords.sanitize
              sourceId = stringProperty "source_id" fscolbert
              sourceDisplayName = stringProperty "source_display_name" fscolbert }

    let private parseTableCell element =
        { text = stringProperty "text" element |> Option.defaultValue ""
          bbox = tryProperty "bbox" element |> Option.map parseBBox
          startRowOffsetIndex = intProperty "start_row_offset_idx" 0 element
          endRowOffsetIndex = intProperty "end_row_offset_idx" 1 element
          startColOffsetIndex = intProperty "start_col_offset_idx" 0 element
          endColOffsetIndex = intProperty "end_col_offset_idx" 1 element
          rowHeader = boolProperty "row_header" false element
          columnHeader = boolProperty "column_header" false element
          rowSection = boolProperty "row_section" false element }

    let private parseTableData element =
        let cells =
            match tryProperty "table_cells" element with
            | Some cells when cells.ValueKind = JsonValueKind.Array ->
                cells.EnumerateArray() |> Seq.map parseTableCell |> Seq.toList
            | _ -> []

        { numRows = intProperty "num_rows" cells.Length element
          numCols = intProperty "num_cols" 0 element
          tableCells = cells }

    let private parseClassifications element =
        let parsePredictedClass prediction =
            { className = stringProperty "class_name" prediction |> Option.defaultValue ""
              confidence = float32 (floatProperty "confidence" 0.0 prediction) }

        match tryProperty "annotations" element with
        | Some annotations when annotations.ValueKind = JsonValueKind.Array ->
            annotations.EnumerateArray()
            |> Seq.filter (fun annotation ->
                stringProperty "kind" annotation
                |> Option.map (fun kind -> String.Equals(kind, "classification", StringComparison.OrdinalIgnoreCase))
                |> Option.defaultValue false)
            |> Seq.collect (fun annotation ->
                match tryProperty "predicted_classes" annotation with
                | Some predictions when predictions.ValueKind = JsonValueKind.Array ->
                    predictions.EnumerateArray() |> Seq.map parsePredictedClass
                | _ -> Seq.empty)
            |> Seq.filter (fun cls -> not (String.IsNullOrWhiteSpace cls.className))
            |> Seq.toList
        | _ -> []

    let private parsePage fallbackPageNo element =
        let size = tryProperty "size" element

        { pageNo = intProperty "page_no" fallbackPageNo element
          size =
            { width = size |> Option.map (floatProperty "width" 0.0) |> Option.defaultValue 0.0
              height = size |> Option.map (floatProperty "height" 0.0) |> Option.defaultValue 0.0 } }

    let private parsePages root =
        match tryProperty "pages" root with
        | Some pages when pages.ValueKind = JsonValueKind.Object ->
            pages.EnumerateObject()
            |> Seq.choose (fun property ->
                match Int32.TryParse property.Name with
                | true, pageNo -> Some(pageNo, parsePage pageNo property.Value)
                | _ -> None)
            |> Map.ofSeq
        | _ -> Map.empty

    let private parseGroupRefs groupName root =
        match tryProperty groupName root |> Option.bind (tryProperty "children") with
        | Some children when children.ValueKind = JsonValueKind.Array ->
            children.EnumerateArray()
            |> Seq.map (parseRef "")
            |> Seq.filter (String.IsNullOrWhiteSpace >> not)
            |> Seq.toList
        | _ -> []

    let private parseTextItem index element =
        let label =
            stringProperty "label" element
            |> Option.map DoclingLabels.ofJsonValue
            |> Option.defaultValue Text

        let meta = parseFscolbertMeta element

        { selfRef = stringProperty "self_ref" element |> Option.defaultValue $"#/texts/{index}"
          parent = parseParent "#/body" element
          label = label
          text = stringProperty "text" element |> Option.defaultValue ""
          orig =
            stringProperty "orig" element
            |> Option.defaultValue (stringProperty "text" element |> Option.defaultValue "")
          contentLayer = parseContentLayer (stringProperty "content_layer" element)
          prov = parseProvenanceArray element
          keywords = DoclingKeywords.weakDefaults label [] meta.keywords
          sourceId = meta.sourceId
          sourceDisplayName = meta.sourceDisplayName }

    let private parseTableItem index element =
        let label =
            stringProperty "label" element
            |> Option.map DoclingLabels.ofJsonValue
            |> Option.defaultValue Table

        let meta = parseFscolbertMeta element

        { selfRef = stringProperty "self_ref" element |> Option.defaultValue $"#/tables/{index}"
          parent = parseParent "#/body" element
          label = label
          contentLayer = parseContentLayer (stringProperty "content_layer" element)
          prov = parseProvenanceArray element
          data =
            tryProperty "data" element
            |> Option.map parseTableData
            |> Option.defaultValue
                { numRows = 0
                  numCols = 0
                  tableCells = [] }
          keywords = DoclingKeywords.weakDefaults label [] meta.keywords
          sourceId = meta.sourceId
          sourceDisplayName = meta.sourceDisplayName }

    let private parsePictureItem index element =
        let label =
            stringProperty "label" element
            |> Option.map DoclingLabels.ofJsonValue
            |> Option.defaultValue Picture

        let meta = parseFscolbertMeta element
        let classifications = parseClassifications element

        { selfRef = stringProperty "self_ref" element |> Option.defaultValue $"#/pictures/{index}"
          parent = parseParent "#/body" element
          label = label
          contentLayer = parseContentLayer (stringProperty "content_layer" element)
          prov = parseProvenanceArray element
          classifications = classifications
          keywords = DoclingKeywords.weakDefaults label classifications meta.keywords
          sourceId = meta.sourceId
          sourceDisplayName = meta.sourceDisplayName }

    let private parseArray propertyName parseItem root =
        match tryProperty propertyName root with
        | Some items when items.ValueKind = JsonValueKind.Array ->
            items.EnumerateArray() |> Seq.mapi parseItem |> Seq.toList
        | _ -> []

    let tryDeserialize (json: string) =
        try
            use parsed = JsonDocument.Parse json
            let root = parsed.RootElement

            match stringProperty "schema_name" root with
            | Some schema when String.Equals(schema, schemaName, StringComparison.Ordinal) ->
                let origin = tryProperty "origin" root

                Ok
                    { name = stringProperty "name" root |> Option.defaultValue "document"
                      originFileName = origin |> Option.bind (stringProperty "filename")
                      originMimeType = origin |> Option.bind (stringProperty "mimetype")
                      pages = parsePages root
                      texts = parseArray "texts" parseTextItem root
                      tables = parseArray "tables" parseTableItem root
                      pictures = parseArray "pictures" parsePictureItem root
                      bodyChildren = parseGroupRefs "body" root
                      furnitureChildren = parseGroupRefs "furniture" root }
            | Some schema -> Error $"Expected schema_name '{schemaName}', got '{schema}'."
            | None -> Error "JSON is not a DoclingDocument because schema_name is missing."
        with ex ->
            Error $"Unable to parse DoclingDocument JSON: {ex.Message}"

    let deserialize json =
        match tryDeserialize json with
        | Ok document -> document
        | Error err -> invalidArg (nameof json) err

    let validateSubset (document: DoclingDocument) =
        let errors = ResizeArray<string>()

        if String.IsNullOrWhiteSpace document.name then
            errors.Add("Document name is required.")

        if Map.isEmpty document.pages then
            errors.Add("At least one page is required.")

        let existingRefs =
            seq {
                yield "#/body"
                yield "#/furniture"

                for item in document.texts do
                    yield item.selfRef

                for item in document.tables do
                    yield item.selfRef

                for item in document.pictures do
                    yield item.selfRef
            }
            |> Set.ofSeq

        for child in document.bodyChildren @ document.furnitureChildren do
            if not (existingRefs.Contains child) then
                errors.Add($"Missing referenced child: {child}")

        for item in document.texts do
            if String.IsNullOrWhiteSpace item.selfRef then
                errors.Add("Text item self_ref is required.")

            if String.IsNullOrWhiteSpace item.text then
                errors.Add($"Text item {item.selfRef} has empty text.")

        for item in document.tables do
            if String.IsNullOrWhiteSpace item.selfRef then
                errors.Add("Table item self_ref is required.")

            if item.data.numRows < 0 || item.data.numCols < 0 then
                errors.Add($"Table item {item.selfRef} has invalid dimensions.")

        errors |> Seq.toList
