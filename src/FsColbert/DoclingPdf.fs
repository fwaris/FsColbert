namespace FsColbert

open System
open UglyToad.PdfPig
open UglyToad.PdfPig.Content
open UglyToad.PdfPig.DocumentLayoutAnalysis.WordExtractor

module DoclingPdfNative =
    let private wordCell (word: Word) : DoclingOcrCell option =
        let text = Text.normalizeWhitespace word.Text

        if String.IsNullOrWhiteSpace text then
            None
        else
            let bbox = word.BoundingBox

            Some
                { text = text
                  bbox = DoclingGeometry.bottomLeftBox bbox.Left bbox.Bottom bbox.Right bbox.Top
                  confidence = None }

    let readPageCells (path: string) : Async<Result<DoclingNativePageText list, string>> =
        async {
            try
                use document = PdfDocument.Open path
                let wordExtractor = NearestNeighbourWordExtractor.Instance

                let pages =
                    document.GetPages()
                    |> Seq.map (fun page ->
                        let cells =
                            wordExtractor.GetWords(page.Letters) |> Seq.choose wordCell |> Seq.toList

                        { pageNo = page.Number
                          size =
                            { width = page.Width
                              height = page.Height }
                          cells = cells })
                    |> Seq.toList

                return Ok pages
            with ex ->
                return Error $"Unable to extract native PDF text cells from '{path}': {ex.Message}"
        }
