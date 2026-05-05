namespace FsColbert

module SourceDocuments =
    let create id displayName location text enabled =
        { id = id
          displayName = displayName
          location = location
          text = text
          enabled = enabled }

    let fromFsKamePdf id displayName storedPath text selected =
        create id $"PDF: {displayName}" storedPath text selected

    let createPreChunked id displayName location chunks enabled : PreChunkedDocument =
        { id = id
          displayName = displayName
          location = location
          chunks = chunks
          enabled = enabled }

    let fromFsKamePdfChunked id displayName storedPath chunks selected =
        createPreChunked id $"PDF: {displayName}" storedPath chunks selected

module SearchHits =
    let renderContext maxChars (hits: SearchHit list) =
        if List.isEmpty hits then
            "No selected PDF context was available."
        else
            hits
            |> List.mapi (fun index hit ->
                let body = Text.truncate maxChars hit.reference.text
                $"[{index + 1}] {hit.reference.sourceDisplayName} chunk {hit.reference.index}\n{body}")
            |> String.concat "\n\n"

    let sourceInventory (sources: SourceDocument list) =
        let enabled = sources |> List.filter _.enabled

        if List.isEmpty enabled then
            "No PDF sources are currently selected and ready."
        else
            enabled
            |> List.mapi (fun index source -> $"[{index + 1}] {source.displayName}")
            |> String.concat "\n"
