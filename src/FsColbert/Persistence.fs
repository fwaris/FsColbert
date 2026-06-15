namespace FsColbert

open System
open System.Collections.Frozen
open System.IO

module IndexPersistence =
    let private magic = "FSCOLBERT-IDX"
    let private version = 6
    let private minimumReadableVersion = 2

    let private writeConfig (writer: BinaryWriter) (config: EncoderConfig) =
        writer.Write config.queryLength
        writer.Write config.documentLength
        writer.Write config.embeddingDim
        writer.Write config.padTokenId
        writer.Write config.maskTokenId
        writer.Write config.clsTokenId
        writer.Write config.sepTokenId
        writer.Write config.queryPrefixId
        writer.Write config.documentPrefixId
        writer.Write config.doLowerCase
        writer.Write config.normalizeOutput
        writer.Write config.skiplistWords.Count

        for word in config.skiplistWords do
            writer.Write word

    let private readConfig (reader: BinaryReader) =
        let queryLength = reader.ReadInt32()
        let documentLength = reader.ReadInt32()
        let embeddingDim = reader.ReadInt32()
        let padTokenId = reader.ReadInt32()
        let maskTokenId = reader.ReadInt32()
        let clsTokenId = reader.ReadInt32()
        let sepTokenId = reader.ReadInt32()
        let queryPrefixId = reader.ReadInt32()
        let documentPrefixId = reader.ReadInt32()
        let doLowerCase = reader.ReadBoolean()
        let normalizeOutput = reader.ReadBoolean()
        let skiplistCount = reader.ReadInt32()

        let skiplistWords =
            [ for _ = 1 to skiplistCount do
                  reader.ReadString() ]
            |> Set.ofList

        { queryLength = queryLength
          documentLength = documentLength
          embeddingDim = embeddingDim
          padTokenId = padTokenId
          maskTokenId = maskTokenId
          clsTokenId = clsTokenId
          sepTokenId = sepTokenId
          queryPrefixId = queryPrefixId
          documentPrefixId = documentPrefixId
          doLowerCase = doLowerCase
          normalizeOutput = normalizeOutput
          skiplistWords = skiplistWords }

    let private writeVector (writer: BinaryWriter) (embedding: MultiVector) =
        writer.Write embedding.embeddingDim
        writer.Write embedding.tokenCount
        writer.Write embedding.tokenIds.Length

        for tokenId in embedding.tokenIds do
            writer.Write tokenId

        writer.Write embedding.vectors.Length

        for value in embedding.vectors do
            writer.Write value

    let private readVector (reader: BinaryReader) =
        let embeddingDim = reader.ReadInt32()
        let tokenCount = reader.ReadInt32()

        let tokenIds =
            [| for _ = 1 to reader.ReadInt32() do
                   reader.ReadInt32() |]

        let vectors =
            [| for _ = 1 to reader.ReadInt32() do
                   reader.ReadSingle() |]

        { tokenIds = tokenIds
          vectors = vectors
          tokenCount = tokenCount
          embeddingDim = embeddingDim }

    let private writeTfidfOptions (writer: BinaryWriter) (options: TfidfOptions) =
        writer.Write options.textWeight
        writer.Write options.keywordWeight

    let private readTfidfOptions (reader: BinaryReader) =
        { textWeight = reader.ReadSingle()
          keywordWeight = reader.ReadSingle() }

    let private writeStringList (writer: BinaryWriter) (values: string list) =
        writer.Write values.Length

        for value in values do
            writer.Write value

    let private readStringList (reader: BinaryReader) =
        [ for _ = 1 to reader.ReadInt32() do
              reader.ReadString() ]

    let private writeIntList (writer: BinaryWriter) (values: int list) =
        writer.Write values.Length

        for value in values do
            writer.Write value

    let private readIntList (reader: BinaryReader) =
        [ for _ = 1 to reader.ReadInt32() do
              reader.ReadInt32() ]

    let private writeTfidf (writer: BinaryWriter) (index: TfidfIndex) =
        writer.Write index.passageCount
        writer.Write index.averageDocumentLength
        writer.Write index.vocabulary.Count

        for KeyValue(term, termInfo) in index.vocabulary |> Seq.sortBy _.Key do
            writer.Write term
            writer.Write termInfo.documentFrequency
            writer.Write termInfo.inverseDocumentFrequency
            writer.Write termInfo.postings.Length

            for posting in termInfo.postings do
                writer.Write posting.passageOrdinal
                writer.Write posting.termFrequency

    let private readTfidf (reader: BinaryReader) =
        let passageCount = reader.ReadInt32()
        let averageDocumentLength = reader.ReadSingle()

        let vocabulary =
            [ for _ = 1 to reader.ReadInt32() do
                  let term = reader.ReadString()
                  let documentFrequency = reader.ReadInt32()
                  let inverseDocumentFrequency = reader.ReadSingle()

                  let postings =
                      [| for _ = 1 to reader.ReadInt32() do
                             { passageOrdinal = reader.ReadInt32()
                               termFrequency = reader.ReadSingle() } |]

                  term,
                  { documentFrequency = documentFrequency
                    inverseDocumentFrequency = inverseDocumentFrequency
                    postings = postings } ]
            |> Seq.map (fun (term, termInfo) -> Collections.Generic.KeyValuePair(term, termInfo))
            |> fun pairs -> FrozenDictionary.ToFrozenDictionary(pairs, StringComparer.Ordinal)

        { passageCount = passageCount
          averageDocumentLength = averageDocumentLength
          vocabulary = vocabulary }

    let save (path: string) (index: ColbertIndex) =
        Path.GetDirectoryName(Path.GetFullPath path)
        |> Option.ofObj
        |> Option.iter (fun folder -> Directory.CreateDirectory folder |> ignore)

        use stream = File.Create path
        use writer = new BinaryWriter(stream)
        writer.Write magic
        writer.Write version
        writeConfig writer index.config
        writer.Write index.chunkOptions.maxChars
        writer.Write index.chunkOptions.overlapChars
        writer.Write index.chunkOptions.minChars
        writeTfidfOptions writer index.tfidfOptions
        writer.Write(index.createdAt.ToUnixTimeMilliseconds())
        writer.Write index.passages.Length

        for passage in index.passages do
            writer.Write passage.reference.sourceId
            writer.Write passage.reference.sourceDisplayName
            writer.Write passage.reference.sourceLocation
            writer.Write passage.reference.index
            writer.Write passage.reference.text
            writeStringList writer passage.reference.keywords
            writeStringList writer passage.reference.sectionPath
            writer.Write(PassageContentRole.storageValue passage.reference.contentRole)
            writeIntList writer passage.reference.pageNumbers
            writeStringList writer passage.reference.layoutLabels
            writeStringList writer passage.reference.captions
            writer.Write passage.terms.Count

            for term in passage.terms do
                writer.Write term

            writeVector writer passage.embedding

        writeTfidf writer index.tfidf

    let load (path: string) =
        use stream = File.OpenRead path
        use reader = new BinaryReader(stream)

        let fileMagic = reader.ReadString()

        if fileMagic <> magic then
            raise (InvalidDataException $"'{path}' is not an FsColbert index file.")

        let fileVersion = reader.ReadInt32()

        if fileVersion < minimumReadableVersion || fileVersion > version then
            raise (InvalidDataException $"Unsupported FsColbert index version {fileVersion}.")

        let config = readConfig reader

        let chunkOptions =
            { maxChars = reader.ReadInt32()
              overlapChars = reader.ReadInt32()
              minChars = reader.ReadInt32() }

        let tfidfOptions =
            if fileVersion >= 3 then
                readTfidfOptions reader
            else
                TfidfOptions.defaults

        let createdAt = DateTimeOffset.FromUnixTimeMilliseconds(reader.ReadInt64())

        let passages =
            [ for _ = 1 to reader.ReadInt32() do
                  let sourceId = reader.ReadString()
                  let sourceDisplayName = reader.ReadString()
                  let sourceLocation = reader.ReadString()
                  let index = reader.ReadInt32()
                  let text = reader.ReadString()

                  let keywords = if fileVersion >= 3 then readStringList reader else []

                  let sectionPath = if fileVersion >= 4 then readStringList reader else []

                  let contentRole =
                      if fileVersion >= 5 then
                          reader.ReadString() |> PassageContentRole.ofStorageValue
                      else
                          PassageContentRole.Unknown

                  let pageNumbers = if fileVersion >= 5 then readIntList reader else []
                  let layoutLabels = if fileVersion >= 6 then readStringList reader else []
                  let captions = if fileVersion >= 6 then readStringList reader else []

                  let reference =
                      { sourceId = sourceId
                        sourceDisplayName = sourceDisplayName
                        sourceLocation = sourceLocation
                        index = index
                        text = text
                        sectionPath = sectionPath
                        contentRole = contentRole
                        pageNumbers = pageNumbers
                        layoutLabels = layoutLabels
                        captions = captions
                        keywords = keywords }

                  let terms =
                      [ for _ = 1 to reader.ReadInt32() do
                            reader.ReadString() ]
                      |> Set.ofList

                  { reference = reference
                    terms = terms
                    embedding = readVector reader } ]

        { config = config
          chunkOptions = chunkOptions
          tfidfOptions = tfidfOptions
          passages = passages
          tfidf = readTfidf reader
          createdAt = createdAt }

    let tryLoad (path: string) =
        if File.Exists path then
            try
                Ok(Some(load path))
            with ex ->
                Error $"Unable to load FsColbert index '{path}': {ex.Message}"
        else
            Ok None

    let loadMany (folder: string) =
        if Directory.Exists folder then
            Directory.EnumerateFiles(folder, "*.fsci")
            |> Seq.map (fun path -> path, tryLoad path)
            |> Seq.toList
        else
            []

    let saveAsync path index = async { save path index }

    let loadAsync path = async { return load path }

    let tryLoadAsync path = async { return tryLoad path }
