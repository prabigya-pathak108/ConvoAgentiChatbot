from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import HTMLHeaderTextSplitter
from langchain_text_splitters import RecursiveJsonSplitter


class CharacterTextSplitting:
    def __init__(self, document):
        self.document = document
    
    def chunking(self):
        try:
            text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base", chunk_size=1000, chunk_overlap=200
        )
            texts = text_splitter.split_documents(self.document)
            return texts
        except:
            return None


class RecursiveCharacterTextSplitting:
    def __init__(self, document):
        self.document = document
    
    def chunking(self):
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(self.document)
            return texts
        except:
            return None

class JSONChunker:
    def __init__(self, document):
        self.document = document
    
    def chunking(self):
        try:
            splitter = RecursiveJsonSplitter(max_chunk_size=300)
            json_chunks = splitter.create_documents(texts=[self.document])
            return json_chunks
        except:
            return None

class HTMLSplitting:
    def __init__(self, document):
        self.headers_to_split_on = document
    
    def chunking(self):
        try:
            headers_to_split_on = [
                ("h1", "Header 1"),
                ("h2", "Header 2"),
                ("h3", "Header 3"),
            ]

            html_splitter = HTMLHeaderTextSplitter(headers_to_split_on)
            html_header_splits = html_splitter.split_documents(self.html_string)
            return html_header_splits
        except:
            return None

class VectorStoreMakingFactory():
    def __init__(self,document,file_extension,splitting_type:str=None):
        self.type=splitting_type
        self.file_parsers={
            "character":CharacterTextSplitting(document),
            "recursive":RecursiveCharacterTextSplitting(document),
            "html":HTMLSplitting(document),
            "json":JSONChunker(document)
        }
        self.file_extension=file_extension

    def splittext(self):
        if self.file_extension in ["htm","html"]:
            return self.file_parsers["html"].chunking()
        if self.file_extension in ["json","csv"]:
            return self.file_parsers["json"].chunking()
        if self.type is not None:
            return self.file_parsers[self.type].chunking()
        else:
            print("Please enter a valid splitting type")
            raise ValueError