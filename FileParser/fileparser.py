import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader
import json

from .baseclass import FileParserBaseClass

#### csv file parser currently not used
class CSVParser(FileParserBaseClass):
    def __init__(self,filename):
        self.file_name=filename
    
    def parse(self):
        try:
            df = pd.read_csv(self.file_name)
            return df
        except:
            return None

class JSONparser(FileParserBaseClass):
    def __init__(self,filename):
        self.file_name=filename
    
    def parse(self):
        try:
            with open(self.file_name, "r") as file:
                data = json.load(file)
                return data
        except:
            return None
        
class HTMLparser(FileParserBaseClass):
    def __init__(self,filename):
        self.file_name=filename
    
    def parse(self):
        try:
            loader=UnstructuredHTMLLoader(self.file_name)
            return loader.load()
        except:
            return None

class PDFParser(FileParserBaseClass):
    def __init__(self, filename):
        self.file_name = filename
    
    def parse(self):
        try:
            loader = PyPDFLoader(self.file_name)
            pages = loader.load_and_split()
            return pages
        except:
            return None


class TextFileParser(FileParserBaseClass):
    def __init__(self, filename):
        self.file_name = filename
    
    def parse(self):
        try:
            loader = TextLoader(self.file_name, encoding = 'UTF-8')
            return loader.load()
        except FileNotFoundError:
            return None
        except Exception as e:
            return None


class FileParserFactory():
    def __init__(self,file_type,file_name) -> None:
        self.type=file_type
        self.file_parsers={
            "csv":CSVParser(file_name),
            "pdf":PDFParser(file_name),
            "txt":TextFileParser(file_name),
            "html":HTMLparser(file_name),
            "htm":HTMLparser(file_name),
            "json":JSONparser(file_name)
        }

    def parse(self):
        return self.file_parsers[self.type].parse()


