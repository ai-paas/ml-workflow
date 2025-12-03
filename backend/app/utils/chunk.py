from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypeAlias

from fastapi import UploadFile
from langchain_core.documents import Document
from surrochat.document_loaders import (
    CSVFileLoader,
    DocFileLoader,
    DocxFileLoader,
    ExcelLoader,
    PDFLoader,
    PowerpointLoader,
)
from surrochat.document_loaders.utils import cleanse_whitespace

TypeFileLoader: TypeAlias = CSVFileLoader | ExcelLoader | PDFLoader | PowerpointLoader | DocFileLoader | DocxFileLoader


class FileTypeEnum(Enum):
    CSV = (".csv", CSVFileLoader)
    EXCEL = (".xls", ExcelLoader)
    EXCEL_XLSX = (".xlsx", ExcelLoader)
    PDF = (".pdf", PDFLoader)
    POWERPOINT = (".ppt", PowerpointLoader)
    POWERPOINT_X = (".pptx", PowerpointLoader)
    WORD = (".doc", DocFileLoader)
    WORD_X = (".docx", DocxFileLoader)

    def __init__(self, extension: str, loader_class: TypeFileLoader):
        self.extension = extension
        self.loader_class = loader_class

    @staticmethod
    def get_loader_class(ext: str) -> TypeFileLoader:
        for file_type in FileTypeEnum:
            if file_type.extension == ext:
                return file_type.loader_class
        raise ValueError(f"Unsupported file extension: {ext}")


def file_load_and_split(file_data, filename: str, chunk_size: int, overlap: int) -> list[Document]:
    ext = Path(filename).suffix.lower()
    file_loader = FileTypeEnum.get_loader_class(ext)
    with TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        temp_file_path = temp_dir_path / filename
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file_data)
        assert temp_file_path.exists(), f"File path {temp_file_path} does not exist"
        file_path = str(temp_file_path)
        documents = file_loader(file_path).load_and_split(chunk_size=chunk_size, overlap=overlap)
        return documents


def get_file_extension(file: UploadFile):
    filename = Path(file.filename)
    ext = filename.suffix.lower().lstrip(".")
    return ext if ext else "no extension"
