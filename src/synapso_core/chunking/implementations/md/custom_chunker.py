from typing import List

import nltk
from bs4 import BeautifulSoup

# Step 1: Parse Markdown to clean text
from langchain_text_splitters import NLTKTextSplitter
from markdown import markdown

from ...interface import Chunker


# Step 1: Parse Markdown to clean text
def markdown_to_text(md):
    html = markdown(md)
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text()


# Step 2: Apply NLTK sentence-aware splitting
nltk.download("punkt")  # Run once


class CustomChunker(Chunker):
    def __init__(self):
        self.splitter = NLTKTextSplitter(chunk_size=512, chunk_overlap=50)

    def chunk_text(self, text: str) -> List[str]:
        converted_text = markdown_to_text(text)
        return self.splitter.split_text(converted_text)

    def chunk_file(self, file_path: str) -> List[str]:
        with open(file_path, "r", encoding="utf-8") as file:
            return self.chunk_text(file.read())

    def is_file_supported(self, file_path: str) -> bool:
        return file_path.endswith(".md")
