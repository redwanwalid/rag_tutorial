"""Stages 1 and 2: load documents and recursively split them into passages."""
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(folder: Path):
    if not folder.is_dir():
        raise ValueError(f'Document directory does not exist: {folder}')
    documents = []
    for path in sorted(folder.rglob('*')):
        if path.is_file() and path.suffix.lower() in ('.md', '.txt'):
            text = path.read_text(encoding='utf-8')
            if text.strip():
                documents.append(Document(
                    page_content=text,
                    metadata={'source': path.relative_to(folder).as_posix()},
                ))
    if not documents:
        raise ValueError('No nonempty .md or .txt documents found; existing index preserved.')
    return documents


def split_documents(documents, size=1000, overlap=200):
    if not 0 <= overlap < size <= 4000:
        raise ValueError('Require 0 <= overlap < size <= 4000 characters.')
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap, add_start_index=True,
        strip_whitespace=False, separators=['\n\n', '\n', '. ', ' ', ''],
    )
    chunks = [c for c in splitter.split_documents(documents) if c.page_content.strip()]
    for chunk in chunks:
        metadata = chunk.metadata
        identity = f"{metadata['source']}:{metadata['start_index']}:{chunk.page_content}"
        metadata['chunk_id'] = str(uuid5(NAMESPACE_URL, identity))
    if not chunks:
        raise ValueError('No usable chunks produced.')
    return chunks
