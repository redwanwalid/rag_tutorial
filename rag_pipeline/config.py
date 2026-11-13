"""Configuration shared by ingestion, retrieval, and generation."""
from dataclasses import dataclass, field
import os
from pathlib import Path
from urllib.parse import urlsplit
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    data_dir: Path = ROOT / 'data/books'
    store_dir: Path = ROOT / '.rag'
    embedding_model: str = 'text-embedding-3-small'
    chat_model: str = 'gpt-4o-mini'
    chunk_size: int = 1000
    chunk_overlap: int = 200
    batch_size: int = 32
    vector_mode: str = 'local'
    qdrant_url: str = 'http://localhost:6333'
    qdrant_api_key: str = field(default='', repr=False)
    hnsw_m: int = 16
    hnsw_ef_construct: int = 100
    hnsw_ef: int = 64
    indexing_threshold: int = 20000
    full_scan_threshold: int = 10000

    @classmethod
    def from_env(cls):
        load_dotenv(ROOT / '.env', override=False)
        def path(name, default):
            value = Path(os.getenv(name, default)).expanduser()
            return (ROOT / value).resolve() if not value.is_absolute() else value
        settings = cls(
            data_dir=path('RAG_DATA_DIR', 'data/books'),
            store_dir=path('RAG_STORE_DIR', '.rag'),
            embedding_model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small'),
            chat_model=os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o-mini'),
            chunk_size=int(os.getenv('RAG_CHUNK_SIZE', '1000')),
            chunk_overlap=int(os.getenv('RAG_CHUNK_OVERLAP', '200')),
            vector_mode=os.getenv('RAG_VECTOR_MODE', 'local').strip().lower(),
            qdrant_url=os.getenv('QDRANT_URL', 'http://localhost:6333').strip().rstrip('/'),
            qdrant_api_key=os.getenv('QDRANT_API_KEY', '').strip(),
            hnsw_m=int(os.getenv('RAG_HNSW_M', '16')),
            hnsw_ef_construct=int(os.getenv('RAG_HNSW_EF_CONSTRUCT', '100')),
            hnsw_ef=int(os.getenv('RAG_HNSW_EF', '64')),
            indexing_threshold=int(os.getenv('RAG_INDEXING_THRESHOLD_KB', '20000')),
            full_scan_threshold=int(os.getenv('RAG_FULL_SCAN_THRESHOLD_KB', '10000')),
        )
        settings.validate()
        return settings

    def validate(self):
        if self.vector_mode not in ('local', 'server'):
            raise ValueError('RAG_VECTOR_MODE must be local or server.')
        if self.vector_mode == 'server':
            url = urlsplit(self.qdrant_url)
            if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password or url.query or url.fragment:
                raise ValueError('QDRANT_URL must be an HTTP(S) URL without credentials, query, or fragment; use QDRANT_API_KEY for authentication.')
        if min(self.hnsw_m, self.hnsw_ef_construct, self.hnsw_ef) < 1:
            raise ValueError('HNSW parameters must be positive integers.')
        if self.hnsw_ef_construct < 4:
            raise ValueError('RAG_HNSW_EF_CONSTRUCT must be at least 4.')
        if self.indexing_threshold < 1:
            raise ValueError('RAG_INDEXING_THRESHOLD_KB must be at least 1.')
        if self.full_scan_threshold < 10:
            raise ValueError('RAG_FULL_SCAN_THRESHOLD_KB must be at least 10 for Qdrant 1.14.1.')
        if not 0 <= self.chunk_overlap < self.chunk_size <= 4000:
            raise ValueError('Require 0 <= chunk overlap < chunk size <= 4000 characters.')
        if self.batch_size < 1 or not self.embedding_model.strip() or not self.chat_model.strip():
            raise ValueError('Batch size and model names must be nonempty and valid.')
