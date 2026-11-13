"""Load, split, embed, and persist documents in the configured Qdrant backend."""
import argparse
from dataclasses import replace
from pathlib import Path
from rag_pipeline.config import Settings
from rag_pipeline.documents import load_documents, split_documents
from rag_pipeline.pipeline import ingest
from rag_pipeline.providers import OpenAIProvider
from rag_pipeline.vector_store import VectorStore
from rag_pipeline.cli import run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, help='Folder containing .md or .txt files.')
    parser.add_argument('--chunk-size', type=int)
    parser.add_argument('--overlap', type=int)
    parser.add_argument('--preview', action='store_true', help='Show chunking without API calls or database writes.')
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    settings = replace(settings,
        data_dir=args.data.expanduser().resolve() if args.data else settings.data_dir,
        chunk_size=args.chunk_size if args.chunk_size is not None else settings.chunk_size,
        chunk_overlap=args.overlap if args.overlap is not None else settings.chunk_overlap)
    settings.validate()
    if args.preview:
        documents = load_documents(settings.data_dir)
        chunks = split_documents(documents, settings.chunk_size, settings.chunk_overlap)
        print(f'LOAD: {len(documents)} documents\nSPLIT: {len(chunks)} chunks')
        for chunk in chunks[:3]:
            print(f"\n{chunk.metadata['source']} @ character {chunk.metadata['start_index']} | {len(chunk.page_content)} characters\n{chunk.page_content}")
        return
    provider = OpenAIProvider()
    try:
        with VectorStore.from_settings(settings) as store:
            ingest(settings, provider, store)
    finally:
        provider.close()


if __name__ == '__main__':
    run(main)
