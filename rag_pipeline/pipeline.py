"""Explicit orchestration keeps every RAG stage easy to explain."""
import time
from .documents import load_documents, split_documents
from .providers import validate_vectors


def ingest(settings, provider, store, report=print):
    started = time.perf_counter()
    settings.validate()
    documents = load_documents(settings.data_dir)
    report(f'[1/4 LOAD] {len(documents)} documents from {settings.data_dir}')
    chunks = split_documents(documents, settings.chunk_size, settings.chunk_overlap)
    report(f'[2/4 SPLIT] {len(chunks)} chunks; size={settings.chunk_size}, overlap={settings.chunk_overlap} characters')
    vectors = []
    dimensions = None
    for start in range(0, len(chunks), settings.batch_size):
        batch = chunks[start:start+settings.batch_size]
        embeddings = provider.embed([c.page_content for c in batch], settings.embedding_model)
        dimensions = validate_vectors(embeddings, len(batch), dimensions)
        vectors.extend(embeddings)
        report(f'[3/4 EMBED] {len(vectors)}/{len(chunks)} with {settings.embedding_model} ({dimensions} dimensions)')
    manifest = store.replace(chunks, vectors, settings.embedding_model,
                             settings.chunk_size, settings.chunk_overlap, settings.data_dir)
    report(f"[4/4 STORE] {manifest['chunk_count']} Qdrant points; cosine distance; {time.perf_counter()-started:.2f}s")
    return manifest


def retrieve(question, provider, store, top_k=3, min_score=0.0, source=None):
    if not question.strip():
        raise ValueError('Question cannot be empty.')
    if top_k < 1 or not -1 <= min_score <= 1:
        raise ValueError('Require top-k >= 1 and min-score between -1 and 1.')
    manifest = store.manifest()
    # Always use the index's embedding model, not a possibly changed environment value.
    vector = provider.embed([question], manifest['embedding_model'])[0]
    return store.search(vector, top_k, min_score, source)
