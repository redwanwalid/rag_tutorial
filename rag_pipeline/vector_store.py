"""Persist and search vectors using embedded Qdrant or a Qdrant server."""
from hashlib import sha256
import json
import os
from contextlib import suppress
from pathlib import Path
from uuid import uuid4
from qdrant_client import QdrantClient, models
from .providers import validate_vectors
from .config import Settings


class VectorStore:
    def __init__(self, folder: Path, *, settings=None):
        self.settings = settings or Settings(store_dir=Path(folder))
        self.settings.validate()
        self.mode = self.settings.vector_mode
        self.folder = Path(folder)
        if self.mode == 'server':
            # Each endpoint has its own active collection; preserve the legacy local path.
            endpoint = self.settings.qdrant_url.rstrip('/')
            self.folder /= 'servers/' + sha256(endpoint.encode()).hexdigest()[:16]
        self.folder.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.folder / 'active_index.json'
        if self.mode == 'local':
            self.client = QdrantClient(path=str(self.folder / 'qdrant'))
        else:
            self.client = QdrantClient(url=self.settings.qdrant_url,
                api_key=self.settings.qdrant_api_key or None, timeout=60)

    @classmethod
    def from_settings(cls, settings):
        return cls(settings.store_dir, settings=settings)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.client.close()

    def manifest(self):
        if not self.manifest_path.is_file():
            raise ValueError(f'No Qdrant index for {self.mode} mode yet. Run python -m rag_pipeline ingest with the same configuration first.')
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding='utf-8'))
            for key in ('collection', 'embedding_model', 'dimensions', 'chunk_count'):
                if key not in manifest:
                    raise ValueError('Incomplete index metadata.')
            if not self.client.collection_exists(manifest['collection']):
                raise ValueError('Active Qdrant collection is missing.')
            return manifest
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError('Index metadata is unreadable. Rebuild the index.') from exc

    def replace(self, chunks, vectors, model, chunk_size, overlap, data_dir):
        dimensions = validate_vectors(vectors, len(chunks))
        if len({c.metadata['chunk_id'] for c in chunks}) != len(chunks):
            raise ValueError('Duplicate chunk identifiers found.')
        collection = 'books_' + uuid4().hex
        manifest = dict(collection=collection, embedding_model=model, dimensions=dimensions,
                        chunk_count=len(chunks), chunk_size=chunk_size, chunk_overlap=overlap,
                        data_dir=str(data_dir), distance='Cosine', schema_version=1)
        temporary = self.folder / ('manifest_' + uuid4().hex + '.tmp')
        try:
            options = {}
            if self.mode == 'server':
                options = dict(
                    hnsw_config=models.HnswConfigDiff(m=self.settings.hnsw_m,
                        ef_construct=self.settings.hnsw_ef_construct,
                        full_scan_threshold=self.settings.full_scan_threshold),
                    optimizers_config=models.OptimizersConfigDiff(
                        indexing_threshold=self.settings.indexing_threshold))
            self.client.create_collection(collection_name=collection,
                vectors_config=models.VectorParams(size=dimensions, distance=models.Distance.COSINE),
                **options)
            for start in range(0, len(chunks), 64):
                points = [models.PointStruct(
                    id=c.metadata['chunk_id'], vector=v,
                    payload={**c.metadata, 'text': c.page_content},
                ) for c, v in zip(chunks[start:start+64], vectors[start:start+64])]
                self.client.upsert(collection_name=collection, points=points, wait=True)
            actual_count = self.client.count(collection_name=collection, exact=True).count
            if actual_count != len(chunks):
                raise ValueError('Stored point count does not match the document chunks.')
            temporary.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            os.replace(temporary, self.manifest_path)
        except Exception:
            # A failed upload/publication never switches the active index.
            # A disconnected server can reject cleanup too; retain the original error.
            with suppress(Exception):
                if self.client.collection_exists(collection):
                    self.client.delete_collection(collection)
            raise
        finally:
            temporary.unlink(missing_ok=True)
        # Previous collections remain available for inspection and rollback.
        return manifest

    def search(self, vector, top_k=3, min_score=0.0, source=None):
        if top_k < 1 or not -1 <= min_score <= 1:
            raise ValueError('Require top-k >= 1 and a cosine cutoff between -1 and 1.')
        manifest = self.manifest()
        validate_vectors([vector], 1, manifest['dimensions'])
        query_filter = None
        if source:
            query_filter = models.Filter(must=[models.FieldCondition(
                key='source', match=models.MatchValue(value=source))])
        points = self.client.query_points(
            collection_name=manifest['collection'], query=vector,
            limit=top_k, score_threshold=min_score, query_filter=query_filter,
            with_payload=True, with_vectors=False,
            **({'search_params': models.SearchParams(hnsw_ef=self.settings.hnsw_ef, exact=False)}
               if self.mode == 'server' else {}),
        ).points
        return [dict(p.payload, score=p.score) for p in points]

    def stats(self):
        manifest = self.manifest()
        result = {**manifest, 'vector_mode': self.mode, 'manifest_path': str(self.manifest_path),
            'stored_points': self.client.count(
            collection_name=manifest['collection'], exact=True).count}
        if self.mode == 'server':
            info = self.client.get_collection(manifest['collection'])
            result.update(indexed_vectors_count=info.indexed_vectors_count,
                          status=info.status, optimizer_status=info.optimizer_status)
        return result
