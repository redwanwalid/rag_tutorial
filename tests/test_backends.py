"""Backend selection and server contracts without Docker or paid API calls."""
import contextlib
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from rag_pipeline.config import Settings
from rag_pipeline.vector_store import VectorStore
from rag_pipeline.commands import ingest, query, evaluate


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = Settings(store_dir=self.root, vector_mode='server')
        self.client = Mock()
        self.client.count.return_value.count = 1
        self.client.get_collection.return_value = Mock(
            indexed_vectors_count=1, status='green', optimizer_status='ok')
        self.client.query_points.return_value.points = [Mock(
            payload={'source': 'book.md', 'text': 'A rabbit.'}, score=0.9)]
        self.chunk = Document(page_content='A rabbit.', metadata={
            'source': 'book.md', 'start_index': 0,
            'chunk_id': 'ef6b1e8c-194c-4fb2-804f-a64b3cc97198'})

    def build(self, store):
        return store.replace([self.chunk], [[1., 0.]], 'test-model', 1000, 200, self.root)

    def test_server_build_search_stats_and_manifest_isolation(self):
        settings = replace(self.settings, qdrant_api_key='private-test-key',
            hnsw_m=24, hnsw_ef_construct=120, hnsw_ef=80,
            indexing_threshold=1, full_scan_threshold=10)
        legacy = self.root / 'active_index.json'
        legacy.write_text('local index stays untouched')
        with patch('rag_pipeline.vector_store.QdrantClient', return_value=self.client) as factory:
            with VectorStore.from_settings(settings) as store:
                manifest = self.build(store)
                factory.assert_called_once_with(url=settings.qdrant_url,
                    api_key='private-test-key', timeout=60)
                options = self.client.create_collection.call_args.kwargs
                self.assertEqual(options['hnsw_config'].m, 24)
                self.assertEqual(options['hnsw_config'].ef_construct, 120)
                self.assertEqual(options['hnsw_config'].full_scan_threshold, 10)
                self.assertEqual(options['optimizers_config'].indexing_threshold, 1)
                point = self.client.upsert.call_args.kwargs['points'][0]
                self.assertEqual(point.payload['text'], 'A rabbit.')
                self.assertEqual(point.vector, [1., 0.])
                self.assertEqual(store.search([1., 0.], source='book.md')[0]['score'], 0.9)
                search = self.client.query_points.call_args.kwargs
                self.assertEqual(search['search_params'].hnsw_ef, 80)
                self.assertFalse(search['search_params'].exact)
                self.assertEqual(search['query_filter'].must[0].match.value, 'book.md')
                stats = store.stats()
                self.assertEqual(stats['vector_mode'], 'server')
                self.assertEqual(stats['indexed_vectors_count'], 1)
                self.assertEqual(stats['stored_points'], 1)
                self.assertNotIn('private-test-key', store.manifest_path.read_text())
                self.assertNotEqual(store.manifest_path, legacy)
                self.assertEqual(store.manifest()['collection'], manifest['collection'])
            self.client.close.assert_called_once()
            other = VectorStore.from_settings(replace(settings, qdrant_url='http://localhost:6335'))
            self.addCleanup(other.client.close)
            self.assertNotEqual(other.manifest_path, store.manifest_path)
            with self.assertRaisesRegex(ValueError, 'No Qdrant index for server'):
                other.manifest()
        self.assertEqual(legacy.read_text(), 'local index stays untouched')

    def test_server_upload_failure_preserves_previous_manifest(self):
        with patch('rag_pipeline.vector_store.QdrantClient', return_value=self.client):
            with VectorStore.from_settings(self.settings) as store:
                self.build(store)
                before = store.manifest_path.read_text()
                self.client.upsert.side_effect = RuntimeError('upload failed')
                with self.assertRaisesRegex(RuntimeError, 'upload failed'):
                    self.build(store)
                self.assertEqual(store.manifest_path.read_text(), before)
                self.client.delete_collection.assert_called_once()

    def test_local_uses_legacy_path_and_no_server_options(self):
        with patch('rag_pipeline.vector_store.QdrantClient', return_value=self.client) as factory:
            with VectorStore(self.root) as store:
                factory.assert_called_once_with(path=str(self.root / 'qdrant'))
                self.assertEqual(store.manifest_path, self.root / 'active_index.json')
                self.build(store)
                self.assertNotIn('hnsw_config', self.client.create_collection.call_args.kwargs)
                store.search([1., 0.])
                self.assertNotIn('search_params', self.client.query_points.call_args.kwargs)

    def test_settings_from_environment_and_validation(self):
        values = {'RAG_VECTOR_MODE': 'server', 'QDRANT_URL': 'http://localhost:6335/',
            'QDRANT_API_KEY': 'secret', 'RAG_HNSW_M': '24', 'RAG_HNSW_EF': '80',
            'RAG_HNSW_EF_CONSTRUCT': '120', 'RAG_INDEXING_THRESHOLD_KB': '1',
            'RAG_FULL_SCAN_THRESHOLD_KB': '10'}
        with patch.dict(os.environ, values, clear=True), patch('rag_pipeline.config.load_dotenv'):
            settings = Settings.from_env()
        self.assertEqual(settings.qdrant_url, 'http://localhost:6335')
        self.assertEqual(settings.hnsw_m, 24)
        self.assertEqual(settings.hnsw_ef_construct, 120)
        self.assertEqual(settings.hnsw_ef, 80)
        self.assertEqual(settings.indexing_threshold, 1)
        self.assertEqual(settings.full_scan_threshold, 10)
        self.assertNotIn('secret', repr(settings))
        for changes in ({'vector_mode': 'bad'}, {'qdrant_url': 'not-a-url'},
                        {'qdrant_url': 'http://user:secret@localhost:6333'},
                        {'hnsw_m': 0}, {'hnsw_ef': 0}, {'hnsw_ef_construct': -1},
                        {'hnsw_ef_construct': 3}, {'indexing_threshold': 0},
                        {'full_scan_threshold': 1}, {'full_scan_threshold': 9}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.settings, **changes).validate()

    def test_all_database_commands_forward_settings_and_preview_is_offline(self):
        cases = self.root / 'cases.json'
        cases.write_text(json.dumps([{'question': 'rabbit?', 'evidence': 'rabbit'}]))
        for command, args in ((ingest, []), (query, ['rabbit?', '--search-only']),
                              (query, ['--stats']), (evaluate, ['--cases', str(cases)])):
            with self.subTest(command=command.__name__, args=args), \
                 patch.object(command.Settings, 'from_env', return_value=self.settings), \
                 patch.object(command, 'VectorStore') as factory, \
                 patch.object(command, 'OpenAIProvider'), \
                 contextlib.redirect_stdout(io.StringIO()):
                store = factory.from_settings.return_value.__enter__.return_value
                store.manifest.return_value = {'embedding_model': 'test-model'}
                store.stats.return_value = {}
                with patch.object(command, 'ingest' if command is ingest else 'retrieve') as action:
                    action.return_value = [{'text': 'rabbit', 'source': 'book.md',
                                            'start_index': 0, 'score': 0.9}]
                    command.main(args)
                factory.from_settings.assert_called_once_with(self.settings)
        (self.root / 'sample.txt').write_text('A rabbit.')
        with patch.object(ingest.Settings, 'from_env', return_value=replace(
                self.settings, data_dir=self.root)), \
             patch.object(ingest, 'VectorStore') as factory, \
             patch.object(ingest, 'OpenAIProvider') as provider, \
             contextlib.redirect_stdout(io.StringIO()):
            ingest.main(['--preview'])
        factory.from_settings.assert_not_called()
        provider.assert_not_called()

    def test_server_requests_through_real_qdrant_sdk(self):
        import httpx
        from qdrant_client import QdrantClient

        requests = []

        def handler(request):
            body = json.loads(request.content) if request.content else None
            requests.append((request.method, request.url.path, body))
            if request.url.path.endswith('/exists'):
                result = {'exists': True}
            elif request.url.path.endswith('/points/count'):
                result = {'count': 1}
            elif request.url.path.endswith('/points/query'):
                result = {'points': [{'id': self.chunk.metadata['chunk_id'],
                    'version': 0, 'score': 0.9, 'payload': {
                        **self.chunk.metadata, 'text': self.chunk.page_content}}]}
            elif request.url.path.endswith('/points'):
                result = {'operation_id': 1, 'status': 'completed'}
            else:
                result = True
            return httpx.Response(200, json={'result': result, 'status': 'ok', 'time': 0.001})

        client = QdrantClient(url=self.settings.qdrant_url,
            transport=httpx.MockTransport(handler), check_compatibility=False)
        with patch('rag_pipeline.vector_store.QdrantClient', return_value=client):
            with VectorStore.from_settings(self.settings) as store:
                self.build(store)
                hits = store.search([1., 0.], top_k=1, min_score=0.5, source='book.md')
                self.assertEqual(hits[0]['text'], 'A rabbit.')
        create = next(body for method, path, body in requests
                      if method == 'PUT' and '/points' not in path)
        self.assertEqual(create['vectors']['size'], 2)
        self.assertEqual(create['hnsw_config']['m'], 16)
        search = next(body for _, path, body in requests if path.endswith('/points/query'))
        self.assertEqual(search['query'], {'nearest': [1., 0.]})
        self.assertEqual(search['params'], {'hnsw_ef': 64, 'exact': False})
        self.assertEqual(search['limit'], 1)
        self.assertEqual(search['score_threshold'], 0.5)
        self.assertEqual(search['filter']['must'][0]['match']['value'], 'book.md')

    def test_server_cli_errors_do_not_dump_response_or_credentials(self):
        import httpx
        from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
        from rag_pipeline.cli import run

        errors = [ResponseHandlingException(RuntimeError('secret')),
                  UnexpectedResponse(401, 'Unauthorized', b'secret', httpx.Headers())]
        for error in errors:
            with self.subTest(error=type(error).__name__), \
                 contextlib.redirect_stderr(io.StringIO()) as output, \
                 self.assertRaises(SystemExit) as result:
                run(Mock(side_effect=error))
            self.assertEqual(result.exception.code, 1)
            self.assertIn('Qdrant', output.getvalue())
            self.assertNotIn('secret', output.getvalue())
