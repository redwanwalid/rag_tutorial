"""Use real local Qdrant and real splitting; replace only paid OpenAI calls."""
import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import httpx
from openai import OpenAI
from langchain_core.documents import Document
from rag_pipeline.config import Settings
from rag_pipeline.documents import load_documents, split_documents
from rag_pipeline.pipeline import ingest, retrieve
from rag_pipeline.providers import OpenAIProvider, validate_vectors
from rag_pipeline.vector_store import VectorStore
from rag_pipeline.commands.evaluate import evaluate
from rag_pipeline.commands import ingest as create_database
from rag_pipeline.commands import query as query_data


class FixtureProvider:
    """Known vectors for predictable offline assertions, not semantic embeddings."""
    def embed(self, texts, model):
        return [[1.0, 0.0, 0.0] if 'rabbit' in text.lower() else [0.0, 1.0, 0.0] for text in texts]

    def answer(self, *args):
        raise AssertionError('Unexpected generation call')

    def close(self):
        pass


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / 'books'
        self.data.mkdir()
        (self.data / 'rabbit.md').write_text('The rabbit carries a pocket watch.', encoding='utf-8')
        (self.data / 'tea.txt').write_text('The Hatter drinks tea at the table.', encoding='utf-8')
        self.settings = Settings(data_dir=self.data, store_dir=self.root / 'store', embedding_model='fixture-model')

    def tearDown(self):
        self.temp.cleanup()

    def test_document_loading_recursive_and_empty(self):
        sub = self.data / 'nested'
        sub.mkdir()
        (sub / 'extra.MD').write_text('A cat.', encoding='utf-8')
        (self.data / 'ignore.csv').write_text('ignored', encoding='utf-8')
        docs = load_documents(self.data)
        self.assertEqual(len(docs), 3)
        self.assertIn('nested/extra.MD', [d.metadata['source'] for d in docs])
        with self.assertRaises(ValueError):
            load_documents(self.root / 'missing')

    def test_recursive_chunks_have_stable_ids_and_correct_offsets(self):
        text = 'First paragraph.\n\nSecond paragraph.\n\n' * 30
        doc = Document(page_content=text, metadata={'source': 'book.md'})
        chunks = split_documents([doc], 100, 20)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            start = c.metadata['start_index']
            self.assertGreaterEqual(start, 0)
            self.assertEqual(text[start:start+len(c.page_content)], c.page_content)
            self.assertLessEqual(len(c.page_content), 100)
        self.assertEqual([c.metadata['chunk_id'] for c in chunks],
                         [c.metadata['chunk_id'] for c in split_documents([doc], 100, 20)])
        short = split_documents([Document(page_content='Hi', metadata={'source': 'short.md'})])
        self.assertEqual(len(short), 1)

    def test_configuration_validation(self):
        for size, overlap in ((0, 0), (100, 100), (100, -1), (4001, 0)):
            with self.assertRaises(ValueError):
                replace(self.settings, chunk_size=size, chunk_overlap=overlap).validate()

    def test_real_qdrant_persistence_filtering_and_threshold(self):
        with VectorStore(self.settings.store_dir) as store:
            ingest(self.settings, FixtureProvider(), store, report=lambda _: None)
            self.assertEqual(store.stats()['stored_points'], 2)
        # A different client reopens vectors from disk.
        with VectorStore(self.settings.store_dir) as store:
            hits = retrieve('rabbit?', FixtureProvider(), store, top_k=1)
            self.assertEqual(hits[0]['source'], 'rabbit.md')
            self.assertAlmostEqual(hits[0]['score'], 1.0)
            self.assertEqual(store.search([1.0, 0.0, 0.0], source='tea.txt')[0]['source'], 'tea.txt')
            self.assertEqual(store.search([1.0, 0.0, 0.0], source='tea.txt', min_score=0.8), [])
            self.assertEqual(store.search([1.0, 0.0, 0.0], source='missing.md'), [])

    def test_rebuild_does_not_duplicate_and_removes_deleted_sources(self):
        with VectorStore(self.settings.store_dir) as store:
            ingest(self.settings, FixtureProvider(), store, report=lambda _: None)
            (self.data / 'tea.txt').unlink()
            ingest(self.settings, FixtureProvider(), store, report=lambda _: None)
            self.assertEqual(store.stats()['stored_points'], 1)
            self.assertEqual(store.search([0.0, 1.0, 0.0], source='tea.txt'), [])

    def test_failed_embedding_keeps_previous_collection(self):
        with VectorStore(self.settings.store_dir) as store:
            ingest(self.settings, FixtureProvider(), store, report=lambda _: None)
            before = store.manifest()
            provider = Mock()
            provider.embed.side_effect = RuntimeError('API unavailable')
            with self.assertRaises(RuntimeError):
                ingest(self.settings, provider, store, report=lambda _: None)
            self.assertEqual(store.manifest(), before)
            self.assertEqual(store.stats()['stored_points'], 2)

    def test_failed_publication_preserves_active_index(self):
        with VectorStore(self.settings.store_dir) as store:
            ingest(self.settings, FixtureProvider(), store, report=lambda _: None)
            before = store.manifest()
            collections_before = len(store.client.get_collections().collections)
            with patch('rag_pipeline.vector_store.os.replace', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    ingest(self.settings, FixtureProvider(), store, report=lambda _: None)
            self.assertEqual(store.manifest(), before)
            self.assertEqual(len(store.client.get_collections().collections), collections_before)

    def test_model_is_taken_from_index_and_dimension_is_checked(self):
        with VectorStore(self.settings.store_dir) as store:
            ingest(self.settings, FixtureProvider(), store, report=lambda _: None)
            provider = Mock(wraps=FixtureProvider())
            retrieve('rabbit?', provider, store)
            provider.embed.assert_called_once_with(['rabbit?'], 'fixture-model')
            with self.assertRaises(ValueError):
                store.search([1.0, 0.0])
            with self.assertRaises(ValueError):
                retrieve('   ', provider, store)

    def test_missing_index_and_empty_documents(self):
        with VectorStore(self.settings.store_dir) as store:
            with self.assertRaises(ValueError):
                store.manifest()
            for file in self.data.iterdir():
                file.write_text('   ', encoding='utf-8')
            with self.assertRaises(ValueError):
                ingest(self.settings, FixtureProvider(), store, report=lambda _: None)

    def test_invalid_embedding_batches(self):
        for vectors, count in (([], 1), ([[1, 0]], 2), ([[0, 0]], 1),
                               ([[float('nan'), 0]], 1), ([[1, 0], [1]], 2)):
            with self.assertRaises(ValueError):
                validate_vectors(vectors, count)

    def test_preview_needs_neither_key_nor_database(self):
        with patch('sys.argv', ['create_database.py', '--preview']), \
             patch.object(create_database.Settings, 'from_env', return_value=self.settings), \
             patch.object(create_database, 'OpenAIProvider') as provider, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            create_database.main()
        provider.assert_not_called()
        self.assertFalse(self.settings.store_dir.exists())
        self.assertIn('SPLIT: 2 chunks', output.getvalue())

    def test_search_only_and_no_hits_skip_generation(self):
        with VectorStore(self.settings.store_dir) as store:
            ingest(self.settings, FixtureProvider(), store, report=lambda _: None)
        provider = Mock(wraps=FixtureProvider())
        for options in (['--search-only'], ['--source', 'missing.md']):
            with patch('sys.argv', ['query_data.py', 'rabbit?', *options]), \
                 patch.object(query_data.Settings, 'from_env', return_value=self.settings), \
                 patch.object(query_data, 'OpenAIProvider', return_value=provider), \
                 contextlib.redirect_stdout(io.StringIO()):
                query_data.main()
        provider.answer.assert_not_called()

    def test_retrieval_metrics(self):
        cases = [{'question': 'watch?', 'evidence': 'pocket watch'},
                 {'question': 'unknown?', 'evidence': 'spaceship'}]
        with contextlib.redirect_stdout(io.StringIO()):
            scores = evaluate(cases, lambda q, k: [{'text': 'Tea.'}, {'text': 'A pocket watch.'}], 2)
        self.assertEqual(scores['Hit@2'], 0.5)
        self.assertEqual(scores['MRR@2'], 0.25)


class SDKTests(unittest.TestCase):
    def test_cli_forwards_command_help(self):
        from rag_pipeline.__main__ import main
        with contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as result:
                main(['query', '--help'])
        self.assertEqual(result.exception.code, 0)
        self.assertIn('--stats', output.getvalue())

    def test_real_sdk_embedding_and_generation_request_shapes(self):
        requests = []
        def handler(request):
            body = json.loads(request.content)
            requests.append((request.url.path, body))
            if request.url.path.endswith('/embeddings'):
                return httpx.Response(200, json={'object': 'list', 'model': 'test',
                    'data': [{'index': 1, 'object': 'embedding', 'embedding': [0.0, 1.0]},
                             {'index': 0, 'object': 'embedding', 'embedding': [1.0, 0.0]}],
                    'usage': {'prompt_tokens': 2, 'total_tokens': 2}})
            return httpx.Response(200, json={'id': 'test', 'object': 'chat.completion', 'created': 0,
                'model': 'test', 'choices': [{'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': 'A watch [1].'}}]})
        with httpx.Client(transport=httpx.MockTransport(handler)) as http:
            provider = OpenAIProvider(OpenAI(api_key='offline-test-key', http_client=http))
            self.assertEqual(provider.embed(['a', 'b'], 'test'), [[1.0, 0.0], [0.0, 1.0]])
            hits = [dict(source='rabbit.md', start_index=0, text='A pocket watch.')]
            self.assertEqual(provider.answer('What?', hits, 'test'), 'A watch [1].')
        self.assertEqual(requests[0][1]['encoding_format'], 'float')
        self.assertIn('[1] Source: rabbit.md', requests[1][1]['messages'][1]['content'])
        self.assertIn('untrusted', requests[1][1]['messages'][0]['content'])


if __name__ == '__main__':
    unittest.main()
