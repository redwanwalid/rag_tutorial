"""Exercise real Streamlit interactions without paid API calls."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest

from rag_pipeline.config import ROOT, Settings


class AppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        data = root / 'docs'
        data.mkdir()
        (data / 'rabbit.md').write_text('The rabbit carries a pocket watch.')
        self.settings = Settings(data_dir=data, store_dir=root / 'store')
        self.config = patch('rag_pipeline.config.Settings.from_env', return_value=self.settings)
        self.config.start()
        self.addCleanup(self.config.stop)
        self.provider = Mock()
        self.provider.embed.side_effect = lambda texts, model: [[1., 0., 0.] for _ in texts]
        self.provider.answer.return_value = 'The rabbit carries a pocket watch [1].'
        self.factory = patch('rag_pipeline.ui.OpenAIProvider', return_value=self.provider)
        self.factory_mock = self.factory.start()
        self.addCleanup(self.factory.stop)
        self.app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=15).run()

    def click(self, label):
        next(button for button in self.app.button if button.label == label).click().run()
        self.assertFalse(self.app.exception)

    def test_chat_input_is_outside_tabs(self):
        self.assertEqual(len(self.app.chat_input), 1)
        for tab in self.app.tabs:
            self.assertEqual(len(tab.get('chat_input')), 0)

    def test_preview_and_missing_index_make_no_api_calls(self):
        self.assertFalse(self.app.exception)
        self.factory_mock.assert_not_called()
        self.click('Preview chunks (free)')
        self.assertIn('1 documents → 1 chunks', self.app.success[0].value)
        self.assertFalse(self.settings.store_dir.exists())
        self.click('Inspect index (free)')
        self.assertIn('No Qdrant index', self.app.error[0].value)
        self.factory_mock.assert_not_called()

    def test_small_talk_without_index_or_api_calls(self):
        with patch('rag_pipeline.ui.VectorStore') as store:
            for question in ('Hi!', 'How are you?', 'Thanks'):
                self.app.chat_input[0].set_value(question).run()
                self.assertFalse(self.app.exception)
                self.assertNotIn('hits', self.app.session_state.messages[-1])
            store.from_settings.assert_not_called()
        self.factory_mock.assert_not_called()
        self.assertFalse(self.settings.store_dir.exists())

    def test_build_query_retrieval_only_and_no_hits(self):
        self.click('Build / rebuild index')
        self.assertIn('Index ready: 1 chunks', self.app.success[0].value)
        self.click('Inspect index (free)')
        self.app.chat_input[0].set_value('What does the rabbit carry?').run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.provider.answer.call_count, 1)
        self.assertIn('pocket watch [1]', self.app.session_state.messages[-1]['content'])
        self.assertEqual(self.app.session_state.messages[-1]['hits'][0]['source'], 'rabbit.md')
        self.app.run()
        self.assertTrue(any('pocket watch [1]' in item.value for item in self.app.markdown))
        self.assertEqual(self.provider.answer.call_count, 1)
        self.app.checkbox[0].check().run()
        self.app.chat_input[0].set_value('watch?').run()
        self.assertIn('generation is disabled', self.app.session_state.messages[-1]['content'])
        self.assertEqual(self.provider.answer.call_count, 1)
        self.app.text_input[0].set_value('missing.md').run()
        self.app.chat_input[0].set_value('watch?').run()
        self.assertIn('No passages meet', self.app.session_state.messages[-1]['content'])
        self.assertEqual(self.provider.answer.call_count, 1)
        self.click('Clear conversation')
        self.assertEqual(self.app.session_state.messages, [])

    def test_greeting_with_document_question_still_uses_rag(self):
        self.click('Build / rebuild index')
        self.app.chat_input[0].set_value('Hi, what does the rabbit carry?').run()
        self.assertFalse(self.app.exception)
        self.provider.answer.assert_called_once()
        self.assertTrue(self.app.session_state.messages[-1]['hits'])

    def test_failed_generation_is_displayed_and_releases_store(self):
        self.click('Build / rebuild index')
        self.provider.answer.side_effect = RuntimeError('Generation unavailable')
        self.app.chat_input[0].set_value('watch?').run()
        self.assertFalse(self.app.exception)
        self.assertIn('Generation unavailable', self.app.error[0].value)
        self.provider.close.assert_called()
        self.click('Inspect index (free)')
        self.assertTrue(self.app.json)
