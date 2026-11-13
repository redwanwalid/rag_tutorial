"""Small talk must bypass paid calls without swallowing document questions."""
import contextlib
import io
import unittest
from unittest.mock import patch

from rag_pipeline.commands import query
from rag_pipeline.routing import small_talk_reply


class RoutingTests(unittest.TestCase):
    def test_social_messages(self):
        for message in ('Hi', ' HELLO! ', 'Hey there', 'Good morning', 'how are you?',
                        'Hi, how are you doing?', 'How’s it going?', 'Thank you!', 'Bye.'):
            with self.subTest(message=message):
                self.assertIsNotNone(small_talk_reply(message))

    def test_document_and_mixed_messages_fall_through(self):
        for message in ('', 'What does the Rabbit carry?', 'Hi, what does Alice drink?',
                        'Thanks, but who is the Hatter?', 'What does hello mean?',
                        'How are you described in the book?', 'hello alice',
                        'Who is the president?'):
            with self.subTest(message=message):
                self.assertIsNone(small_talk_reply(message))

    def test_cli_greeting_does_not_open_store_or_provider(self):
        with patch.object(query, 'Settings') as settings, \
             patch.object(query, 'VectorStore') as store, \
             patch.object(query, 'OpenAIProvider') as provider, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            query.main(['Hi'])
        settings.from_env.assert_not_called()
        store.from_settings.assert_not_called()
        provider.assert_not_called()
        self.assertIn('ROUTE SMALL TALK', output.getvalue())

    def test_explicit_retrieval_bypasses_small_talk_router(self):
        with patch.object(query, 'small_talk_reply') as router, \
             patch.object(query.Settings, 'from_env', side_effect=ValueError('reached RAG')):
            with self.assertRaisesRegex(ValueError, 'reached RAG'):
                query.main(['Hi', '--search-only'])
        router.assert_not_called()
