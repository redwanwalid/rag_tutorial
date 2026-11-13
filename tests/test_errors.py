"""Shared UI/CLI errors preserve actionable hints without exposing SDK bodies."""
import contextlib
import io
import unittest
from unittest.mock import Mock

import httpx
from openai import APIStatusError

from rag_pipeline.cli import run
from rag_pipeline.errors import error_message


class ErrorTests(unittest.TestCase):
    def test_openai_errors_hide_response_content_and_keep_hints(self):
        for status, hint in ((401, 'OPENAI_API_KEY'), (404, 'model name'), (429, 'billing/quota')):
            with self.subTest(status=status):
                response = httpx.Response(status, request=httpx.Request('POST', 'https://example.test'))
                error = APIStatusError('private response', response=response, body={'secret': 'private'})
                message = error_message(error)
                self.assertIn(str(status), message)
                self.assertIn(hint, message)
                self.assertNotIn('private', message)
                with contextlib.redirect_stderr(io.StringIO()) as output:
                    with self.assertRaises(SystemExit) as result:
                        run(Mock(side_effect=error))
                self.assertEqual(result.exception.code, 1)
                self.assertEqual(output.getvalue().strip(), f'Error: {message}')

    def test_cli_does_not_hide_programming_errors(self):
        error = TypeError('bug')
        self.assertIsNone(error_message(error))
        with self.assertRaises(TypeError):
            run(Mock(side_effect=error))

    def test_cli_interrupt_exit_code_is_preserved(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                run(Mock(side_effect=KeyboardInterrupt))
        self.assertEqual(result.exception.code, 130)
