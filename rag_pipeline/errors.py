"""Consistent user-facing errors without SDK response bodies or credentials."""
from openai import APIConnectionError, APIStatusError
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse


def error_message(exc):
    """Return a friendly message for known failures, or None for programming errors."""
    if isinstance(exc, ResponseHandlingException):
        return 'Cannot communicate with Qdrant. Check QDRANT_URL and start the server (docker compose up -d).'
    if isinstance(exc, UnexpectedResponse):
        return f'Qdrant HTTP {exc.status_code}. Check the server, QDRANT_URL, and QDRANT_API_KEY.'
    if isinstance(exc, APIConnectionError):
        return 'Cannot connect to OpenAI. Check your network and Python TLS certificates.'
    if isinstance(exc, APIStatusError):
        hints = {401: 'Check OPENAI_API_KEY.', 403: 'Check model permissions.',
                 404: 'Check the configured model name and account access.',
                 429: 'Check API billing/quota or retry later.'}
        return f'OpenAI HTTP {exc.status_code}. {hints.get(exc.status_code, "Retry later.")}'
    if isinstance(exc, (ValueError, RuntimeError, OSError)):
        return str(exc)
    return None
