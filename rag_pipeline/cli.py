"""Friendly CLI errors without dumping API keys, request bodies, or tracebacks."""
import sys
from .errors import error_message


def run(main):
    try:
        main()
    except Exception as exc:
        message = error_message(exc)
        if message is None:
            raise
        print(f'Error: {message}', file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        print('\nCancelled.', file=sys.stderr)
        raise SystemExit(130)
