"""Process-wide UI locking, terminal logging, and action error display."""
from contextlib import contextmanager
import logging
import sys
import threading
import time

import streamlit as st

from .errors import error_message


@st.cache_resource
def runtime():
    # Cache only a lock and logger, never an open embedded Qdrant connection.
    logger = logging.getLogger('rag_demo')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
        logger.addHandler(handler)
    return threading.Lock(), logger


@contextmanager
def action(name):
    lock, log = runtime()
    started = time.perf_counter()
    log.info('%s START', name)
    try:
        with lock:
            yield
    except Exception as exc:
        message = error_message(exc) or f'Unexpected {type(exc).__name__}; check configuration and dependencies.'
        log.error('%s FAILED | %s', name, message)
        st.error(message)
    else:
        log.info('%s DONE | %.2fs', name, time.perf_counter() - started)
