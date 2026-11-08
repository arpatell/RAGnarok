"""One monotonic allowance shared by optional stages of a search request."""

from contextlib import contextmanager
from contextvars import ContextVar
import time


_deadline = ContextVar("rag_request_deadline", default=None)


def remaining_seconds(maximum, *, reserve=0.0):
    deadline = _deadline.get()
    return maximum if deadline is None else max(0.0, min(maximum, deadline - time.monotonic() - reserve))


@contextmanager
def request_budget(seconds):
    deadline = time.monotonic() + max(0.0, seconds)
    previous = _deadline.get()
    token = _deadline.set(min(previous, deadline) if previous is not None else deadline)
    try:
        yield
    finally:
        _deadline.reset(token)
