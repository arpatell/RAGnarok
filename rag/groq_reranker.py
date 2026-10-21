"""Small HTTP client with a bounded FIFO and organization-wide rate-limit pacing."""
from __future__ import annotations

import json
import math
import logging
import os
import random
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from types import SimpleNamespace

import httpx


class RerankBusy(RuntimeError):
    pass


def duration(value: str) -> float:
    try:
        return max(0.0, float(value))
    except ValueError:
        parts = re.findall(r"([\d.]+)(ms|s|m|h)", value)
        if parts:
            return sum(float(n) * {"ms": .001, "s": 1, "m": 60, "h": 3600}[unit] for n, unit in parts)
        try:
            return max(0.0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
        except (ValueError, TypeError):
            return 0.0


@dataclass
class Reservation:
    at: float
    tokens: int


class GroqReranker:
    def __init__(self, api_key: str, *, client=None, clock=time.monotonic):
        self.client = client or httpx.Client(base_url="https://api.groq.com/openai/v1/", headers={"Authorization": f"Bearer {api_key}"}, timeout=8)
        self.model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        self.clock = clock
        self.condition = threading.Condition()
        self.queue = deque()
        self.usage = deque()
        self.next_at = 0.0
        self.blocked_until = 0.0
        self.rpm = max(1, int(os.getenv("GROQ_REQUESTS_PER_MINUTE", "30")))
        self.tpm = max(1000, int(os.getenv("GROQ_TOKENS_PER_MINUTE", "8000")))
        self.max_queue = max(1, int(os.getenv("GROQ_MAX_QUEUE", "12")))
        self.max_wait = max(0.1, float(os.getenv("GROQ_MAX_WAIT_SECONDS", "18")))
        self.max_retries = max(0, min(3, int(os.getenv("GROQ_MAX_RETRIES", "2"))))
        self.notice = threading.local()

    def _wait_time(self, estimate):
        now = self.clock()
        while self.usage and self.usage[0].at <= now - 60:
            self.usage.popleft()
        wait = max(self.next_at, self.blocked_until) - now
        if sum(r.tokens for r in self.usage) + estimate > self.tpm and self.usage:
            wait = max(wait, self.usage[0].at + 60 - now)
        return max(0.0, wait)

    def _headers(self, response, estimate):
        headers = response.headers
        now = self.clock()
        try:
            # Groq's request limit header is DAILY, not RPM.
            limit = int(headers.get("x-ratelimit-limit-tokens", self.tpm))
            if limit > 0:
                self.tpm = limit
            if int(headers.get("x-ratelimit-remaining-requests", "1")) <= 0:
                self.blocked_until = max(self.blocked_until, now + duration(headers.get("x-ratelimit-reset-requests", "60")))
            if int(headers.get("x-ratelimit-remaining-tokens", str(self.tpm))) < estimate:
                self.blocked_until = max(self.blocked_until, now + duration(headers.get("x-ratelimit-reset-tokens", "1")))
        except (ValueError, TypeError):
            pass
        if response.status_code == 429:
            self.blocked_until = max(self.blocked_until, now + max(1.0, duration(headers.get("retry-after", "1"))))
        if response.status_code in {401, 403}:
            self.blocked_until = max(self.blocked_until, now + 60)

    def invoke(self, messages):
        self.notice.value = None
        # Budget the full request, not only the prose. This is deliberately
        # conservative; actual usage replaces the reservation after completion.
        mapped = [{"role": "user" if role == "human" else role, "content": content} for role, content in messages]
        prompt_bytes = len(json.dumps(mapped, ensure_ascii=False).encode("utf-8"))
        estimate = math.ceil(prompt_bytes / 3) + 768
        if prompt_bytes > 18000 or estimate > self.tpm:
            raise RerankBusy("Search context exceeds the current service allowance")
        deadline = self.clock() + self.max_wait
        ticket = object()
        with self.condition:
            if len(self.queue) >= self.max_queue:
                raise RerankBusy("Search queue is full")
            self.queue.append(ticket)
        try:
            for attempt in range(self.max_retries + 1):
                with self.condition:
                    while True:
                        remaining = deadline - self.clock()
                        wait = self._wait_time(estimate)
                        if remaining <= 0 or wait > remaining:
                            raise RerankBusy("Search service is busy; using available matches")
                        if self.queue[0] is ticket and wait <= 0:
                            reservation = Reservation(self.clock(), estimate)
                            self.usage.append(reservation)
                            self.next_at = self.clock() + 60 / self.rpm
                            break
                        self.condition.wait(min(remaining, max(.05, wait) if self.queue[0] is ticket else .25))
                try:
                    response = self.client.post("chat/completions", json={
                        "model": self.model, "messages": mapped, "temperature": 0,
                        "reasoning_effort": "low", "max_completion_tokens": 768,
                        "response_format": {"type": "json_object"},
                    }, timeout=min(8.0, max(.1, deadline - self.clock())))
                except httpx.TransportError:
                    if attempt == self.max_retries:
                        raise
                    with self.condition:
                        self.blocked_until = max(self.blocked_until, self.clock() + 2 ** attempt + random.uniform(.05, .25))
                    continue
                with self.condition:
                    self._headers(response, estimate)
                    if response.is_success:
                        payload = response.json()
                        reservation.tokens = int(payload.get("usage", {}).get("total_tokens", estimate))
                    elif response.status_code == 429:
                        reservation.tokens = 0
                    elif response.status_code >= 500:
                        self.blocked_until = max(self.blocked_until, self.clock() + max(duration(response.headers.get("retry-after", "0")), 2 ** attempt + random.uniform(.05, .25)))
                if response.is_success:
                    choice = payload["choices"][0]
                    content = choice["message"].get("content")
                    logging.getLogger(__name__).info("Groq rerank model=%s tokens=%d finish=%s", self.model, reservation.tokens, choice.get("finish_reason", "unknown"))
                    if not content:
                        raise RerankBusy("Search model did not finish within its output budget")
                    return SimpleNamespace(content=content)
                if response.status_code != 429 and response.status_code < 500:
                    raise RuntimeError(f"Groq request rejected ({response.status_code})")
                if attempt == self.max_retries:
                    raise RerankBusy(f"Search service unavailable ({response.status_code})")
            raise RerankBusy("Search service unavailable")
        finally:
            with self.condition:
                self.queue.remove(ticket)
                self.condition.notify_all()
