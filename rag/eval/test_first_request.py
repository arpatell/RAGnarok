"""Uncached request, transport-fault, and startup contracts; no remote API traffic."""

import asyncio
import importlib.util
from concurrent.futures import ThreadPoolExecutor
import random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
from types import SimpleNamespace
from pathlib import Path
import unittest
from unittest.mock import patch

import httpx

from rag import rag_api
from rag.eval import test_search_contracts as contracts
from rag.groq_reranker import GroqReranker, RerankBusy
from rag.request_budget import remaining_seconds, request_budget


class BudgetTests(unittest.TestCase):
    def test_transient_first_connection_is_retried_inside_the_original_request(self):
        service = contracts.CacheTests().fallback_service()
        calls = []

        def query(**_kwargs):
            calls.append(True)
            if len(calls) == 1:
                raise TimeoutError("Stale first connection")
            return []

        service._run_namespace_query = query
        with request_budget(2):
            service.hybrid_retrieve("a traveler searches for a vanished city")
        self.assertEqual(len(calls), 2)
        self.assertEqual(service._namespace_backoff, {})
        self.assertIsNone(service._rerank_status.notice)

    def test_nested_budget_cannot_extend_the_original_deadline(self):
        now = [100.0]
        with patch("rag.request_budget.time.monotonic", side_effect=lambda: now[0]):
            with request_budget(5):
                now[0] += 3
                self.assertEqual(remaining_seconds(10), 2)
                with request_budget(20):
                    self.assertEqual(remaining_seconds(10), 2)
                now[0] += 3
                self.assertEqual(remaining_seconds(10), 0)
            self.assertEqual(remaining_seconds(10), 10)

    def test_pinecone_call_receives_the_remaining_allowance(self):
        service = rag_api.HybridRagService.__new__(rag_api.HybridRagService)
        calls = []
        service._index = SimpleNamespace(query=lambda **kwargs: calls.append(kwargs) or {"matches": []})
        with request_budget(2):
            service._run_namespace_query(namespace="fixture", dense_query=[], sparse_query={}, top_k=10)
        self.assertGreater(calls[0]["_request_timeout"], 0)
        self.assertLessEqual(calls[0]["_request_timeout"], 1)
        with request_budget(0):
            with self.assertRaises(TimeoutError):
                service._run_namespace_query(namespace="fixture", dense_query=[], sparse_query={}, top_k=10)
        self.assertEqual(len(calls), 1)

    def test_groq_does_not_start_a_new_budget_after_retrieval(self):
        calls = []
        client = httpx.Client(base_url="https://fixture/", transport=httpx.MockTransport(lambda request: calls.append(request) or httpx.Response(429, headers={"retry-after": "2"})))
        self.addCleanup(client.close)
        model = GroqReranker("fixture", client=client)
        started = time.monotonic()
        with request_budget(.4):
            time.sleep(.05)
            with self.assertRaises(RerankBusy):
                model.invoke([("human", "uncached memory")])
        self.assertEqual(len(calls), 1)
        self.assertLess(time.monotonic() - started, .4)

    def test_expired_budget_does_not_make_a_model_request(self):
        client = httpx.Client(base_url="https://fixture/", transport=httpx.MockTransport(lambda _: self.fail("Expired request called the provider")))
        self.addCleanup(client.close)
        model = GroqReranker("fixture", client=client)
        with request_budget(0), self.assertRaises(RerankBusy):
            model.invoke([("human", "fixture")])
        self.assertEqual(len(model.queue), 0)

    def test_network_phases_share_the_model_allowance(self):
        def handler(request):
            self.assertLessEqual(sum(request.extensions["timeout"].values()), .5)
            return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

        client = httpx.Client(base_url="https://fixture/", transport=httpx.MockTransport(handler))
        self.addCleanup(client.close)
        with request_budget(.75):
            GroqReranker("fixture", client=client).invoke([("human", "fixture")])


class BenchmarkReadinessTests(unittest.TestCase):
    def helper(self):
        spec = importlib.util.spec_from_file_location("budget_readiness_fixture", Path("scripts/bench/windows-budget.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_startup_error_is_not_counted_as_a_ready_benchmark(self):
        module = self.helper()
        response = SimpleNamespace(is_success=True, json=lambda: {"status": "error", "detail": "fixture failed warmup"})
        with patch.object(module.httpx, "get", return_value=response), self.assertRaisesRegex(RuntimeError, "failed warmup"):
            module.wait_ready("http://fixture/health", [SimpleNamespace(poll=lambda: None)])

    def test_benchmark_waits_for_initialized_true(self):
        module = self.helper()
        responses = [SimpleNamespace(is_success=True, json=lambda: {"initialized": "false"}), SimpleNamespace(is_success=True, json=lambda: {"initialized": "true"})]
        with patch.object(module.httpx, "get", side_effect=responses) as requests, patch.object(module.time, "sleep"):
            module.wait_ready("http://fixture/health", [SimpleNamespace(poll=lambda: None)])
        self.assertEqual(requests.call_count, 2)


class TransportTests(unittest.TestCase):
    def test_pinned_sdk_stalled_socket_is_bounded_without_hidden_retries(self):
        from pinecone import Index
        from pinecone.config.openapi import OpenApiConfigFactory

        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                requests.append(self.path)
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                time.sleep(.2)
                try:
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'{"matches":[]}')
                except ConnectionError:
                    pass

            def log_message(self, *_):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=.01), daemon=True)
        thread.start()
        transport = OpenApiConfigFactory.build(api_key="fixture")
        transport.retries = 0
        try:
            with Index(api_key="fixture", host=f"http://127.0.0.1:{server.server_port}", openapi_config=transport) as index:
                started = time.monotonic()
                with self.assertRaises(Exception):
                    index.query(vector=[.1], top_k=1, _request_timeout=.05)
                self.assertLess(time.monotonic() - started, .5)
                self.assertEqual(len(requests), 1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(1)


class TitlePruningTests(unittest.TestCase):
    def test_upper_bound_never_changes_an_accepted_title_score(self):
        rng = random.Random(673)
        vocabulary = ["alpha", "girl", "time", "travel", "hunter", "quiet", "wedding", "no", "norae", "yuryeong", "hero", "knight", "family", "dream", "2"]
        pairs = [("firefly weding", "firefly wedding"), ("one peice", "one piece"), ("yuryeongeui norae", "yuryeong ui norae"), ("the girl", "the girl who crossed the sea")]
        for _ in range(2000):
            left = " ".join(rng.choices(vocabulary, k=rng.randint(1, 7)))
            right = left if rng.random() < .15 else " ".join(rng.choices(vocabulary, k=rng.randint(1, 7)))
            pairs.append((left, right))
        for left, right in pairs:
            full = rag_api.HybridRagService._title_similarity(left, right)
            for threshold in (.80, .88, .935, .99):
                pruned = rag_api.HybridRagService._title_similarity(left, right, minimum_score=threshold)
                if full >= threshold:
                    self.assertEqual(pruned, full, (left, right, threshold))
                else:
                    self.assertLess(pruned, threshold, (left, right, threshold))


class StartupTests(unittest.TestCase):
    def test_readiness_waits_for_real_embedding_and_sparse_warmup(self):
        entered, release = threading.Event(), threading.Event()
        calls = []

        def embed(_query):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("Test did not release warmup")
            calls.append("embedding")
            return []

        service = SimpleNamespace(_local_only=False, _embeddings=SimpleNamespace(embed_query=embed), _bm25=SimpleNamespace(encode_queries=lambda _: calls.append("sparse")))
        with patch.object(rag_api, "_service", service), patch.object(rag_api, "_startup_error", None), patch.object(rag_api, "_startup_complete", False), patch.object(rag_api, "get_service", return_value=service):
            with ThreadPoolExecutor(max_workers=1) as pool:
                pending = pool.submit(rag_api.warm_start_service)
                try:
                    self.assertTrue(entered.wait(2))
                    self.assertEqual(asyncio.run(rag_api.health())["initialized"], "false")
                finally:
                    release.set()
                    pending.result()
            self.assertEqual(asyncio.run(rag_api.health())["initialized"], "true")
        self.assertEqual(calls, ["embedding", "sparse"])

    def test_local_only_mode_does_not_warm_an_unused_model(self):
        service = SimpleNamespace(_local_only=True)
        with patch.object(rag_api, "_service", service), patch.object(rag_api, "_startup_error", None), patch.object(rag_api, "_startup_complete", False), patch.object(rag_api, "get_service", return_value=service):
            rag_api.warm_start_service()
            self.assertEqual(asyncio.run(rag_api.health())["initialized"], "true")


class FirstAttemptTests(unittest.IsolatedAsyncioTestCase):
    async def test_fresh_queries_return_matches_on_the_first_attempt_during_provider_throttling(self):
        calls = []
        client = httpx.Client(base_url="https://fixture/", transport=httpx.MockTransport(lambda request: calls.append(request) or httpx.Response(429, headers={"retry-after": "2"})))
        self.addCleanup(client.close)
        service = rag_api.HybridRagService.__new__(rag_api.HybridRagService)
        service.config = SimpleNamespace(top_k=10)
        service._llm_rerank_enabled, service._llm_backoff_until = True, 0
        service._llm = GroqReranker("fixture", client=client)
        service._alias_phrase_set, service._alias_token_set = set(), set()
        service._rerank_status = threading.local()
        service.direct_title_retrieve = service.character_retrieve = service.evidence_retrieve = lambda *_: []
        row = rag_api.RagResult(rank=1, mal_id=7, title="Verified Fixture", snippet="Type: manga\nSynopsis: A traveler discovers a city that moves at night.", score=1, citation="Fixture", read_now_title="Verified Fixture", media_type="manga")

        def uncached_retrieve(*_):
            time.sleep(.05)
            return [row]

        service.hybrid_retrieve = uncached_retrieve
        service.normalize_ranked_results = lambda _query, rows: rows
        # Deliberately bypass both response caches: every request takes the full
        # production search/rerank path, rather than retrying a completed query.
        service.search = service._search_uncached
        with patch.object(rag_api, "_service", service), patch.object(rag_api, "_startup_complete", True), patch.object(rag_api, "_startup_error", None):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=rag_api.app), base_url="http://fixture") as api:
                for query in ["a traveler visits a vanished city", "someone searching for a city that moves"]:
                    started = time.monotonic()
                    response = await api.post("/rag/search", json={"query": query, "budget_ms": 400})
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json()["top_results"][0]["mal_id"], 7)
                    self.assertTrue(response.json()["notice"])
                    self.assertLess(time.monotonic() - started, .4)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
