"""Readiness and overload checks without models, remote APIs, or a real watchdog."""

import asyncio
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import anyio.to_thread
import httpx

from rag import rag_api


def empty_result(query, _medium=None):
    return rag_api.RagSearchResponse(query=query, top_results=[], highlight=None)


class ReadinessTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        ready = patch.object(rag_api, "_startup_complete", True)
        ready.start()
        self.addCleanup(ready.stop)

    async def test_probe_never_initializes_the_search_service(self):
        with patch.object(rag_api, "_service", None), patch.object(rag_api, "_startup_error", None), patch.object(rag_api, "get_service", side_effect=AssertionError("Health must not initialize models")):
            self.assertEqual(await rag_api.health(), {"status": "starting", "initialized": "false"})

    async def test_probe_reports_startup_failure_without_retrying_models(self):
        with patch.object(rag_api, "_startup_error", "Fixture startup failure"):
            result = await rag_api.health()
        self.assertEqual(result["status"], "error")
        self.assertNotIn("initialized", result)

    async def test_search_during_failed_startup_does_not_initialize_another_model(self):
        with patch.object(rag_api, "_service", None), patch.object(rag_api, "_startup_error", "Fixture startup failure"), patch.object(rag_api, "get_service", side_effect=AssertionError("Request must not reinitialize models")):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=rag_api.app), base_url="http://fixture") as client:
                self.assertEqual((await client.post("/rag/search", json={"query": "fixture"})).status_code, 503)

    async def test_slow_workers_do_not_trigger_watchdog_recovery(self):
        active = {object(): time.monotonic() - 70 for _ in range(2)}
        with patch.object(rag_api, "_service", object()), patch.object(rag_api, "_startup_error", None), patch.object(rag_api, "_search_capacity", 2), patch.object(rag_api, "_active_searches", active):
            self.assertEqual((await rag_api.health())["initialized"], "true")

    async def test_watchdog_can_recover_when_every_worker_is_genuinely_stalled(self):
        active = {object(): time.monotonic() - 121 for _ in range(2)}
        with patch.object(rag_api, "_service", object()), patch.object(rag_api, "_startup_error", None), patch.object(rag_api, "_search_capacity", 2), patch.object(rag_api, "_active_searches", active), patch.dict("os.environ", {"RAG_STALLED_SEARCH_SECONDS": "120"}):
            self.assertEqual((await rag_api.health())["status"], "error")
            active.pop(next(iter(active)))
            self.assertEqual((await rag_api.health())["initialized"], "true")

    async def test_probe_stays_responsive_when_search_worker_pool_is_saturated(self):
        entered, release = threading.Event(), threading.Event()

        def search(query, medium):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("Test did not release search")
            return empty_result(query, medium)

        limiter = anyio.to_thread.current_default_thread_limiter()
        original_tokens = limiter.total_tokens
        limiter.total_tokens = 1
        try:
            with patch.object(rag_api, "_service", SimpleNamespace(search=search)), patch.object(rag_api, "_startup_error", None), patch.object(rag_api, "_search_slots", threading.BoundedSemaphore(2)):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=rag_api.app), base_url="http://fixture") as client:
                    pending = asyncio.create_task(client.post("/rag/search", json={"query": "slow fixture"}))
                    try:
                        self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                        response = await asyncio.wait_for(client.get("/health"), timeout=.5)
                        self.assertEqual(response.json()["initialized"], "true")
                        self.assertFalse(pending.done())
                    finally:
                        release.set()
                        await pending
        finally:
            limiter.total_tokens = original_tokens

    async def test_overload_is_bounded_and_does_not_block_health(self):
        entered = threading.Barrier(3)
        release = threading.Event()

        def search(query, medium):
            entered.wait(3)
            if not release.wait(5):
                raise TimeoutError("Test did not release searches")
            return empty_result(query, medium)

        with patch.object(rag_api, "_service", SimpleNamespace(search=search)), patch.object(rag_api, "_startup_error", None), patch.object(rag_api, "_search_slots", threading.BoundedSemaphore(2)):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=rag_api.app), base_url="http://fixture") as client:
                pending = [asyncio.create_task(client.post("/rag/search", json={"query": f"slow fixture {index}"})) for index in range(2)]
                try:
                    await asyncio.to_thread(entered.wait, 2)
                    busy = await asyncio.wait_for(client.post("/rag/search", json={"query": "excess request"}), timeout=.5)
                    self.assertEqual(busy.status_code, 503)
                    self.assertEqual(busy.headers["retry-after"], "2")
                    self.assertEqual((await client.get("/health")).json()["initialized"], "true")
                finally:
                    release.set()
                    self.assertTrue(all(response.status_code == 200 for response in await asyncio.gather(*pending)))

    async def test_failed_search_releases_its_capacity_for_the_next_user(self):
        def failure(*_):
            raise RuntimeError("Fixture search failure")

        service = SimpleNamespace(search=failure)
        with patch.object(rag_api, "_service", service), patch.object(rag_api, "_search_slots", threading.BoundedSemaphore(1)):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=rag_api.app), base_url="http://fixture") as client:
                self.assertEqual((await client.post("/rag/search", json={"query": "failed fixture"})).status_code, 500)
                service.search = empty_result
                self.assertEqual((await client.post("/rag/search", json={"query": "next fixture"})).status_code, 200)


if __name__ == "__main__":
    unittest.main()
