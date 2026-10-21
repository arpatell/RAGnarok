from concurrent.futures import ThreadPoolExecutor
import os
import importlib.util
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import httpx

from rag.character_data import ProfileParser, RosterParser
from rag.groq_reranker import GroqReranker, RerankBusy, duration
from rag.evidence_index import explicit_creator
from rag.rag_api import _relevant_excerpt, _extract_character_aliases


class CharacterTests(unittest.TestCase):
    def test_biography_words_never_become_character_name_aliases(self):
        aliases = _extract_character_aliases("Characters: Given Family\nCharacter Search Aliases: Given Family; Family Given\nCharacter Details: A bounty hunter travels across the galaxy.")
        self.assertEqual(aliases, {"given family", "family given"})
    def test_creator_identity_not_an_incidental_word_in_plot(self):
        self.assertTrue(explicit_creator("a fantasy manga by Kentaro Miura", "kentaro miura"))
        self.assertTrue(explicit_creator("Madhouse anime about a traveler", "madhose"))
        self.assertFalse(explicit_creator("a boy travels past the sunrise to find his friend", "sunrise"))
        self.assertFalse(explicit_creator("he gets a smile from Yamada at the supermarket", "yamada"))
        self.assertFalse(explicit_creator("Family, Given", "family"))
        self.assertTrue(explicit_creator("Naoki Urasawa science fiction mystery", "naoki urasawa"))
        self.assertFalse(explicit_creator("I'm trying to remember a story. A boy called Light visits a madhouse.", "madhose"))

    def test_query_aware_excerpt_finds_later_clues_without_full_biography(self):
        text = "An ordinary introduction. " + "Unrelated biographical details. " * 40 + "The traveler has a glass eye and works as a bounty hunter."
        excerpt = _relevant_excerpt(text, "bounty hunter with a glass eye", 160)
        self.assertIn("glass eye", excerpt)
        self.assertLessEqual(len(excerpt), 160)
    def test_roster_ids_roles_and_sidebar_exclusion(self):
        parser = RosterParser()
        parser.feed('<a href="https://myanimelist.net/character/99/Sidebar">Not a cast member</a><a href="/character/12/Example"><h3 class="h3_character_name">Family, Given</h3></a><div>Main&nbsp;</div><a href="/character/13/Other"><h3 class="h3_character_name">Other</h3></a><div>Supporting</div>')
        self.assertEqual(set(parser.records), {12, 13})
        self.assertEqual(parser.records[12]["role"], "Main")
        self.assertEqual(parser.records[13]["role"], "Supporting")

    def test_profile_keeps_biography_not_ads_actors_or_footer(self):
        parser = ProfileParser()
        parser.feed('<h2 class="normal_header">Given "Alias" Family</h2>Born on Mars.<br>A bounty hunter.<script>ad code</script><div class="normal_header">Voice Actors</div>Actor name<footer>Footer</footer>')
        result = parser.result()
        self.assertIn("Alias", result["aliases"])
        self.assertIn("bounty hunter", result["about"])
        self.assertNotIn("ad code", result["about"])
        self.assertNotIn("Actor name", result["about"])


class GroqTests(unittest.TestCase):
    def client(self, handler):
        return httpx.Client(base_url="https://example.test/", transport=httpx.MockTransport(handler))

    def test_duration_headers(self):
        self.assertEqual(duration("2m3.5s"), 123.5)
        self.assertEqual(duration("150ms"), .15)
        self.assertEqual(duration("invalid"), 0)

    def test_top_ten_prompt_and_actual_usage_daily_not_rpm(self):
        def handler(request):
            self.assertNotIn("secret", request.content.decode())
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"ordered_candidate_ids":[1]}'}}], "usage": {"total_tokens": 123}}, headers={"x-ratelimit-limit-requests": "1000", "x-ratelimit-limit-tokens": "8000"})
        model = GroqReranker("secret", client=self.client(handler))
        model.invoke([("system", "Rank candidates"), ("human", "Example")])
        self.assertEqual(model.usage[0].tokens, 123)
        self.assertEqual(model.rpm, 30)
        self.assertEqual(len(model.queue), 0)

    def test_retry_after_longer_than_budget_falls_back_without_retry_storm(self):
        calls = []
        model = GroqReranker("secret", client=self.client(lambda r: calls.append(r) or httpx.Response(429, headers={"retry-after": "60"})))
        model.max_wait = .1
        with self.assertRaises(RerankBusy):
            model.invoke([("human", "Example")])
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(model.queue), 0)
        with self.assertRaises(RerankBusy):
            model.invoke([("human", "Another user")])
        self.assertEqual(len(calls), 1)

    def test_auth_failures_are_not_retried_and_do_not_expose_key(self):
        calls = []
        model = GroqReranker("secret", client=self.client(lambda r: calls.append(r) or httpx.Response(401)))
        with self.assertRaisesRegex(RuntimeError, r"rejected \(401\)"):
            model.invoke([("human", "Example")])
        self.assertEqual(len(calls), 1)

    def test_queue_is_bounded_and_released_on_completion(self):
        entered, release = threading.Event(), threading.Event()
        def handler(request):
            entered.set()
            release.wait(2)
            return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})
        model = GroqReranker("secret", client=self.client(handler))
        model.max_queue = 1
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(model.invoke, [("human", "first")])
            self.assertTrue(entered.wait(1))
            try:
                with self.assertRaisesRegex(RerankBusy, "full"):
                    model.invoke([("human", "second")])
            finally:
                release.set()
            pending.result()
        self.assertEqual(len(model.queue), 0)

    def test_context_budget_rejects_oversize_before_http(self):
        model = GroqReranker("secret", client=self.client(lambda _: self.fail("Unexpected HTTP")))
        with self.assertRaises(RerankBusy):
            model.invoke([("human", "x" * 20000)])


class DeployTests(unittest.TestCase):
    def test_existing_env_updates_preserve_unrelated_values_and_are_idempotent(self):
        spec = importlib.util.spec_from_file_location("configure_groq", Path("scripts/gcp/configure-groq.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rag.env"
            path.write_text("# Existing configuration\nMAL_CLIENT_ID=fixture-mal\nGROQ_API_KEY=old-fixture\nRAG_LLM_RERANK_ENABLED=0\n")
            changes = {"GROQ_API_KEY": "new-fixture", "RAG_LLM_RERANK_ENABLED": "1"}
            module.update_env(path, changes)
            first = path.read_text()
            module.update_env(path, changes)
            self.assertEqual(path.read_text(), first)
            self.assertIn("MAL_CLIENT_ID=fixture-mal", first)
            self.assertIn("# Existing configuration", first)
            self.assertEqual(first.count("GROQ_API_KEY="), 1)


if __name__ == "__main__":
    unittest.main()
