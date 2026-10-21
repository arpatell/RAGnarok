from __future__ import annotations

import gzip
import json
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import httpx

from rag.catalog_policy import excluded_genres, extract_section
from rag.catalog_sync import CharacterParser, compose, open_db, store
from rag.eval.catalog_eval import generated_cases, human_cases
from rag.eval.golden_top4000_pipeline import GoldenCase, evaluate_case
from rag.rag_api import HybridRagService, RagResult, RagSearchResponse, _is_character_query, _query_tokens, _character_query_tokens, _load_title_index, _title_lookup_keys


class PolicyTests(unittest.TestCase):
    def test_all_explicit_exclusions_and_variants(self):
        for genre in ["Boys Love", "Girls Love", "Erotica", "Hentai", "Boys' Love", "Yaoi", "Yuri", "Shounen Ai"]:
            with self.subTest(genre=genre):
                self.assertTrue(excluded_genres({"genres": [{"name": genre}]}))
                self.assertTrue(excluded_genres({"chunk_text": f"Type: manga Genres: Action, {genre} Synopsis: example"}))

    def test_mentions_in_synopsis_are_not_genre_tags(self):
        self.assertEqual(excluded_genres({"text": "Genres: Drama\nSynopsis: She draws boys love manga.\nCharacters: Unknown"}), [])
        self.assertEqual(extract_section("Genres: Drama\nSynopsis: Some story\nCharacters: Someone", "Genres"), "Drama")

    def test_character_parser_ignores_sidebar_recommendations(self):
        parser = CharacterParser()
        parser.feed('<a href="https://myanimelist.net/character/1/Unrelated">Popular sidebar person</a><a href="https://myanimelist.net/character/2/Person"><h3 class="h3_character_name">Family, Given</h3></a>')
        self.assertEqual(parser.names, ["Family, Given"])

    def test_composition_preserves_roster_and_adds_search_details(self):
        row = {"type": "manga", "title": "Fixture", "genres": ["Mystery"], "synopsis": "A mystery.", "search_details": {"Authors": "A Creator", "Started": "2001"}}
        text = compose(row, "Characters: Person\nCharacter Search Aliases: Person Alias")
        self.assertIn("Authors: A Creator", text)
        self.assertEqual(extract_section(text, "Characters"), "Person")
        self.assertEqual(extract_section(text, "Character Search Aliases"), "Person Alias")

    def test_checkpoint_update_is_idempotent_and_preserves_order(self):
        with tempfile.TemporaryDirectory() as directory:
            with closing(open_db(Path(directory) / "catalog.sqlite")) as db:
                row = {"type": "anime", "mal_id": 123, "title": "Before"}
                store(db, row)
                before = db.execute("SELECT rowid FROM documents").fetchone()[0]
                store(db, row | {"title": "After"}, enriched=True)
                self.assertEqual(db.execute("SELECT COUNT(*) FROM documents").fetchone()[0], 1)
                self.assertEqual(db.execute("SELECT rowid FROM documents").fetchone()[0], before)


class TitleTests(unittest.TestCase):
    def test_character_names_keep_internal_particles_without_matching_only_one_of_multiple_names(self):
        from rag.rag_api import _character_alias_query
        self.assertEqual(_character_alias_query("series with character Keeper of the Gate", {"keeper of the gate"}), "keeper of the gate")
        self.assertEqual(_character_alias_query("Guts Griffith Casca", {"guts", "griffith", "casca"}), "guts griffith casca")
    def test_disk_snippets_preserve_all_text_and_medium_identity(self):
        from rag.evidence_index import build, EvidenceIndex
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "catalog.jsonl.gz"
            raw = "Type: manga\nTitle: Fixture\nSynopsis: Original synopsis.\nCharacters: Given Family\nCharacter Search Aliases: Given Family"
            with gzip.open(corpus, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps({"type":"manga","mal_id":7,"title":"Fixture","text":raw}))
            path = Path(directory) / "evidence.sqlite"
            build(corpus, path)
            index = EvidenceIndex(path)
            try:
                self.assertTrue(index.has_serving_snippets)
                self.assertEqual(index.serving_snippet("manga",7), " ".join(raw.split()))
                self.assertEqual(index.serving_snippet("anime",7), "")
                titles = _load_title_index(str(corpus), str(corpus.with_suffix('.missing')), retain_snippets=False)
                self.assertEqual(titles[("manga",7)].snippet, "")
            finally:
                index.db.close()

    def test_large_name_vocabulary_does_not_turn_plot_descriptions_into_people(self):
        self.assertFalse(_is_character_query("red haired pirate", ["red", "haired", "pirate"], {"red person", "haired person", "pirate captain"}, {"red", "haired", "pirate", "person", "captain"}))
        self.assertTrue(_is_character_query("family given", ["family", "given"], {"family given"}, {"family", "given"}))

    def test_natural_character_questions_keep_the_name_not_the_scaffolding(self):
        question = "What's the series with a character named Light Yagami?"
        tokens = _character_query_tokens(question)
        self.assertEqual(tokens, ["light", "yagami"])
        self.assertTrue(_is_character_query(question, tokens, {"light yagami"}, {"light", "yagami"}))
        plot = "I remember a story where someone finds a notebook"
        self.assertFalse(_is_character_query(plot, _query_tokens(plot), {"light yagami"}, {"light", "yagami"}))

    def test_small_typos_not_related_titles(self):
        for query, alias in [("firefly weding", "firefly wedding"), ("one peice", "one piece"), ("fullmetal alchmist", "fullmetal alchemist"), ("attack on titen", "attack on titan"), ("chainsaw mn", "chainsaw man")]:
            with self.subTest(query=query):
                self.assertGreaterEqual(HybridRagService._title_similarity(query, alias), .935)
        for query, alias in [("solo leveling", "solo leveling ragnarok"), ("anime 1", "anime 2"), ("food", "fool")]:
            with self.subTest(query=query):
                self.assertLess(HybridRagService._title_similarity(query, alias), .935)

    def test_romanization_spacing_and_particles(self):
        self.assertTrue(_title_lookup_keys("Sin-ui Tap") & _title_lookup_keys("Sinui tap"))
        self.assertTrue(_title_lookup_keys("Yuryeong-ui Norae") & _title_lookup_keys("Yuryeongeui norae"))
        self.assertTrue(_title_lookup_keys("Kyoujin") & _title_lookup_keys("Kyojin"))

    def test_deployable_snapshot_retains_identity_aliases_and_covers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl.gz"
            rows = [
                {"type": "manga", "mal_id": 7, "title": "Fixture", "text": "Genres: Drama", "alternative_titles": {"en": "English Fixture"}, "romanized_titles": ["Romanized Fixture"], "image_url": "https://example.test/manga.jpg"},
                {"type": "anime", "mal_id": 7, "title": "Different Work", "text": "Genres: Action"},
                {"type": "manga", "mal_id": 8, "title": "Blocked", "genres": ["Hentai"]},
            ]
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write("\n".join(json.dumps(r) for r in rows))
            index = _load_title_index(str(path), str(path.with_suffix(".missing")))
            self.assertEqual(set(index), {("manga", 7), ("anime", 7)})
            self.assertIn("English Fixture", index[("manga", 7)].aliases)
            self.assertIn("Romanized Fixture", index[("manga", 7)].aliases)
            self.assertEqual(index[("manga", 7)].image_url, "https://example.test/manga.jpg")


class EvaluationTests(unittest.TestCase):
    def test_recognition_can_recover_a_verified_title_from_one_or_no_local_candidates(self):
        service = HybridRagService.__new__(HybridRagService)
        service._llm_rerank_enabled, service._llm_backoff_until = True, 0
        service._alias_phrase_set, service._alias_token_set = set(), set()
        service._title_index = {}
        service._llm = SimpleNamespace(invoke=lambda _: SimpleNamespace(content='{"expanded_title":"Known Fixture","ordered_candidate_ids":[],"reasoning":"Matches the compass clue."}'))
        known = RagResult(rank=1, mal_id=7, title="Known Fixture", snippet="Type: manga Synopsis: A traveler finds a mysterious compass.", score=1, citation="Fixture", read_now_title="Known Fixture", media_type="manga", title_candidates=["Known Fixture"])
        service.direct_title_retrieve = lambda title, medium: [known] if title == "Known Fixture" and medium == "manga" else []
        for candidates in ([], [RagResult(rank=1, mal_id=8, title="Other", snippet="Synopsis: unrelated", score=1, citation="Fixture", read_now_title="Other", media_type="manga")]):
            ranked, _ = service.llm_rank_candidates("guy finds compass", candidates, "manga")
            self.assertEqual(ranked[0].mal_id, 7)
    def test_model_failure_retains_unfused_retrieval_order(self):
        service = HybridRagService.__new__(HybridRagService)
        service._llm_rerank_enabled, service._llm_backoff_until = True, 0
        service._alias_phrase_set, service._alias_token_set = set(), set()
        service._llm = SimpleNamespace(invoke=lambda *_: (_ for _ in ()).throw(RuntimeError("provider unavailable")))
        rows = [RagResult(rank=i, mal_id=i, title=f"Fixture {i}", snippet="Synopsis: unrelated", score=1, citation="Fixture", read_now_title="Fixture") for i in (1,2)]
        rows[0]._fusion_score, rows[1]._fusion_score = .01, .03
        ranked, reasoning = service.llm_rank_candidates("a traveler searches for a lost city", rows)
        self.assertEqual(ranked[0].mal_id, 1)
        self.assertIsNone(reasoning)
    def test_rank_fusion_preserves_identity_and_rewards_independent_evidence(self):
        remote = [{"media_type":"manga","mal_id":1,"score":99}, {"media_type":"manga","mal_id":2,"score":1}]
        local = [{"media_type":"manga","mal_id":2,"score":.01}, {"media_type":"anime","mal_id":1,"score":99}]
        combined = HybridRagService._fuse_candidates(remote, local)
        self.assertEqual(combined[0]["mal_id"], 2)
        self.assertEqual(len(combined), 3)
        self.assertEqual(remote[0]["score"], 99)
    def test_short_memories_recall_without_candidate_anchoring_and_validate_clues(self):
        service = HybridRagService.__new__(HybridRagService)
        service._llm_rerank_enabled = True
        service._llm_backoff_until = 0
        service._alias_phrase_set, service._alias_token_set = set(), set()
        service.config = SimpleNamespace(top_k=10)
        service._title_index = {}
        prompts = []
        def invoke(messages):
            prompts.extend(messages)
            return SimpleNamespace(content='{"expanded_title":"Known Fixture","ordered_candidate_ids":[],"reasoning":"Matched the clue."}')
        service._llm = SimpleNamespace(invoke=invoke)
        recalled = RagResult(rank=1, mal_id=42, title="Known Fixture", snippet="Synopsis: A boy discovers a mysterious compass.", score=1, citation="Fixture", read_now_title="Known Fixture", media_type="manga", title_candidates=["Known Fixture"])
        service.direct_title_retrieve = lambda *_: [recalled]
        candidates = [RagResult(rank=i, mal_id=i, title=f"Obscure Candidate {i}", snippet="Synopsis: unrelated", score=.5, citation="Fixture", read_now_title="Fixture", media_type="manga") for i in (1,2)]
        ranked, _ = service.llm_rank_candidates("guy finds compass", candidates)
        self.assertEqual(ranked[0].mal_id, 42)
        self.assertNotIn("Obscure Candidate", prompts[1][1])
        recalled.snippet = "Synopsis: A completely unrelated school comedy."
        ranked, reasoning = service.llm_rank_candidates("guy finds compass", candidates)
        self.assertEqual(ranked[0].mal_id, 1)
        self.assertIsNone(reasoning)

    def test_optional_reranker_receives_plot_not_only_long_metadata(self):
        service = HybridRagService.__new__(HybridRagService)
        service._llm_rerank_enabled = True
        service._llm_backoff_until = 0
        service._alias_phrase_set, service._alias_token_set = set(), set()
        service.config = SimpleNamespace(top_k=10)
        prompts = []
        def invoke(messages):
            prompts.extend(messages)
            return SimpleNamespace(content='{"ordered_candidate_ids":[2,1],"reasoning":"Matched the plot."}')
        service._llm = SimpleNamespace(invoke=invoke)
        snippet = "Type: manga\nDetails: Authors: " + "A Creator " * 80 + "\nGenres: Mystery\nSynopsis: A traveler discovers that the missing city moves every night.\nCharacters: Unknown"
        candidates = [RagResult(rank=i, mal_id=i, title=f"Fixture {i}", snippet=snippet, score=1, citation="Fixture", read_now_title="Fixture") for i in (1, 2)]
        ranked, _ = service.llm_rank_candidates("moving city", candidates)
        self.assertEqual(ranked[0].mal_id, 2)
        self.assertIn("the missing city moves every night", prompts[1][1])
        self.assertIn("A Creator", prompts[1][1])

    def test_wrong_medium_sequel_and_shared_title_do_not_count(self):
        case = GoldenCase("fixture", "manga", "title", "Fixture", "Fixture", 7)
        for result in [{"title": "Fixture", "mal_id": 7, "media_type": "anime"}, {"title": "Fixture Season 2", "mal_id": 8, "media_type": "manga"}, {"title": "Fixture", "mal_id": 9, "media_type": "manga"}]:
            with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"top_results": [result]}))) as client:
                self.assertFalse(evaluate_case(client, "https://example.test", case, 10)["hit"])

    def test_negative_policy_case_requires_absence_not_a_substitute(self):
        case = GoldenCase("excluded", "manga", "title", "Excluded", "Excluded", 7, True)
        for present in [True, False]:
            results = [{"title": "Excluded", "mal_id": 7, "media_type": "manga"}] if present else []
            with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"top_results": results}))) as client:
                self.assertEqual(evaluate_case(client, "https://example.test", case, 10)["hit"], not present)

    def test_generated_selection_has_ninety_percent_distinct_title_coverage(self):
        rows = [{"type": "manga", "mal_id": i, "title": f"Title {i}", "text": "", "synopsis": "A person searches for a vanished friend in a distant city."} for i in range(100)]
        cases = generated_cases(rows, .9, {"title", "semantic"})
        self.assertEqual(len({c["target_mal_id"] for c in cases}), 90)
        self.assertEqual({c["kind"] for c in cases}, {"title", "semantic"})
        self.assertEqual(cases, generated_cases(rows, .9, {"title", "semantic"}))

    def test_explicit_fixture_id_beats_same_title_oneshot(self):
        rows = [{"type": "manga", "mal_id": 1, "title": "Shared", "display_type": "one_shot"}, {"type": "manga", "mal_id": 9, "title": "Shared", "display_type": "manga"}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixtures.json"
            path.write_text(json.dumps({"cases": [{"target": "Shared", "target_mal_id": 9, "medium": "manga", "kind": "recollection", "query": "A detailed memory"}]}))
            cases, missing = human_cases(rows, path)
            self.assertEqual(cases[0]["target_mal_id"], 9)
            self.assertEqual(missing, [])


class CacheTests(unittest.TestCase):
    def fallback_service(self):
        service = HybridRagService.__new__(HybridRagService)
        service.config = SimpleNamespace(top_k=10, candidate_pool_size=30, pinecone_namespaces=("fixture",), hybrid_dense_alpha=.35)
        service._namespace_backoff, service._local_only = {}, False
        service._llm_rerank_enabled, service._llm, service._llm_backoff_until = False, None, 0
        service._alias_phrase_set, service._alias_token_set = set(), set()
        service._title_index = {}
        service._rerank_status = SimpleNamespace(notice=None)
        service._evidence_index = SimpleNamespace(search=lambda *_args, **_kwargs: [])
        service._embeddings = SimpleNamespace(embed_query=lambda _: [])
        service._bm25 = SimpleNamespace(encode_queries=lambda _: {"indices": [], "values": []})
        return service

    def test_provider_quota_backoff_avoids_repeat_requests_and_embedding_work(self):
        service = self.fallback_service()
        calls, embeds = [], []
        service._embeddings.embed_query = lambda q: embeds.append(q) or []
        class Exhausted(RuntimeError):
            status = 429
        def unavailable(**kwargs):
            calls.append(kwargs)
            raise Exhausted("Monthly data allowance exhausted")
        service._run_namespace_query = unavailable
        self.assertEqual(service.hybrid_retrieve("a traveler finds a missing city", "manga"), [])
        self.assertIn("locally available", service._rerank_status.notice)
        self.assertEqual(service.hybrid_retrieve("another plot recollection", "manga"), [])
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(embeds), 1)

    def test_local_fallback_keeps_identity_and_supports_a_short_clue(self):
        service = self.fallback_service()
        service._local_only = True
        service._embeddings.embed_query = lambda _: self.fail("Local-only mode must not embed")
        service._run_namespace_query = lambda **_: self.fail("Local-only mode must not contact Pinecone")
        service._title_index[("manga",7)] = SimpleNamespace(mal_id=7, media_type="manga", title="Fixture", snippet="Type: manga Synopsis: A lost notebook.", aliases=("Fixture",), image_url=None)
        service._evidence_index.search = lambda *_args, **_kwargs: [{"media_type":"manga","mal_id":7,"score":1}]
        result = service.hybrid_retrieve("notebook", "manga")
        self.assertEqual(result[0].mal_id, 7)
        self.assertEqual(result[0].media_type, "manga")
        self.assertIn("temporarily limited", service._rerank_status.notice)

    def test_empty_limited_search_still_returns_the_provider_notice(self):
        service = self.fallback_service()
        service.direct_title_retrieve = service.character_retrieve = service.evidence_retrieve = lambda *_: []
        def limited(*_):
            service._rerank_status.notice = "Search is temporarily limited."
            return []
        service.hybrid_retrieve = limited
        result = service._search_uncached("unknown memory", "manga")
        self.assertEqual(result.top_results, [])
        self.assertEqual(result.notice, "Search is temporarily limited.")

    def test_concurrent_identical_requests_share_work_and_cache_isolated_copies(self):
        service = HybridRagService.__new__(HybridRagService)
        service._response_cache_ttl = 60
        service._response_cache_max = 2
        service._response_cache = {}
        service._response_cache_bytes = 0
        service._response_cache_budget = 4096
        service._inflight_searches = {}
        service._search_cache_lock = threading.Lock()
        calls = []
        def search(query, medium):
            calls.append((query, medium))
            time.sleep(.05)
            return RagSearchResponse(query=query, top_results=[], highlight=None)
        with patch.object(service, "_search_uncached", search):
            with ThreadPoolExecutor(max_workers=6) as pool:
                results = list(pool.map(lambda _: service.search("Fixture", "manga"), range(6)))
            self.assertEqual(len(calls), 1)
            results[0].query = "mutated"
            self.assertEqual(service.search("Fixture", "manga").query, "Fixture")
            service.search("Fixture", "anime")
            self.assertEqual(len(calls), 2)
            self.assertLessEqual(service._response_cache_bytes, service._response_cache_budget)


if __name__ == "__main__":
    unittest.main()
