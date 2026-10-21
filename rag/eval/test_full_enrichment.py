"""Complete-import contracts; isolated databases and no external requests."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from rag.catalog_sync import MalClient, compose, open_db, store
from rag.character_data import ProfileParser, update_title
from rag.full_enrichment import (
    DETAIL_FIELDS, apply_details, collect_profile, collect_roster, coverage,
    run_batch, seed_profiles, setup, verify_page_identity,
)


class FullEnrichmentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.db = open_db(Path(self.directory.name) / "catalog.sqlite")
        self.addCleanup(self.db.close)
        self.row = {"type": "manga", "mal_id": 7, "title": "Example", "genres": ["Adventure"], "catalog_rank": 1}
        self.row["text"] = compose(self.row)
        store(self.db, self.row, upserted=True)
        self.db.commit()
        setup(self.db)

    def client(self, body, status=200):
        class Client:
            def get(inner, path, **kwargs):
                response = httpx.Response(status, text=body, request=httpx.Request("GET", path))
                response.raise_for_status()
                return response
        return Client()

    def test_requests_all_public_detail_fields_not_user_account_data(self):
        for medium, fields in DETAIL_FIELDS.items():
            self.assertIn("background", fields)
            self.assertIn("pictures", fields)
            self.assertIn("recommendations", fields)
            self.assertIn("related_manga", fields)
            self.assertNotIn("my_list_status", fields)
        self.assertIn("serialization", DETAIL_FIELDS["manga"])
        self.assertIn("statistics", DETAIL_FIELDS["anime"])

    def test_setup_resumes_checkpoints_without_resetting_success(self):
        self.db.execute("UPDATE enrichment_tasks SET status='done' WHERE kind='details'")
        self.db.commit()
        setup(self.db)
        self.assertEqual(coverage(self.db)["stages"]["details"], {"done": 1})

    def test_manga_without_list_rank_does_not_wait_behind_all_anime(self):
        for identifier in range(1, 4):
            store(self.db, {"type": "anime", "mal_id": identifier, "catalog_rank": identifier, "title": "Anime"})
        self.row.pop("catalog_rank")
        store(self.db, self.row)
        self.db.commit()
        setup(self.db)
        self.assertEqual(self.db.execute("SELECT priority FROM enrichment_tasks WHERE kind='details' AND key='manga:7'").fetchone()[0], 1)

    def test_redirected_wrong_character_is_rejected(self):
        response = httpx.Response(200, request=httpx.Request("GET", "https://myanimelist.net/character/8/Wrong"))
        with self.assertRaises(ValueError):
            verify_page_identity(response, "character", 7)

    def test_mal_retry_after_is_not_shortened_to_sixty_seconds(self):
        responses = iter([httpx.Response(429, headers={"Retry-After": "120"}), httpx.Response(200, json={})])
        transport = httpx.MockTransport(lambda _: next(responses))
        with patch.dict("os.environ", {"MAL_CLIENT_ID": "fixture"}):
            client = MalClient(interval=1.1)
        client.api.close()
        client.api = httpx.Client(base_url="https://example.test/", transport=transport)
        try:
            with patch("rag.catalog_sync.time.sleep") as sleep:
                self.assertEqual(client.get("manga/7").status_code, 200)
                self.assertIn(((120.0,), {}), sleep.call_args_list)
        finally:
            client.api.close()
            client.web.close()

    def test_metadata_preserves_entire_payload_and_searchable_details(self):
        node = {"id": 7, "title": "Example", "genres": [{"name": "Adventure"}],
                "background": "Published internationally.", "num_chapters": 37,
                "serialization": [{"node": {"name": "Example Magazine"}}],
                "related_manga": [{"node": {"id": 8, "title": "Not an alias"}}],
                "statistics": {"status": {"reading": "12"}}}
        with self.db:
            apply_details(self.db, "manga:7", node)
        self.assertEqual(json.loads(self.db.execute("SELECT payload_json FROM mal_details").fetchone()[0]), node)
        row, indexed = self.db.execute("SELECT row_json,upserted FROM documents").fetchone()
        row = json.loads(row)
        self.assertIn("Example Magazine", row["text"])
        self.assertIn("Chapters: 37", row["text"])
        self.assertNotIn("Not an alias", row["alternative_titles"].values())
        self.assertEqual(indexed, 0)

    def test_identity_mismatch_does_not_overwrite_title(self):
        with self.assertRaises(ValueError), self.db:
            apply_details(self.db, "manga:7", {"id": 8, "title": "Wrong work"})
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM mal_details").fetchone()[0], 0)

    def test_supporting_profiles_are_queued_once_for_shared_characters(self):
        self.db.execute("INSERT INTO title_characters VALUES('manga:7',99,'A Person','Supporting')")
        seed_profiles(self.db)
        seed_profiles(self.db)
        self.assertEqual(self.db.execute("SELECT key FROM enrichment_tasks WHERE kind='profile'").fetchall(), [("99",)])

    def test_top_scope_balances_media_and_keeps_existing_data(self):
        for medium in ("anime", "manga"):
            for identifier in range(10, 14):
                store(self.db, {"type": medium, "mal_id": identifier, "title": "Fixture", "catalog_rank": identifier - 9})
        self.db.commit()
        setup(self.db, top_titles=4)
        selected = self.db.execute("SELECT key FROM full_priorities").fetchall()
        self.assertEqual(len(selected), 4)
        self.assertTrue(any(key.startswith("anime:") for key, in selected))
        self.assertTrue(any(key.startswith("manga:") for key, in selected))
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM documents").fetchone()[0], 9)

    def test_missing_rank_does_not_put_high_scoring_legacy_titles_last(self):
        store(self.db, {"type":"manga","mal_id":8,"title":"High score","score":9.2})
        store(self.db, {"type":"manga","mal_id":9,"title":"Low score","catalog_rank":5000,"score":5.0})
        self.db.commit()
        setup(self.db, top_titles=2)
        self.assertIn(("manga:8",),self.db.execute("SELECT key FROM full_priorities").fetchall())

    def test_bulk_profiles_and_rosters_are_not_downloaded_again(self):
        self.db.execute("INSERT INTO title_characters VALUES('manga:7',99,'A Person','Supporting')")
        self.db.execute("INSERT INTO characters VALUES(99,?)", (json.dumps({"about": "Snapshot biography"}),))
        seed_profiles(self.db)
        client = self.client("This must not be requested", 503)
        self.assertEqual(run_batch(self.db, client, "profile", 32), 0)
        self.assertEqual(run_batch(self.db, client, "roster", 32), 0)

    def test_out_of_scope_roster_is_not_requested(self):
        store(self.db, {"type": "manga", "mal_id": 88, "title": "Outside", "catalog_rank": 999})
        self.db.commit()
        setup(self.db, top_titles=1)
        run_batch(self.db, self.client("No characters have been added."), "roster", 32)
        self.assertIsNone(self.db.execute("SELECT attempts FROM enrichment_tasks WHERE kind='roster' AND key='manga:88'").fetchone())

    def test_complete_text_does_not_cap_profiles_or_biographies(self):
        biography = "Biography details. " * 1000
        for identifier in range(6):
            self.db.execute("INSERT INTO title_characters VALUES(?,?,?,?)", ("manga:7", identifier, f"Person {identifier}", "Supporting"))
            self.db.execute("INSERT INTO characters VALUES(?,?)", (identifier, json.dumps({"about": biography})))
        update_title(self.db, "manga:7", complete=True)
        row = json.loads(self.db.execute("SELECT row_json FROM documents").fetchone()[0])
        self.assertEqual(row["text"].count(biography), 6)
        self.assertIn("Person 5 (Supporting)", row["text"])

    def test_profile_has_full_biography_native_alias_and_image(self):
        biography = "Long biography. " * 2000
        body = '<img class="portrait-225x350" data-src="https://cdn.myanimelist.net/c.jpg"><h2 class="normal_header">Name (Native)</h2>' + biography + '<div class="normal_header">Voice Actors</div>Not biography'
        parser = ProfileParser()
        parser.feed(body)
        profile = parser.result()
        self.assertEqual(profile["about"], biography.strip())
        self.assertIn("Native", profile["aliases"])
        self.assertEqual(profile["image_url"], "https://cdn.myanimelist.net/c.jpg")

    def test_full_profile_raw_response_retained(self):
        body = '<h2 class="normal_header">Name</h2>Biography<div class="normal_header">Voice Actors</div>Voice actor'
        with self.db:
            collect_profile(self.db, self.client(body), "99")
        raw = self.db.execute("SELECT html_gzip FROM mal_pages").fetchone()[0]
        self.assertEqual(gzip.decompress(raw).decode(), body)

    def test_empty_roster_is_verified_and_challenge_is_not_empty(self):
        with self.db:
            self.assertEqual(collect_roster(self.db, self.client("No characters have been added."), "manga:7"), "empty")
        with self.assertRaises(ValueError), self.db:
            collect_roster(self.db, self.client("Verify you are human"), "manga:7")

    def test_successful_checkpoint_is_not_fetched_twice(self):
        with patch("rag.full_enrichment.collect_roster"):
            client = self.client("No characters have been added.")
            self.assertEqual(run_batch(self.db, client, "roster", 32), 1)
            self.assertEqual(run_batch(self.db, client, "roster", 32), 0)
        self.assertEqual(coverage(self.db)["stages"]["roster"], {"empty": 1})

    def test_missing_and_parse_failure_are_not_reported_as_enriched(self):
        run_batch(self.db, self.client("Not found", 404), "roster", 32)
        self.assertEqual(coverage(self.db)["stages"]["roster"], {"missing": 1})
        self.assertEqual(self.db.execute("SELECT enriched FROM documents").fetchone()[0], 0)
        self.db.execute("UPDATE enrichment_tasks SET status='pending' WHERE kind='roster'")
        self.db.commit()
        with patch("rag.full_enrichment.time.sleep"):
            run_batch(self.db, self.client("Unrecognized page"), "roster", 32)
        self.assertEqual(coverage(self.db)["stages"]["roster"], {"failed": 1})

    def test_access_rejection_stops_without_bypassing_restriction(self):
        with self.assertRaisesRegex(RuntimeError, "paused"):
            run_batch(self.db, self.client("Forbidden", 403), "roster", 32)

    def test_title_not_complete_until_all_supporting_profiles_checked(self):
        self.db.execute("UPDATE enrichment_tasks SET status='done'")
        self.db.execute("INSERT INTO title_characters VALUES('manga:7',99,'A Person','Supporting')")
        seed_profiles(self.db)
        self.assertEqual(coverage(self.db)["fully_checked_titles"], 0)
        self.db.execute("UPDATE enrichment_tasks SET status='empty' WHERE kind='profile'")
        self.assertEqual(coverage(self.db)["fully_checked_titles"], 1)


if __name__ == "__main__":
    unittest.main()
