import json
import gzip
from contextlib import closing
from pathlib import Path
import tempfile
import unittest

from rag.bulk_enrichment import identity, listed, profile_from_row, text_values
from rag.catalog_sync import compose, name_aliases, open_db, store
from rag.character_data import schema, update_title


class BulkContracts(unittest.TestCase):
    def test_length_grouping_restores_original_document_vector_alignment(self):
        from rag.offline_embedding import embed_ordered
        from types import SimpleNamespace
        calls = []
        def embed(texts, batch_size):
            calls.extend(texts)
            return iter([len(text), text] for text in texts)
        texts = ["A very long text", "B", "A medium text"]
        vectors = embed_ordered(SimpleNamespace(embed=embed), texts, 16)
        self.assertEqual([v[1] for v in vectors], texts)
        self.assertEqual(calls, ["B", "A medium text", "A very long text"])

    def test_disk_lexicon_keeps_full_names_without_loading_them_into_ram(self):
        from rag.evidence_index import build, EvidenceIndex
        from rag.rag_api import _extract_character_aliases, _normalized_text
        from rag.character_lexicon import normalized_alias
        with tempfile.TemporaryDirectory() as directory:
            corpus, target = Path(directory) / "catalog.gz", Path(directory) / "evidence.sqlite"
            text = "Characters: Family, Given\nCharacter Search Aliases: Other Person\nCharacter Details: Biography words must not become names."
            with gzip.open(corpus, "wt") as handle:
                handle.write(json.dumps({"type":"manga","mal_id":1,"title":"Fixture","text":text}) + "\n")
            build(corpus, target)
            index = EvidenceIndex(target)
            try:
                names, words = index.character_lexicons()
                self.assertEqual(len(names), len(_extract_character_aliases(text)))
                self.assertIn("given family", names)
                self.assertIn("other person", names)
                self.assertIn("family", words)
                self.assertNotIn("biography", words)
                self.assertEqual(index.character_search("given family", "manga"), [{"media_type":"manga", "mal_id":1}])
                self.assertEqual(index.character_search("given family", "anime"), [])
                self.assertEqual(index.character_search("unrelated name", "manga"), [])
                self.assertEqual(names.contains.cache_info().maxsize, 4096)
                for value in ["Family, Given", "A-B/C:D.E(F)", "O\u2019Brien"]:
                    self.assertEqual(normalized_alias(value), _normalized_text(value))
            finally:
                index.db.close()

    def test_offline_sparse_cache_preserves_frozen_vectors_and_is_bounded(self):
        from pinecone_text.sparse import BM25Encoder
        from rag.offline_embedding import cache_sparse_stemming
        encoder = BM25Encoder().load("rag/ingestion/bm25_values.json")
        texts = ["Travelers searched abandoned cities and remembered vanished friends.", "The travelers were searching for friends."]
        expected = encoder.encode_documents(texts)
        cache_sparse_stemming(encoder)
        self.assertEqual(encoder.encode_documents(texts), expected)
        self.assertEqual(encoder._tokenizer._stemmer.stem.cache_info().maxsize, 131072)

    def profile(self, **changes):
        return {"character_id": "77", "full_name": "Given Family", "name": "Family, Given", "url": "https://myanimelist.net/character/77/Given_Family", "description": "A full biography.", "attributes": "[]", **changes}

    def test_wrong_character_identity_rejected(self):
        with self.assertRaises(ValueError):
            profile_from_row(self.profile(url="https://myanimelist.net/character/78/Other"), "profiles")

    def test_media_identity_is_not_shared_between_manga_and_anime(self):
        self.assertFalse(identity("https://myanimelist.net/anime/1/Example", "manga", 1))
        self.assertFalse(identity("https://myanimelist.net.example.org/manga/1", "manga", 1))

    def test_boilerplate_is_missing_data_not_a_biography(self):
        self.assertEqual(profile_from_row(self.profile(description="No biography written."), "profiles")["about"], "")

    def test_attributes_and_all_biography_text_retained(self):
        biography = "Detailed biography. " * 1000
        p = profile_from_row(self.profile(description=biography, attributes='[{"name":"Occupation","value":"Teacher"}]'), "profiles")
        self.assertIn("Occupation: Teacher", p["about"])
        self.assertTrue(p["about"].endswith(biography.strip()))

    def test_structured_parser_does_not_execute_code(self):
        with self.assertRaises((ValueError, SyntaxError)):
            listed("__import__('os').system('false')")
        self.assertEqual(listed("['A', 'B']"), ["A", "B"])

    def test_scalar_metadata_does_not_split_a_title_at_commas(self):
        self.assertEqual(text_values("Name, With Comma"), ["Name, With Comma"])
        self.assertEqual(text_values('["Seinen"]'), ["Seinen"])

    def test_full_alias_mode_includes_large_cast(self):
        self.assertEqual(len(name_aliases([f"Person {i}" for i in range(200)], None)), 200)

    def test_metadata_only_recomposition_does_not_claim_verified_cast(self):
        with tempfile.TemporaryDirectory() as directory, closing(open_db(Path(directory) / "catalog.sqlite")) as db:
            schema(db)
            row = {"type": "manga", "mal_id": 1, "title": "Fixture", "search_details": {"Themes": "Space"}}
            row["text"] = compose(row)
            store(db, row)
            update_title(db, "manga:1", complete=True)
            result = json.loads(db.execute("SELECT row_json FROM documents").fetchone()[0])
            self.assertNotIn("character_schema_version", result)
            self.assertNotIn("character_source", result)


if __name__ == "__main__":
    unittest.main()
