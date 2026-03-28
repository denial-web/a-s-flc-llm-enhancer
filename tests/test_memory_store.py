"""Tests for core.memory_store — SQLite + FAISS/NumPy memory store."""

import json
import tempfile
from pathlib import Path

import pytest

from core.memory_store import Embedder, MemoryStore, _hash_embed


def test_hash_embed_shape():
    vec = _hash_embed("hello world")
    assert vec.shape == (384,)
    assert abs(float((vec ** 2).sum()) - 1.0) < 1e-5


def test_hash_embed_deterministic():
    a = _hash_embed("test string")
    b = _hash_embed("test string")
    assert (a == b).all()


def test_store_and_retrieve_in_memory():
    store = MemoryStore(db_path=":memory:")
    store.store("user prefers window seats", "always book window", category="preference")
    store.store("budget is $500 for electronics", 500, category="finance")
    store.store("vegetarian diet", True, category="health")

    assert store.count() == 3

    results = store.retrieve("what seat should I book on my flight?", top_k=2)
    assert len(results) > 0
    assert any("window" in r.key for r in results)


def test_store_and_retrieve_with_category_filter():
    store = MemoryStore(db_path=":memory:")
    store.store("favorite color is blue", "blue", category="preference")
    store.store("salary is $80k", 80000, category="finance")

    finance_results = store.retrieve("money questions", top_k=5, category="finance")
    assert all(r.category == "finance" for r in finance_results)


def test_list_all():
    store = MemoryStore(db_path=":memory:")
    store.store("fact one", "val1", category="a")
    store.store("fact two", "val2", category="b")
    store.store("fact three", "val3", category="a")

    all_entries = store.list_all()
    assert len(all_entries) == 3

    cat_a = store.list_all(category="a")
    assert len(cat_a) == 2


def test_empty_store_retrieve():
    store = MemoryStore(db_path=":memory:")
    results = store.retrieve("anything", top_k=5)
    assert results == []


def test_persistence_on_disk():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = Path(tmpdir) / "test.db"
        store = MemoryStore(db_path=db_file)
        store.store("persistent fact", {"important": True})
        store.close()

        store2 = MemoryStore(db_path=db_file)
        assert store2.count() == 1
        results = store2.retrieve("persistent fact")
        assert len(results) == 1
        assert results[0].value == {"important": True}
        store2.close()


def test_embedder_fallback():
    embedder = Embedder(model_name="nonexistent-model-xyz")
    vecs = embedder.encode(["hello", "world"])
    assert vecs.shape == (2, 384)
