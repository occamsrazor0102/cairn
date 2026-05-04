import pytest


def test_retrieval_index_finds_near_duplicates():
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("faiss")
    from cairn.index.retrieval import RetrievalIndex
    idx = RetrievalIndex()
    idx.add("a", "The quick brown fox jumps over the lazy dog. " * 5)
    idx.add("b", "Dogs are loyal companions to humans throughout history. " * 5)
    idx.add("c", "The quick brown fox jumps over the lazy dog. " * 5)
    nbrs = idx.query("c", k=2)
    nbr_ids = [nid for nid, _ in nbrs if nid != "c"]
    assert nbr_ids[0] == "a"


def test_retrieval_index_empty_returns_no_neighbors():
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("faiss")
    from cairn.index.retrieval import RetrievalIndex
    idx = RetrievalIndex()
    idx.add("a", "Short text. " * 20)
    nbrs = idx.query("a", k=5)
    nbr_ids = [nid for nid, _ in nbrs if nid != "a"]
    assert nbr_ids == []


def test_persistent_index_round_trips(tmp_path):
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("faiss")
    from cairn.index.store import PersistentRetrievalIndex
    db = tmp_path / "ix.db"
    idx = PersistentRetrievalIndex(db_path=db)
    idx.add("a", "The quick brown fox jumps over the lazy dog. " * 5)
    idx.add("b", "Dogs are loyal companions to humans. " * 5)
    idx.close()
    idx2 = PersistentRetrievalIndex(db_path=db)
    assert idx2.size == 2
    nbrs = idx2.query("a", k=2)
    nbr_ids = [n for n, _ in nbrs if n != "a"]
    assert nbr_ids[0] == "b"
