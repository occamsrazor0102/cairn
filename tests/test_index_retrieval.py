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


def test_retrieval_index_concurrent_adds_are_thread_safe():
    """Concurrent add() calls must not corrupt the index or raise."""
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("faiss")
    import threading
    from cairn.index.retrieval import RetrievalIndex

    idx = RetrievalIndex()
    n_threads = 8
    n_adds_per_thread = 5
    errors: list[Exception] = []

    def worker(tid):
        try:
            for i in range(n_adds_per_thread):
                idx.add(f"t{tid}_i{i}", f"thread {tid} item {i} text. " * 10)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent add raised: {errors}"
    assert len(idx._ids) == n_threads * n_adds_per_thread
    # Spot-check that the FAISS index agrees with our id list:
    assert idx._index.ntotal == n_threads * n_adds_per_thread


def test_persistent_index_cross_thread_query(tmp_path):
    """PersistentRetrievalIndex must allow query() from a different thread than __init__."""
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("faiss")
    import threading
    from cairn.index.store import PersistentRetrievalIndex

    db = tmp_path / "ix.db"
    idx = PersistentRetrievalIndex(db_path=db)
    idx.add("a", "Sample text. " * 20)

    result_holder = {}

    def worker():
        try:
            result_holder["nbrs"] = idx.query("a", k=2)
        except Exception as e:
            result_holder["error"] = e

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert "error" not in result_holder, f"cross-thread query raised: {result_holder.get('error')}"
    assert "nbrs" in result_holder
    idx.close()
