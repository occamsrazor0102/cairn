from __future__ import annotations
import logging

logger = logging.getLogger("cairn.index")

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class RetrievalIndex:
    def __init__(self, model_name: str = _DEFAULT_MODEL):
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
        except ImportError as e:
            raise ImportError(
                "RetrievalIndex requires sentence-transformers + faiss-cpu. "
                "Install with: pip install 'cairn[retrieval]'"
            ) from e
        self._model = SentenceTransformer(model_name)
        self._dim = int(self._model.get_sentence_embedding_dimension())
        self._faiss = faiss
        self._index = faiss.IndexFlatIP(self._dim)
        self._ids: list[str] = []
        self._texts: dict[str, str] = {}

    def add(self, sid: str, text: str) -> None:
        emb = self._embed(text)
        self._index.add(emb)
        self._ids.append(sid)
        self._texts[sid] = text

    def query(self, sid_or_text: str, k: int = 5) -> list[tuple[str, float]]:
        text = self._texts.get(sid_or_text, sid_or_text)
        if self._index.ntotal == 0:
            return []
        emb = self._embed(text)
        k_eff = min(k, self._index.ntotal)
        scores, idx = self._index.search(emb, k_eff)
        out: list[tuple[str, float]] = []
        for s, i in zip(scores[0], idx[0]):
            if i < 0:
                continue
            out.append((self._ids[i], float(s)))
        return out

    def _embed(self, text: str):
        import numpy as np
        emb = self._model.encode([text], normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb, dtype="float32")
