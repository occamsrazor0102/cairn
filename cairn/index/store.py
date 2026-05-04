from __future__ import annotations
import logging
import sqlite3
from pathlib import Path

from cairn.index.retrieval import RetrievalIndex

logger = logging.getLogger("cairn.index.store")


class PersistentRetrievalIndex(RetrievalIndex):
    def __init__(self, db_path: Path, **kw):
        super().__init__(**kw)
        self._db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None
        self._open_db()
        self._rebuild_from_db()

    @property
    def size(self) -> int:
        return len(self._ids)

    def _open_db(self) -> None:
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            try:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS submissions ("
                    " sid TEXT PRIMARY KEY, text TEXT NOT NULL, "
                    " ts REAL DEFAULT (strftime('%s','now')))"
                )
                conn.commit()
            except sqlite3.Error:
                try:
                    conn.close()
                except sqlite3.Error:
                    pass
                raise
            self._conn = conn
        except sqlite3.Error:
            logger.warning(
                "PersistentRetrievalIndex: db open failed at %s",
                self._db_path,
                exc_info=True,
            )
            self._conn = None

    def _rebuild_from_db(self) -> None:
        if self._conn is None:
            return
        try:
            cur = self._conn.execute("SELECT sid, text FROM submissions ORDER BY ts ASC")
            for sid, text in cur.fetchall():
                super().add(sid, text)
        except sqlite3.Error:
            logger.warning("PersistentRetrievalIndex: rebuild failed", exc_info=True)

    def add(self, sid: str, text: str) -> None:
        super().add(sid, text)
        if self._conn is None:
            return
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO submissions(sid, text) VALUES(?, ?)",
                (sid, text),
            )
            self._conn.commit()
        except sqlite3.Error:
            logger.warning("PersistentRetrievalIndex: insert failed for %r", sid, exc_info=True)

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:
                logger.debug("PersistentRetrievalIndex.close failed", exc_info=True)
            self._conn = None
