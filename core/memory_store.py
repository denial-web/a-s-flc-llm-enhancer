"""Memory Store — SQLite + FAISS for persistent, semantically-searchable memory.

Each memory entry has a text key, a JSON-serializable value, a category tag,
a timestamp, and a dense embedding for similarity search.

Embedding model is pluggable; defaults to all-MiniLM-L6-v2 (384-dim) via
sentence-transformers when available, with a sha256-hash fallback for tests.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

EMBED_DIM = 384


@dataclass
class MemoryEntry:
    key: str
    value: Any
    category: str = "general"
    timestamp: float = 0.0
    score: float = 0.0


def _hash_embed(text: str) -> np.ndarray:
    """Deterministic pseudo-embedding from SHA-256 (for tests / no-GPU envs)."""
    digest = hashlib.sha256(text.encode()).digest()
    rng = np.random.RandomState(int.from_bytes(digest[:4], "big"))
    vec = rng.randn(EMBED_DIM).astype(np.float32)
    vec /= np.linalg.norm(vec) + 1e-9
    return vec


class Embedder:
    """Lazy-loaded sentence-transformer; falls back to hash-based embedding."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None

    def encode(self, texts: List[str]) -> np.ndarray:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self._model_name)
            except Exception:
                self._model = "fallback"
        if self._model == "fallback":
            return np.stack([_hash_embed(t) for t in texts])
        return self._model.encode(texts, normalize_embeddings=True)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def _blob(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


def _unblog(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32).copy()


class MemoryStore:
    """SQLite-backed memory with FAISS similarity index.

    Falls back to brute-force NumPy cosine search when faiss is not installed.
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        embedder: Optional[Embedder] = None,
    ):
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self._embedder = embedder or Embedder()
        self._faiss_index = None
        self._id_map: List[int] = []
        self._init_db()
        self._rebuild_index()

    def _init_db(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                key       TEXT    NOT NULL,
                value     TEXT    NOT NULL,
                category  TEXT    NOT NULL DEFAULT 'general',
                timestamp REAL    NOT NULL,
                embedding BLOB    NOT NULL
            )
        """)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_cat ON memories(category)"
        )
        self.conn.commit()

    def _rebuild_index(self) -> None:
        rows = self.conn.execute(
            "SELECT id, embedding FROM memories ORDER BY id"
        ).fetchall()
        if not rows:
            self._faiss_index = None
            self._id_map = []
            return

        vecs = np.stack([_unblog(r[1]) for r in rows])
        self._id_map = [r[0] for r in rows]
        try:
            import faiss

            idx = faiss.IndexFlatIP(EMBED_DIM)
            faiss.normalize_L2(vecs)
            idx.add(vecs)
            self._faiss_index = idx
        except ImportError:
            self._faiss_index = vecs

    def store(
        self,
        key: str,
        value: Any,
        category: str = "general",
        timestamp: Optional[float] = None,
    ) -> int:
        ts = timestamp or time.time()
        emb = self._embedder.encode_one(key)
        cur = self.conn.execute(
            "INSERT INTO memories (key, value, category, timestamp, embedding) VALUES (?, ?, ?, ?, ?)",
            (key, json.dumps(value), category, ts, _blob(emb)),
        )
        self.conn.commit()
        row_id = cur.lastrowid

        if self._faiss_index is not None:
            try:
                import faiss

                vec = emb.reshape(1, -1).copy()
                faiss.normalize_L2(vec)
                self._faiss_index.add(vec)
            except ImportError:
                self._faiss_index = np.vstack(
                    [self._faiss_index, emb.reshape(1, -1)]
                )
        else:
            self._rebuild_index()

        self._id_map.append(row_id)
        return row_id

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None,
    ) -> List[MemoryEntry]:
        if self._faiss_index is None:
            return []

        q_emb = self._embedder.encode_one(query).reshape(1, -1).astype(np.float32)

        try:
            import faiss

            faiss.normalize_L2(q_emb)
            scores, indices = self._faiss_index.search(q_emb, min(top_k * 3, len(self._id_map)))
            candidate_ids = [
                self._id_map[i] for i in indices[0] if 0 <= i < len(self._id_map)
            ]
            candidate_scores = {
                self._id_map[i]: float(scores[0][j])
                for j, i in enumerate(indices[0])
                if 0 <= i < len(self._id_map)
            }
        except ImportError:
            q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)
            db_norm = self._faiss_index / (
                np.linalg.norm(self._faiss_index, axis=1, keepdims=True) + 1e-9
            )
            sims = (db_norm @ q_norm.T).flatten()
            top_idx = np.argsort(sims)[::-1][: top_k * 3]
            candidate_ids = [self._id_map[i] for i in top_idx]
            candidate_scores = {
                self._id_map[i]: float(sims[i]) for i in top_idx
            }

        if category:
            placeholders = ",".join("?" for _ in candidate_ids)
            rows = self.conn.execute(
                f"SELECT id, key, value, category, timestamp FROM memories "
                f"WHERE id IN ({placeholders}) AND category = ? ORDER BY timestamp DESC",
                (*candidate_ids, category),
            ).fetchall()
        else:
            placeholders = ",".join("?" for _ in candidate_ids)
            rows = self.conn.execute(
                f"SELECT id, key, value, category, timestamp FROM memories "
                f"WHERE id IN ({placeholders}) ORDER BY timestamp DESC",
                candidate_ids,
            ).fetchall()

        entries = []
        for row in rows:
            rid, k, v, cat, ts = row
            entries.append(
                MemoryEntry(
                    key=k,
                    value=json.loads(v),
                    category=cat,
                    timestamp=ts,
                    score=candidate_scores.get(rid, 0.0),
                )
            )
        entries.sort(key=lambda e: e.score, reverse=True)
        return entries[:top_k]

    def list_all(self, category: Optional[str] = None) -> List[MemoryEntry]:
        if category:
            rows = self.conn.execute(
                "SELECT key, value, category, timestamp FROM memories "
                "WHERE category = ? ORDER BY timestamp DESC",
                (category,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT key, value, category, timestamp FROM memories ORDER BY timestamp DESC"
            ).fetchall()
        return [
            MemoryEntry(key=r[0], value=json.loads(r[1]), category=r[2], timestamp=r[3])
            for r in rows
        ]

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def close(self) -> None:
        self.conn.close()
