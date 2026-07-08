"""
═══════════════════════════════════════════════════════════════════
AETHER — Phase 5 · Retrieval Seam
═══════════════════════════════════════════════════════════════════
Retrieval is the "R" in RAG: given a query (a track + its detected emotion),
find the knowledge docs most relevant to explaining it. The explainer then
grounds its prose ONLY in what comes back here.

Two interchangeable backends behind one interface (same pattern as Phase 4's
MusicProvider / NullProvider):

  • TfidfRetriever  — pure-NumPy TF-IDF cosine. Zero external deps, fully
                      offline, deterministic. The default, and what CI/self-
                      tests run on: an interviewer can clone and run with no
                      model download and no network.
  • ChromaRetriever — the production backend named in config: ChromaDB +
                      sentence-transformers `all-MiniLM-L6-v2` dense embeddings.
                      Lazily imported so its (heavy) deps are optional; if they
                      aren't installed, build_retriever() transparently falls
                      back to TF-IDF.

Because the corpus is tiny (~two dozen curated docs), TF-IDF retrieval is not a
compromise — lexical overlap is more than enough to route "sad, low energy" to
the right emotion/feature docs. Dense embeddings mainly help once the corpus
grows or queries get more paraphrased; the seam lets us swap up with no change
to the explainer.
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# ── Path bootstrap: root config + local knowledge module ──
_ROOT = Path(__file__).resolve().parent.parent
_HERE = Path(__file__).resolve().parent
for _p in (_ROOT, _HERE):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from knowledge import KnowledgeDoc, build_knowledge_base  # noqa: E402

try:
    from config import (                                   # type: ignore
        CHROMA_COLLECTION_NAME, EMBEDDING_MODEL, RAG_TOP_K,
    )
except Exception:  # pragma: no cover
    CHROMA_COLLECTION_NAME, EMBEDDING_MODEL, RAG_TOP_K = "aether_songs", "all-MiniLM-L6-v2", 10


# ──────────────────────────────────────────────
# Result type
# ──────────────────────────────────────────────
@dataclass
class RetrievedDoc:
    """A knowledge doc returned by a retriever, with its relevance score."""
    doc: KnowledgeDoc
    score: float


# ──────────────────────────────────────────────
# Interface
# ──────────────────────────────────────────────
class Retriever(ABC):
    """Abstract retriever over the Phase 5 knowledge base."""

    #: Short identifier for logging / provenance.
    name: str = "abstract"

    @abstractmethod
    def index(self, docs: list[KnowledgeDoc]) -> "Retriever":
        """Ingest/embed the corpus. Returns self for chaining."""
        raise NotImplementedError

    @abstractmethod
    def query(self, text: str, k: int = RAG_TOP_K) -> list[RetrievedDoc]:
        """Return the top-`k` docs most relevant to `text`, best first."""
        raise NotImplementedError


# ──────────────────────────────────────────────
# Default backend — pure-NumPy TF-IDF
# ──────────────────────────────────────────────
# Compact English stopword set — enough to keep TF-IDF focused on content words
# without pulling in a dependency.
_STOPWORDS = frozenset("""
a an and are as at be by for from has have in into is it its of on or that the
to was were will with this these those it's you your they them their we our but
if then than so such not no can more most much very whats what which who whom
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase, split to alphanumeric tokens, drop stopwords and 1-char noise."""
    return [
        tok for tok in _TOKEN_RE.findall(text.lower())
        if len(tok) > 1 and tok not in _STOPWORDS
    ]


class TfidfRetriever(Retriever):
    """
    Deterministic TF-IDF cosine retriever. No external dependencies.

    Builds a vocabulary from the corpus, forms L2-normalized TF-IDF row vectors,
    and scores a query by cosine similarity (dot product of unit vectors).
    """

    name = "tfidf"

    def __init__(self) -> None:
        self._docs: list[KnowledgeDoc] = []
        self._vocab: dict[str, int] = {}
        self._idf: np.ndarray = np.empty((0,), dtype=np.float64)
        self._matrix: np.ndarray = np.empty((0, 0), dtype=np.float64)  # (N, V)

    def index(self, docs: list[KnowledgeDoc]) -> "TfidfRetriever":
        if not docs:
            raise ValueError("Cannot index an empty corpus.")
        self._docs = list(docs)
        tokenized = [_tokenize(d.searchable_text()) for d in self._docs]

        # Vocabulary (sorted for determinism).
        vocab_terms = sorted({t for toks in tokenized for t in toks})
        self._vocab = {t: i for i, t in enumerate(vocab_terms)}
        V, N = len(vocab_terms), len(self._docs)

        # Document frequency → smoothed idf.
        df = np.zeros(V, dtype=np.float64)
        for toks in tokenized:
            for t in set(toks):
                df[self._vocab[t]] += 1.0
        self._idf = np.log((1.0 + N) / (1.0 + df)) + 1.0   # smoothed, always > 0

        # TF-IDF matrix, L2-normalized rows.
        mat = np.zeros((N, V), dtype=np.float64)
        for r, toks in enumerate(tokenized):
            if not toks:
                continue
            for t in toks:
                mat[r, self._vocab[t]] += 1.0
            mat[r] /= len(toks)                # term frequency
            mat[r] *= self._idf                # × idf
        self._matrix = _l2_normalize_rows(mat)
        return self

    def query(self, text: str, k: int = RAG_TOP_K) -> list[RetrievedDoc]:
        if self._matrix.size == 0:
            raise RuntimeError("Retriever not indexed — call index(docs) first.")
        toks = _tokenize(text)
        q = np.zeros(len(self._vocab), dtype=np.float64)
        for t in toks:
            j = self._vocab.get(t)
            if j is not None:
                q[j] += 1.0
        if toks:
            q /= len(toks)
        q *= self._idf
        norm = float(np.linalg.norm(q))
        if norm > 0:
            q /= norm

        scores = self._matrix @ q                       # cosine (unit vectors)
        k = max(1, min(k, len(self._docs)))
        # argpartition for top-k, then sort those k by score desc (stable-ish).
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [RetrievedDoc(self._docs[i], float(scores[i])) for i in top]


# ──────────────────────────────────────────────
# Production backend — ChromaDB + MiniLM (optional deps)
# ──────────────────────────────────────────────
class ChromaRetriever(Retriever):
    """
    Dense-embedding retriever using ChromaDB + sentence-transformers.

    Honors config: EMBEDDING_MODEL (all-MiniLM-L6-v2), CHROMA_COLLECTION_NAME,
    RAG_TOP_K. Deps are imported lazily so the module stays importable (and the
    TF-IDF default keeps working) even when chromadb / sentence-transformers
    aren't installed.
    """

    name = "chroma"

    def __init__(self, collection_name: str = CHROMA_COLLECTION_NAME,
                 model_name: str = EMBEDDING_MODEL,
                 persist_dir: Optional[str] = None) -> None:
        self.collection_name = collection_name
        self.model_name = model_name
        self.persist_dir = persist_dir
        self._docs_by_id: dict[str, KnowledgeDoc] = {}
        self._model = None
        self._collection = None

    @staticmethod
    def is_available() -> bool:
        """True iff chromadb and sentence-transformers can be imported."""
        try:
            import chromadb  # noqa: F401
            import sentence_transformers  # noqa: F401
            return True
        except Exception:
            return False

    def index(self, docs: list[KnowledgeDoc]) -> "ChromaRetriever":
        if not docs:
            raise ValueError("Cannot index an empty corpus.")
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover — guarded by build_retriever
            raise ImportError(
                "ChromaRetriever needs `chromadb` and `sentence-transformers`. "
                "Install them, or use TfidfRetriever."
            ) from exc

        self._docs_by_id = {d.doc_id: d for d in docs}
        self._model = SentenceTransformer(self.model_name)

        client = (chromadb.PersistentClient(path=self.persist_dir)
                  if self.persist_dir else chromadb.EphemeralClient())
        # Fresh collection each build so re-indexing is idempotent.
        try:
            client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = client.create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"},
        )

        texts = [d.searchable_text() for d in docs]
        embeddings = self._model.encode(texts, normalize_embeddings=True).tolist()
        self._collection.add(
            ids=[d.doc_id for d in docs],
            documents=texts,
            embeddings=embeddings,
            metadatas=[{"kind": d.kind, "title": d.title} for d in docs],
        )
        return self

    def query(self, text: str, k: int = RAG_TOP_K) -> list[RetrievedDoc]:
        if self._collection is None or self._model is None:
            raise RuntimeError("Retriever not indexed — call index(docs) first.")
        k = max(1, min(k, len(self._docs_by_id)))
        q_emb = self._model.encode([text], normalize_embeddings=True).tolist()
        res = self._collection.query(query_embeddings=q_emb, n_results=k)
        ids = res.get("ids", [[]])[0]
        dists = res.get("distances", [[None] * len(ids)])[0]
        out: list[RetrievedDoc] = []
        for doc_id, dist in zip(ids, dists):
            doc = self._docs_by_id.get(doc_id)
            if doc is None:
                continue
            # cosine distance → similarity in [0, 1] (Chroma returns 1 - cos_sim).
            score = 1.0 - float(dist) if dist is not None else 0.0
            out.append(RetrievedDoc(doc, score))
        return out


# ──────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────
def build_retriever(
    docs: Optional[list[KnowledgeDoc]] = None,
    prefer_chroma: bool = True,
    persist_dir: Optional[str] = None,
) -> Retriever:
    """
    Build and index a retriever over the knowledge base.

    Picks ChromaRetriever when `prefer_chroma` and its deps are installed;
    otherwise falls back to the always-available TfidfRetriever. Either way the
    returned retriever is already indexed and ready to query.

    Args:
        docs: corpus to index (defaults to build_knowledge_base()).
        prefer_chroma: try the dense backend first.
        persist_dir: optional on-disk path for Chroma (ephemeral if None).
    """
    corpus = docs if docs is not None else build_knowledge_base()
    if prefer_chroma and ChromaRetriever.is_available():
        try:
            return ChromaRetriever(persist_dir=persist_dir).index(corpus)
        except Exception as exc:  # pragma: no cover — defensive fallback
            print(f"  [retriever] Chroma unavailable ({exc}); using TF-IDF.")
    return TfidfRetriever().index(corpus)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def _l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    """L2-normalize each row; zero rows stay zero (no divide-by-zero)."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return mat / norms


# ─────────────────────────────────────────────────────────────
# Self-test — runs on the always-available TF-IDF backend (offline)
# ─────────────────────────────────────────────────────────────
def _selftest() -> None:
    print("Retriever self-test")
    print("-" * 55)

    docs = build_knowledge_base()
    r = TfidfRetriever().index(docs)
    print(f"  indexed {len(docs)} docs into '{r.name}' "
          f"(vocab={len(r._vocab)})")

    # 1. A sad, low-energy query surfaces the sad emotion doc near the top.
    hits = r.query("sad slow low energy minor key ballad", k=5)
    ids = [h.doc.doc_id for h in hits]
    assert "emotion:sad" in ids, ids
    assert hits[0].score > 0, hits[0].score
    print(f"  'sad slow low energy'    → {ids[:3]}")

    # 2. A tempo query surfaces the tempo feature doc.
    hits = r.query("fast tempo beats per minute arousal", k=5)
    ids = [h.doc.doc_id for h in hits]
    assert "feature:tempo" in ids, ids
    print(f"  'fast tempo arousal'     → {ids[:3]}")

    # 3. An energetic-dance query surfaces the energetic emotion doc.
    hits = r.query("energetic high tempo dance EDM adrenaline movement", k=5)
    ids = [h.doc.doc_id for h in hits]
    assert "emotion:energetic" in ids, ids
    print(f"  'energetic dance EDM'    → {ids[:3]}")

    # 4. Scores are sorted descending, and k is respected.
    hits = r.query("valence positivity happy bright major key", k=4)
    assert len(hits) == 4, len(hits)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True), scores
    assert "feature:valence" in [h.doc.doc_id for h in hits] \
        or "emotion:happy" in [h.doc.doc_id for h in hits]
    print(f"  'valence happy bright'   → {[h.doc.doc_id for h in hits]}")

    # 5. Empty/garbage query returns k docs without crashing (all-zero scores ok).
    hits = r.query("zzzz qqqq", k=3)
    assert len(hits) == 3
    print(f"  garbage query            → {len(hits)} docs, top score "
          f"{hits[0].score:.3f}")

    # 6. Factory returns an indexed, queryable retriever.
    rf = build_retriever(prefer_chroma=False)
    assert rf.query("calm ambient", k=2)
    print(f"  build_retriever()        → '{rf.name}' backend ready")

    # 7. Chroma availability is reported honestly (informational).
    print(f"  chroma deps installed?   → {ChromaRetriever.is_available()}")

    print("-" * 55)
    print("✅ All retriever self-tests passed.")


if __name__ == "__main__":
    _selftest()
