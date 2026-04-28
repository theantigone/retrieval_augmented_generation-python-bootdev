import os
from collections import defaultdict
from lib.search_utils import load_movies

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents: list[dict]):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch("all-MiniLM-L6-v2")
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> dict:
        bm25_results = self._bm25_search(query, limit * 500)
        ss_results = self.semantic_search.search_chunks(query, limit * 500)
        bm25_scores = []
        ss_scores = []
        for bm25_result, ss_result in zip(bm25_results, ss_results):
            bm25_scores.append(bm25_result["score"])
            ss_scores.append(ss_result["score"])
        bm25_norm_scores = normalize_scores(bm25_scores)
        ss_norm_scores = normalize_scores(ss_scores)
        for bm25_norm_score, ss_norm_score, bm25_result, ss_result in zip(
            bm25_norm_scores, ss_norm_scores, bm25_results, ss_results
        ):
            bm25_result["score"] = bm25_norm_score
            ss_result["score"] = ss_norm_score
        doc_mappings = defaultdict(lambda: 0)
        for doc, bm25_score, semantic_score in zip(
            self.documents, bm25_norm_scores, ss_norm_scores
        ):
            doc_mappings[doc["id"]] = doc
            doc_mappings["bm25_score"] = bm25_score
            doc_mappings["semantic_score"] = semantic_score
            doc_mappings["hybrid_score"] = hybrid_score(
                bm25_score, semantic_score, alpha
            )
        return sorted(doc_mappings.items(), key=lambda x: x[1], reverse=True)
        # raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return
    max_score = max(scores)
    min_score = min(scores)
    if max_score == min_score:
        return [1.0] * len(scores)
    else:
        return [(score - min_score) / (max_score - min_score) for score in scores]


def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


def weighted_search(query: str, alpha: float, limit: int = 5) -> dict:
    documents = load_movies()
    hs = HybridSearch(documents)
    return hs.weighted_search(query, alpha, limit)
