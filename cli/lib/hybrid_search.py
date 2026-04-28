import os
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
        sem_results = self.semantic_search.search_chunks(query, limit * 500)
        combined_results = combine_search_results(bm25_results, sem_results)
        return combined_results
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


def normalize_search_results(results: list[dict]) -> list[dict]:
    scores = [r["score"] for r in results]
    norm_scores = normalize_scores(scores)
    for idx, result in enumerate(results):
        result["normalized_score"] = norm_scores[idx]
    return results


def combine_search_results(
    bm25_results: list[dict], semantic_results: list[dict]
) -> dict:
    bm25_norm = normalize_search_results(bm25_results)
    sem_norm = normalize_search_results(semantic_results)

    combined_norm = {}
    for norm in bm25_norm:
        doc_id = norm["id"]
        combined_norm[doc_id] = {
            "id": doc_id,
            "title": norm["title"],
            "description": norm["document"],
            "bm25_score": norm["normalized_score"],
            "sem_score": 0.0,
        }
    for norm in sem_norm:
        doc_id = norm["id"]
        if doc_id not in combined_norm:
            combined_norm[doc_id] = {
                "id": doc_id,
                "title": norm["title"],
                "description": norm["document"],
                "bm25_score": 0.0,
                "sem_score": norm["normalized_score"],
            }
        else:
            combined_norm[doc_id]["sem_score"] = norm["normalized_score"]

    for key, value in combined_norm.items():
        combined_norm[key]["hybrid_score"] = hybrid_score(
            value["bm25_score"], value["sem_score"]
        )

    results = sorted(
        combined_norm.values(), key=lambda x: x["hybrid_score"], reverse=True
    )
    return results


def weighted_search(query: str, alpha: float = 0.5, limit: int = 5) -> None:
    if alpha < 0 or alpha > 1:
        raise ValueError("value must be between 0 and 1")
    documents = load_movies()
    hs = HybridSearch(documents)
    weighted_results = hs.weighted_search(query, alpha, limit)
    for i, weighted_result in enumerate(weighted_results[:limit]):
        print(f"{i + 1}. {weighted_result['title']}")
        print(f"Hybrid Score: {weighted_result['hybrid_score']}")
        print(
            f"BM25: {weighted_result['bm25_score']}, Semantic: {weighted_result['sem_score']}"
        )
        print(f"{weighted_result['description'][:100]}")
