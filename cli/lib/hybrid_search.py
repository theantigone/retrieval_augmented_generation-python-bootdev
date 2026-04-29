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

    def rrf_search(self, query: str, k: int, limit: int = 10):
        bm25_results = self._bm25_search(query, limit * 500)
        sem_results = self.semantic_search.search_chunks(query, limit * 500)
        combined_results = rrf_combine_search_results(bm25_results, sem_results, k)
        return combined_results[:limit]
        # raise NotImplementedError("RRF hybrid search is not implemented yet.")


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


def rrf_combine_search_results(
    bm25_results: list[dict], semantic_results: list[dict], k: int = 60
) -> dict:
    combined_results = {}
    bm25_ranks = rrf_rank(bm25_results)
    sem_ranks = rrf_rank(semantic_results)

    for i, result in enumerate(bm25_results):
        doc_id = result["id"]
        combined_results[doc_id] = {
            "id": doc_id,
            "title": result["title"],
            "description": result["document"],
            "bm25_rank": bm25_ranks[i]["rrf_rank"],
            "sem_rank": 0.0,
        }

    for i, result in enumerate(sem_results):
        doc_id = result["id"]
        combined_results[doc_id] = {
            "id": doc_id,
            "title": result["title"],
            "description": result["document"],
            "bm25_rank": 0.0,
            "sem_rank": sem_ranks[i]["rrf_rank"],
        }

    for key, value in combined_results.items():
        combined_results[key]["rrf_score"] = rrf_score(
            value["bm25_rank"], k
        ) + rrf_score(value["sem_rank"], k)

    results = sorted(
        combined_results.values(), key=lambda x: x["rrf_score"], reverse=True
    )
    return results


def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)


def rrf_rank(search: list[dict]) -> int:
    sorted_search = sorted(search, key=lambda x: x["score"])
    for i, search in zip(range(1, len(sorted_search) - 1), sorted_search):
        search["rrf_rank"] = i
    print("---")
    print(sorted_search[0]["score"])
    print(sorted_search[0]["rrf_rank"])
    print(sorted_search[1]["score"])
    print(sorted_search[1]["rrf_rank"])
    print("---")
    return sorted_search


def weighted_search(query: str, alpha: float = 0.5, limit: int = 5) -> None:
    if alpha < 0 or alpha > 1:
        raise ValueError("value must be between 0 and 1")
    documents = load_movies()
    hs = HybridSearch(documents)
    weighted_results = hs.weighted_search(query, alpha, limit)
    for i, weighted_result in enumerate(weighted_results[:limit]):
        print(f"{i + 1}. {weighted_result['title']}")
        print(f"   Hybrid Score: {weighted_result['hybrid_score']}")
        print(
            f"   BM25: {weighted_result['bm25_score']}, Semantic: {weighted_result['sem_score']}"
        )
        print(f"   {weighted_result['description'][:100]}")


def rrf_search(query: str, k: int = 60, limit: int = 10) -> None:
    documents = load_movies()
    hs = HybridSearch(documents)
    rrf_results = hs.rrf_search(query, k, limit)
    for i, rrf_result in enumerate(rrf_results[:limit]):
        print(f"{i + 1}. {rrf_result['title']}")
        print(f"   RRF Score: {rrf_result['rrf_score']}")
        print(
            f"   BM25 Rank: {rrf_result['bm25_rank']}, Semantic Rank: {rrf_result['sem_rank']}"
        )
        print(f"   {rrf_result['description'][:100]}")
