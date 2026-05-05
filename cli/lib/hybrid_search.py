import os
from lib.search_utils import load_movies
from lib.llm import augment_prompt
from lib.rerank import individual_rerank, batch_rerank

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
        combined_results = combine_search_results(bm25_results, sem_results, alpha)
        return combined_results[:limit]
        # raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query: str, k: int, limit: int = 10):
        bm25_results = self._bm25_search(query, limit * 500)
        sem_results = self.semantic_search.search_chunks(query, limit * 500)
        combined_results = rrf_combine_search_results(bm25_results, sem_results, k)
        return combined_results[:limit]
        # raise NotImplementedError("RRF hybrid search is not implemented yet.")


def normalize_scores(scores: list[float]) -> list[float] | None:
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
    bm25_results: list[dict], sem_results: list[dict], alpha: float
) -> dict:
    bm25_norm = normalize_search_results(bm25_results)
    sem_norm = normalize_search_results(sem_results)

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
            value["bm25_score"], value["sem_score"], alpha
        )

    results = sorted(
        combined_norm.values(), key=lambda x: x["hybrid_score"], reverse=True
    )
    return results


def rrf_combine_search_results(
    bm25_results: list[dict], sem_results: list[dict], k: int = 60
) -> dict:
    scores = {}

    for rank, result in enumerate(bm25_results, start=1):
        doc_id = result["id"]
        scores[doc_id] = {
            "id": doc_id,
            "title": result["title"],
            "description": result["document"],
            "bm25_rank": rank,
            "bm25_score": rrf_score(rank, k),
            "sem_rank": None,
            "sem_score": None,
        }

    for rank, result in enumerate(sem_results, start=1):
        doc_id = result["id"]
        if doc_id not in scores:
            scores[doc_id] = {
                "id": doc_id,
                "title": result["title"],
                "description": result["document"],
                "bm25_rank": None,
                "bm25_score": None,
                "sem_rank": rank,
                "sem_score": rrf_score(rank, k),
            }
        scores[doc_id]["sem_rank"] = rank
        scores[doc_id]["sem_score"] = rrf_score(rank, k)

    for key, value in scores.items():
        scores[key]["rrf_score"] = rrf_final_score(
            value["bm25_score"], value["sem_score"], k
        )

    results = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    return results


def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)


def rrf_final_score(s1: float, s2: float, k: int = 60) -> float:
    return (s1 or 0.0) + (s2 or 0.0)


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


def rrf_search(
    query: str,
    k: int = 60,
    limit: int = 5,
    enhance: str | None = None,
    rerank_method=None,
) -> None:
    documents = load_movies()
    hs = HybridSearch(documents)
    if enhance:
        new_query = augment_prompt(query, enhance)
        print(f"Enhanced query ({enhance}): '{query}' -> '{new_query}'\n")
        query = new_query
    rrf_limit = limit * 5 if rerank_method else limit
    results = hs.rrf_search(query, k, rrf_limit)
    match rerank_method:
        case "individual":
            results = individual_rerank(query, results)
        case "batch":
            results = batch_rerank(query, results)
        case _:
            pass
    if rerank_method:
        print(f"Re-ranking top {limit} results using {rerank_method} method...")
        print(f"Reciprocal Rank Fusion Results for '{query}' (k={k}):")
    for idx, r in enumerate(results[:limit]):
        print(f"{idx}. {r['title']}")
        match rerank_method:
            case "individual":
                print(f"   Re-rank Score: {r['rerank_response']:.3f}/10")
            case _:
                pass
        print(f"   RRF Score: {r['rrf_score']:.3f}")
        print(f"   BM25 Rank: {r['bm25_rank']}, Semantic Rank: {r['sem_rank']}")
        print(f"   {r['description'][:100]}")
