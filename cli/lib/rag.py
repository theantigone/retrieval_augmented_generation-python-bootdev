from lib.hybrid_search import HybridSearch
from lib.llm import (
    answer_question,
    augmented_generation,
    citations_documents,
    summarize_documents,
)
from lib.search_utils import load_movies


def retrieving_augmented_generation(query: str) -> None:
    movies = load_movies()
    hs = HybridSearch(movies)
    rrf_results = hs.rrf_search(query, k=60, limit=5)
    print("Search Results:")
    for res in rrf_results:
        print("-", res["title"])
    response = augmented_generation(query, rrf_results)
    print("\nRAG Response:")
    print(response)


def doc_summarization(query: str, limit: int = 5) -> None:
    movies = load_movies()
    hs = HybridSearch(movies)
    rrf_results = hs.rrf_search(query, k=60, limit=limit)
    print("Search Results:")
    for res in rrf_results:
        print("-", res["title"])
    response = summarize_documents(query, rrf_results)
    print("\nLLM Summary:")
    print(response)


def doc_citations(query: str, limit: int = 5) -> None:
    movies = load_movies()
    hs = HybridSearch(movies)
    rrf_results = hs.rrf_search(query, k=60, limit=limit)
    print("Search Results:")
    for res in rrf_results:
        print("-", res["title"])
    response = citations_documents(query, rrf_results)
    print("\nLLM Answer:")
    print(response)


def question_answering(query: str, limit: int = 5) -> None:
    movies = load_movies()
    hs = HybridSearch(movies)
    rrf_results = hs.rrf_search(query, k=60, limit=limit)
    print("Search Results:")
    for res in rrf_results:
        print("-", res["title"])
    response = answer_question(query, rrf_results)
    print("\nAnswer:")
    print(response)
