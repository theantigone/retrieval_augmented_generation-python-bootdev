from lib.llm import answer_question, summarize_documents, cite_docs
from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies


def question_answering(query: str) -> None:
    movies = load_movies()
    hs = HybridSearch(movies)
    rrf_results = hs.rrf_search(query, k=60, limit=5)
    print("Search Results:")
    for res in rrf_results:
        print("-", res["title"])
    response = answer_question(query, rrf_results)
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


def doc_citing(query: str, limit: int = 5) -> None:
    movies = load_movies()
    hs = HybridSearch(movies)
    rrf_results = hs.rrf_search(query, k=60, limit=limit)
    print("Search Results:")
    for res in rrf_results:
        print("-", res["title"])
    response = cite_docs(query, rrf_results)
    print("\nLLM Answer:")
    print(response)
