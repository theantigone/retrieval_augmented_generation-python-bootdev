from lib.llm import answer_question
from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies


def rag(query: str):
    movies = load_movies()
    hs = HybridSearch(movies)
    results = hs.rrf_search(query, k=60, limit=5)
    print("Search Results:")
    for res in results:
        print("-", res["title"])
    response = answer_question(query, movies)
    print("\nRAG Response:")
    print(response)
