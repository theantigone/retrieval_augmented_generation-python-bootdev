import os
from lib.search_utils import load_movies, PROMPT_PATH
from lib.hybrid_search import HybridSearch
from lib.llm import generate_content


def augmented_generation(query: str):
    movies = load_movies()
    hs = HybridSearch(movies)
    rrf_results = hs.rrf_search(query, limit=5)
    rag_path = os.path.join(PROMPT_PATH, "rag.md")
    with open(rag_path, "r") as f:
        prompt = f.read()
    print("Starting RAG process...")
    response = generate_content(prompt, query, docs=rrf_results)
    if not response:
        raise RuntimeError("Empty response:", response)
    print("Successfully ran RAG process")
    return rrf_results, response
