from pathlib import Path
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from .search_utils import load_movies


class SemanticSearch:
    def __init__(self):
        # Load the model (downloads automatically the first time)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        self.embeddings_path = Path("cache/movie_embeddings.npy")

    def generate_embedding(self, text):
        if not text or not text.strip():
            raise ValueError("Must have text to create an embedding")
        return self.model.encode([text])[0]

    def build_embeddings(self, documents: list[dict]):
        self.documents = documents
        movie_strings = []
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            movie_strings.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)
        os.makedirs("cache", exist_ok=True)
        np.save(self.embeddings_path, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
        if os.path.exists(self.embeddings_path):
            self.embeddings = np.load(self.embeddings_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        else:
            return self.build_embeddings(documents)

    def search(self, query, limit):
        if not self.embeddings:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )
        embedding = self.generate_embedding(query)
        scores = []
        for doc_embedding, doc in zip(self.embeddings, self.documents):
            cos_sim = cosine_similarity(embedding, doc_embedding)
            scores.append((cos_sim, doc))
        sorted_list = sorted(scores, key=lambda x: x[0], reverse=True)[:limit]
        top_results = []
        for score, doc in sorted_list:
            top_results.append(
                {
                    "score": score,
                    "title": doc["title"],
                    "description": doc["description"],
                }
            )
        return top_results


def verify_model() -> None:
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")


def embed_text(text: str) -> None:
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings() -> None:
    ss = SemanticSearch()
    documents = load_movies()
    embeddings = ss.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query_text(query: str) -> None:
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1: float, vec2: float) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def search_movies(query: str, limit: int = 5) -> None:
    ss = SemanticSearch()
    documents = load_movies()
    ss.load_or_create_embeddings(documents)
    search_results = ss.search(query, limit)
    for i, result in enumerate(search_results):
        print(f"{i + 1}. {result['title']} (score: {result['score']:.4f}")
        print(f"\t{result['description']}")


def fixed_sized_chunking(
    text: str, chunk_size: int = 200, overlap: int = 0
) -> list[str]:
    words: list[str] = text.rsplit()
    chunks = []
    for i in range(0, len(words), chunk_size):
        if i > 0 and overlap > 0:
            chunks.append(" ".join(words[i - overlap : i + chunk_size]))
        else:
            chunks.append(" ".join(words[i : i + chunk_size]))
    return chunks


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 0) -> None:
    chunks = fixed_sized_chunking(text, chunk_size)
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")
