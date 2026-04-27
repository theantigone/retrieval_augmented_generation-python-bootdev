from pathlib import Path
import os
import numpy as np
import re
import json
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from .search_utils import load_movies


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
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

    def search(self, query, limit) -> list[dict]:
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )
        embedding = self.generate_embedding(query)
        scores = []
        for doc_embedding, doc in zip(self.embeddings, self.documents):
            sim = cosine_similarity(embedding, doc_embedding)
            scores.append((sim, doc))
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


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_embeddings_path = Path("cache/chunk_embeddings.npy")
        self.chunk_metadata = None
        self.chunk_metadata_path = Path("cache/chunk_metadata.json")

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
        chunk_metadata = []
        all_chunks = []
        for doc_idx, doc in enumerate(self.documents):
            if not doc["description"]:
                continue
            chunks = fixed_sized_semantic_chunking(
                doc["description"], max_chunk_size=4, overlap=1
            )
            start_idx = len(all_chunks)
            all_chunks += chunks
            for local_chunk_idx in range(len(chunks)):
                chunk_metadata.append(
                    {
                        "movie_idx": doc_idx,
                        "chunk_idx": start_idx + local_chunk_idx,
                        "chunk_idx_within_movie": local_chunk_idx,  # optional
                        "total_chunks": len(chunks),
                    }
                )
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata
        os.makedirs("cache", exist_ok=True)
        np.save(self.chunk_embeddings_path, self.chunk_embeddings)
        with open(self.chunk_metadata_path, "w") as f:
            (
                json.dump(
                    {"chunks": chunk_metadata, "total_chunks": len(all_chunks)},
                    f,
                    indent=2,
                ),
            )
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
        if os.path.exists(self.chunk_embeddings_path) and os.path.exists(
            self.chunk_metadata_path
        ):
            chunk_embeddings = np.load(self.chunk_embeddings_path)
            with open(self.chunk_metadata_path, "r") as f:
                chunk_metadata = json.load(f)

            if len(chunk_embeddings) == len(chunk_metadata["chunks"]):
                self.chunk_embeddings = chunk_embeddings
                self.chunk_metadata = chunk_metadata
                return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        query_embedding = self.generate_embedding(query)
        chunk_scores: list[dict] = []
        movie_scores = defaultdict(lambda: 0)
        for chunk_idx, chunk_embedding in enumerate(self.chunk_embeddings):
            metadata = self.chunk_metadata["chunks"][chunk_idx]
            midx, cidx = metadata["movie_idx"], metadata["chunk_idx"]
            sim = cosine_similarity(chunk_embedding, query_embedding)
            chunk_scores.append(
                {
                    "chunk_idx": cidx,
                    "movie_idx": midx,
                    "score": sim,
                }
            )
            movie_scores[midx] = max(movie_scores[midx], sim)
        movie_scores_sorted = sorted(
            movie_scores.items(), key=lambda x: x[1], reverse=True
        )
        formatted_results = []
        for midx, score in movie_scores_sorted[:limit]:
            doc = self.documents[midx]
            formatted_results.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "document": doc["description"][:100],
                    "score": round(score, 4),
                    "metadata": {},
                }
            )
        return formatted_results


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
    step_size = chunk_size - overlap
    for i in range(0, len(words), step_size):
        chunked_words = words[i : i + chunk_size]
        if len(chunked_words) <= overlap:
            break
        chunks.append(" ".join(chunked_words))
    return chunks


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 0) -> None:
    chunks = fixed_sized_chunking(text, chunk_size, overlap)
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")


def fixed_sized_semantic_chunking(
    text: str, max_chunk_size: int = 4, overlap: int = 0
) -> list[str]:
    stripped_text = text.strip()
    if len(stripped_text) == 0:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", stripped_text)
    if len(sentences) == 1 and (sentences[0].endswith((".", "!", "?"))):
        pass
    chunks = []
    step_size = max_chunk_size - overlap
    sentences = [sentence.strip() for sentence in sentences]
    for i in range(0, len(sentences), step_size):
        chunked_sentences = sentences[i : i + max_chunk_size]
        if len(chunked_sentences) <= overlap:
            break
        chunks.append(" ".join(chunked_sentences))
    return chunks


def semantically_chunk_text(
    text: str, max_chunk_size: int = 4, overlap: int = 0
) -> None:
    chunks = fixed_sized_semantic_chunking(text, max_chunk_size, overlap)
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")


def embed_chunks() -> np.ndarray:
    documents = load_movies()
    css = ChunkedSemanticSearch()
    embeddings = css.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")


def search_chunked(query: str, limit: int = 5) -> None:
    documents = load_movies()
    css = ChunkedSemanticSearch()
    css.load_or_create_chunk_embeddings(documents)
    results = css.search_chunks(query, limit)
    for i, result in enumerate(results):
        print(f"\n{i + 1}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['document']}...")
