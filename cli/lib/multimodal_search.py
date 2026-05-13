from lib.search_utils import load_movies
from lib.semantic_search import cosine_similarity
from PIL import Image
from sentence_transformers import SentenceTransformer


class MultimodalSearch:
    def __init__(self, documents: list[dict], model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = []
        for doc in documents:
            self.texts.append(f"{doc['title']}: {doc['description']}")
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image_fpath) -> float:
        img = Image.open(image_fpath, "r")
        return self.model.encode(img, show_progress_bar=True)

    def search_with_image(self, image_fpath, limit: int = 5) -> list[dict]:
        img = Image.open(image_fpath, "r")
        img_emb = self.model.encode(img, show_progress_bar=True)
        similarities = []
        for idx, text_emb in enumerate(self.text_embeddings):
            similarities.append((idx, cosine_similarity(text_emb, img_emb)))
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in similarities:
            _doc = self.documents[idx]
            results.append(
                {
                    "id": _doc["id"],
                    "title": _doc["title"],
                    "description": _doc["description"],
                    "sim_score": score,
                }
            )
        return results[:limit]


def verify_image_embedding(image_fpath):
    movies = load_movies()
    ms = MultimodalSearch(movies)
    embedding = ms.embed_image(image_fpath)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(image_fpath, limit: int = 5):
    movies = load_movies()
    ms = MultimodalSearch(movies)
    results = ms.search_with_image(image_fpath, limit)
    for idx, r in enumerate(results, start=1):
        print(f"{idx}. {r['title']} (similarity: {r['sim_score']:.3f})")
        print(f"   {r['description'][:100]}...\n")
