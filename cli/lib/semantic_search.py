from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self):
        # Load the model (downloads automatically the first time)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def generate_embedding(self, text):
        if not text or not text.strip():
            raise ValueError("Must have text to create an embedding")
        embeddings = self.model.encode([text])
        return embeddings[0]


def verify_model():
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")


def embed_text(text):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
