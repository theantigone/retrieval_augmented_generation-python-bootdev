from sentence_transformers import SentenceTransformer


class SemanticSearch:
	def __init__(self):
		# Load the model (downloads automatically the first time)
		self.model=SentenceTransformer('all-MiniLM-L6-v2')

def verify_model():
	search=SemanticSearch()
	print(f"Model loaded: {search.model}")
	print(f"Max sequence length: {search.model.max_seq_length}")
