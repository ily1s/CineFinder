from sentence_transformers import SentenceTransformer

# Modèle léger, multilingual, marche avec ton PyTorch actuel
MODEL_NAME = "all-MiniLM-L6-v2"

@staticmethod
def get_embedding_model():
    return SentenceTransformer(MODEL_NAME)