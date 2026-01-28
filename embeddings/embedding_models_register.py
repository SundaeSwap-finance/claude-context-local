"""Embedding models registry."""
from functools import partial

from embeddings.sentence_transformer import SentenceTransformerModel

# Create a partial class for the nomic model
NomicEmbeddingModel = partial(SentenceTransformerModel, model_name="nomic-ai/nomic-embed-text-v1.5")

AVAILIABLE_MODELS = {
    "nomic-ai/nomic-embed-text-v1.5": NomicEmbeddingModel,
}
