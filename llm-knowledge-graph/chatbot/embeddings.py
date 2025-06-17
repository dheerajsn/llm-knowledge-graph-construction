from llama_index.core.embeddings import BaseEmbedding
from langchain_localai import LocalAIEmbeddings
from typing import List

class LocalAIEmbeddingWrapper(BaseEmbedding):
    def __init__(self, localai_embedding):
        self.localai_embedding = localai_embedding
        super().__init__()
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a single query."""
        return self.localai_embedding.embed_query(query)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        return self.localai_embedding.embed_query(text)
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        return self.localai_embedding.embed_documents(texts)
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Async version of _get_query_embedding."""
        return self._get_query_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Async version of _get_text_embedding."""
        return self._get_text_embedding(text)
    

from langchain_localai import LocalAIEmbeddings
from llama_index.core import Settings

# Configure LocalAI embeddings
localai_embeddings = LocalAIEmbeddings(
    base_url="http://localhost:8080/v1",  # Your LocalAI server URL
    model="text-embedding-ada-002",      # Your embedding model name
    api_key="not-needed"                 # LocalAI typically doesn't need real API key
)

# Wrap for LlamaIndex compatibility
wrapped_embeddings = LocalAIEmbeddingWrapper(localai_embeddings)

# Set as your embed_model
embed_model = wrapped_embeddings

# Or set globally
Settings.embed_model = wrapped_embeddings