import requests
import numpy as np
from typing import List, Optional
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr


class LocalAIEmbeddings(BaseEmbedding):
    """Custom embedding class for LocalAI integration with LlamaIndex."""
    
    base_url: str = Field(description="Base URL for LocalAI API")
    model_name: str = Field(description="Model name for embeddings")
    api_key: Optional[str] = Field(default=None, description="API key if required")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    
    _session: requests.Session = PrivateAttr()
    
    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: Optional[str] = None,
        timeout: int = 30,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout
        
        # Initialize session
        self._session = requests.Session()
        if self.api_key:
            self._session.headers.update({"Authorization": f"Bearer {self.api_key}"})
    
    @classmethod
    def class_name(cls) -> str:
        return "LocalAIEmbeddings"
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        url = f"{self.base_url}/v1/embeddings"
        
        payload = {
            "model": self.model_name,
            "input": text
        }
        
        try:
            response = self._session.post(
                url, 
                json=payload, 
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            embedding = data["data"][0]["embedding"]
            return embedding
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to get embedding from LocalAI: {e}")
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected response format from LocalAI: {e}")
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for text (required by BaseEmbedding)."""
        return self._get_embedding(text)
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for query (required by BaseEmbedding)."""
        return self._get_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Async version of _get_text_embedding."""
        # For simplicity, using sync version. You can implement async version with aiohttp
        return self._get_text_embedding(text)
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Async version of _get_query_embedding."""
        return self._get_query_embedding(query)


# Example usage
if __name__ == "__main__":
    from llama_index.core import VectorStoreIndex, Document
    from llama_index.core.node_parser import SentenceSplitter
    
    # Initialize LocalAI embeddings
    embed_model = LocalAIEmbeddings(
        base_url="http://localhost:8080",  # Your LocalAI endpoint
        model_name="text-embedding-ada-002",  # Your embedding model name
        # api_key="your-api-key",  # If required
    )
    
    # Test embedding
    test_text = "This is a test document for embedding."
    embedding = embed_model.get_text_embedding(test_text)
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
    
    # Use with LlamaIndex
    documents = [
        Document(text="LocalAI is a drop-in replacement REST API for OpenAI."),
        Document(text="It allows you to run LLMs locally or on-premise."),
        Document(text="LlamaIndex is a framework for building LLM applications."),
    ]
    
    # Create vector store index with custom embeddings
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model,
        transformations=[SentenceSplitter(chunk_size=512)]
    )
    
    # Query the index
    query_engine = index.as_query_engine()
    response = query_engine.query("What is LocalAI?")
    print(f"\nQuery response: {response}")