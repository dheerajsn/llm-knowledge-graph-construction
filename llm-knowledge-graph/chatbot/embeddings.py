from llama_index.core.embeddings import BaseEmbedding
from langchain_localai import LocalAIEmbeddings
from typing import List, Any, Optional
import asyncio

class LocalAIEmbeddingWrapper(BaseEmbedding):
    def __init__(
        self,
        base_url: str = "http://localhost:8080/v1",
        model: str = "text-embedding-ada-002", 
        api_key: str = "not-needed",
        embed_batch_size: int = 10,
        **kwargs: Any
    ):
        self._embedding_model = LocalAIEmbeddings(
            base_url=base_url,
            model=model,
            api_key=api_key
        )
        
        super().__init__(
            model_name=model,
            embed_batch_size=embed_batch_size,
            **kwargs
        )
    
    @classmethod 
    def class_name(cls) -> str:
        return "LocalAIEmbedding"
    
    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embedding_model.embed_query(query)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embedding_model.embed_query(text)
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._embedding_model.embed_documents(texts)
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        # Run in thread pool for true async behavior
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_query_embedding, query)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        loop = asyncio.get_event_loop() 
        return await loop.run_in_executor(None, self._get_text_embedding, text)
    
    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_text_embeddings, texts)


# Test compatibility
wrapper = LocalAIEmbeddingWrapper(
    base_url="http://localhost:8080/v1",
    model="your-model-name"
)

# Or set globally
from llama_index.core import Settings
Settings.embed_model = wrapper