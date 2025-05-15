import os
import torch
from langchain.embeddings.base import Embeddings
from dotenv import load_dotenv
load_dotenv()

# tag::llm[]
# Create the LLM
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from transformers import AutoModel

'''llm = ChatOpenAI(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model_name="gpt-3.5-turbo"
)
# end::llm[]

# tag::embedding[]
# Create the Embedding model
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv('OPENAI_API_KEY')
)
# end::embedding[]


llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)
'''
llm = ChatAnthropic(
            model="claude-3-7-sonnet-latest",
            temperature=0,
        )


class JinaEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model
    
    def embed_query(self, text):
        # Convert the text to tensor and get embeddings
        with torch.no_grad():
            embeddings = self.model.encode(text)
            # Convert to list if it's a tensor
            if hasattr(embeddings, "tolist"):
                embeddings = embeddings.tolist()
            return embeddings
    
    def embed_documents(self, texts):
        # Handle batch embedding
        return [self.embed_query(text) for text in texts]

embeddings = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
embedding_provider = JinaEmbeddings(embeddings)
