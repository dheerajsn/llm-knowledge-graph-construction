from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import os

class GTTRAGEngine:
    def __init__(self, openai_api_key, persist_dir="./gtt_index_storage"):
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.persist_dir = persist_dir
        self.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
        self.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        self.index = None
        self.query_engine = None
        
    def load_index(self):
        """Load the persisted vector index"""
        try:
            storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
            self.index = load_index_from_storage(storage_context, embed_model=self.embed_model)
            
            # Create query engine
            self.query_engine = self.index.as_query_engine(
                llm=self.llm,
                similarity_top_k=5,
                response_mode="compact",
                streaming=False
            )
            
            return True
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
    
    def query(self, question):
        """Query the GTT documentation"""
        if not self.query_engine:
            if not self.load_index():
                return "Error: Could not load the index. Please create the index first."
        
        try:
            # Add context to the query for better results
            enhanced_query = f"""
            Based on the GTT application documentation, please answer the following question:
            {question}
            
            Please provide a detailed answer with specific steps or information from the documentation.
            If you mention any UI elements, features, or procedures, please be specific.
            """
            
            response = self.query_engine.query(enhanced_query)
            return str(response)
            
        except Exception as e:
            return f"Error querying the documentation: {e}"
    
    def get_relevant_chunks(self, question, top_k=3):
        """Get relevant chunks for debugging/transparency"""
        if not self.index:
            if not self.load_index():
                return []
        
        try:
            retriever = self.index.as_retriever(similarity_top_k=top_k)
            nodes = retriever.retrieve(question)
            
            chunks = []
            for node in nodes:
                chunks.append({
                    "content": node.text[:500] + "..." if len(node.text) > 500 else node.text,
                    "score": node.score,
                    "metadata": node.metadata
                })
            
            return chunks
        except Exception as e:
            print(f"Error retrieving chunks: {e}")
            return []

# Test the RAG engine
def test_rag():
    OPENAI_API_KEY = "your-openai-api-key-here"
    
    rag = GTTRAGEngine(OPENAI_API_KEY)
    
    if rag.load_index():
        # Test queries
        questions = [
            "How do I configure GTT settings?",
            "What are the main features of GTT?",
            "How to troubleshoot connection issues?"
        ]
        
        for question in questions:
            print(f"\n❓ Question: {question}")
            answer = rag.query(question)
            print(f"📝 Answer: {answer}")
            print("-" * 80)

if __name__ == "__main__":
    test_rag()