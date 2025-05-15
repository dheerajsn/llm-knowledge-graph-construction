import os
from langchain_community.document_loaders.wikipedia import WikipediaLoader
import logging
from langchain.text_splitter import CharacterTextSplitter
from langchain_neo4j import Neo4jGraph
from langchain_anthropic import ChatAnthropic
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs.graph_document import Node, Relationship
from transformers import AutoModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


llm = ChatAnthropic(
            model="claude-3-7-sonnet-latest",
            temperature=0,
        )

embedding_provider = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)

graph = Neo4jGraph(
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD')
)

doc_transformer = LLMGraphTransformer(
    llm=llm,
    )

# Set up the graph transformer
doc_transformer = LLMGraphTransformer(
    llm=llm,
)

# Load Wikipedia article on Interest Rate Swaps
logger.info("Loading Wikipedia article on Interest Rate Swaps")
loader = WikipediaLoader(
    query="Interest rate swap",
    load_max_docs=1,  # Just load the main article
    lang="en",
    doc_content_chars_max=100000,  # Adjust if needed to get full article
)
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1500,
    chunk_overlap=200,
)

docs = loader.load()
chunks = text_splitter.split_documents(docs)

for i, chunk in enumerate(chunks):

    chunk_id = f"interest_rate_swap.{i+1}"
    print("Processing -", chunk_id)
    chunk.metadata["topic"] = "Finance"
    chunk.metadata["concept"] = "Interest Rate Swap"
    chunk.metadata["source"] = "Wikipedia"

    # Embed the chunk
    chunk_embedding = embedding_provider.encode(chunk.page_content)

    # Add the Document and Chunk nodes to the graph
    properties = {
        "filename": "Wikipedia_Finance_Interest_Rate_Swap",
        "chunk_id": chunk_id,
        "text": chunk.page_content,
        "embedding": chunk_embedding.tolist(),  # Convert to list for Neo4j storage
    }
    
    graph.query("""
        MERGE (d:Document {id: $filename})
        MERGE (c:Chunk {id: $chunk_id})
        SET c.text = $text
        MERGE (d)<-[:PART_OF]-(c)
        WITH c
        CALL db.create.setNodeVectorProperty(c, 'textEmbedding', $embedding)
        """, 
        properties
    )

    # Generate the entities and relationships from the chunk
    graph_docs = doc_transformer.convert_to_graph_documents([chunk])

    # Map the entities in the graph documents to the chunk node
    for graph_doc in graph_docs:
        chunk_node = Node(
            id=chunk_id,
            type="Chunk"
        )

        for node in graph_doc.nodes:

            graph_doc.relationships.append(
                Relationship(
                    source=chunk_node,
                    target=node, 
                    type="HAS_ENTITY"
                    )
                )

    # add the graph documents to the graph
    graph.add_graph_documents(graph_docs)

# Create the vector index
graph.query("""
    CREATE VECTOR INDEX `chunkVector`
    IF NOT EXISTS
    FOR (c: Chunk) ON (c.textEmbedding)
    OPTIONS {indexConfig: {
    `vector.dimensions`: 768,
    `vector.similarity_function`: 'cosine'
    }};""")