import os
import asyncio

from llm import llm
from graph import graph
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_neo4j import Neo4jChatMessageHistory
#from langchain.agents import AgentExecutor, create_react_agent
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langgraph.checkpoint.memory import InMemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate
from langchain import hub
from utils import get_session_id
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langmem.short_term import SummarizationNode
from neo4j import GraphDatabase
from langchain.tools import StructuredTool, Tool
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from typing import Any

from tools.vector import find_chunk
from tools.cypher import run_cypher

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHECKPOINTER = InMemorySaver()

# --- PROMPT SETUP ---
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", '''You are an AI expert providing information about Neo4j and Knowledge Graphs.
                      Call tools efficiently, do not unnecessarily call tools.
                      You have access to below chat history: {chat_history}
        '''),
        ("placeholder", "{messages}"),
    ]
)

summarization_node = SummarizationNode(
    token_counter=count_tokens_approximately,
    model=llm,
    max_tokens=500,
    max_summary_tokens=128,
    output_messages_key="llm_input_messages")

class State(AgentState):
# NOTE: we're adding this key to keep track of previous summary information
# to make sure we're not summarizing on every LLM call
    context: dict[str, Any]

def get_conversation_history(session_id, limit=5):
    """
    Retrieve conversation history for a given session from Neo4j and format it for LLM.
    
    Args:
        session_id (str): The session ID to retrieve history for
        limit (int): Maximum number of messages to retrieve
    
    Returns:
        str: Formatted conversation history text
    """
    # Neo4j connection details
    uri = os.environ.get("NEO4J_URL", "bolt://localhost:7687")
    username = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "password")
    
    # Connect to Neo4j
    driver = GraphDatabase.driver(uri, auth=(username, password))
    
    # Format query with proper parameter passing
    query = """
    MATCH (s:Session)-[:LAST_MESSAGE]->(lastMessage)
    WHERE s.id = $session_id
    MATCH path = (m)-[:NEXT*0..]->(lastMessage)
    WITH DISTINCT lastMessage, nodes(path) AS allMessages
    UNWIND allMessages AS message
    WITH DISTINCT message
    ORDER BY id(message) DESC
    LIMIT $limit
    WITH message
    ORDER BY id(message)
    RETURN message
    """
    
    try:
        with driver.session() as session:
            # Execute the query with parameters
            result = session.run(query, session_id=session_id, limit=limit)
            
            # Process the results
            messages = []
            for record in result:
                message_node = record["message"]
                
                # Extract message properties
                content = message_node.get("content", "")
                msg_type = message_node.get("type", "unknown")
                
                # Format based on message type
                if msg_type.lower() == "user" or msg_type.lower() == "human":
                    messages.append(f"Human: {content}")
                elif msg_type.lower() == "ai" or msg_type.lower() == "assistant":
                    messages.append(f"AI: {content}")
                else:
                    messages.append(f"{msg_type}: {content}")
            
            # Format the conversation history
            if messages:
                history_text = "\n\n".join(messages)
                return f"Previous conversation:\n{history_text}\n"
            else:
                return "No previous conversation history."
    
    except Exception as e:
        print(f"Error retrieving conversation history: {e}")
        return "Error retrieving conversation history."
    
    finally:
        driver.close()

chats_by_session_id = {}

MCP_REGISTRY = {
            "neo4j-cypher": {
                "command": "uv",
                # Make sure to update to the full absolute path to your math_server.py file
                "args": ["run", "mcp-neo4j-cypher",
                         "--db-url", os.environ.get("NEO4J_URL", "bolt://localhost:7687"),
                         "--username", os.environ.get("NEO4J_USER", "neo4j"),
                         "--password", os.environ.get("NEO4J_PASSWORD", "password"),
                        ],
                "transport": "stdio",
            },
            "mcp-neo4j-memory": {
                "command": "uv",
                # Make sure to update to the full absolute path to your math_server.py file
                "args": ["run", "mcp-neo4j-memory",
                         "--db-url", os.environ.get("NEO4J_URL", "bolt://localhost:7687"),
                         "--username", os.environ.get("NEO4J_USER", "neo4j"),
                         "--password", os.environ.get("NEO4J_PASSWORD", "password"),
                        ],
                "transport": "stdio",
            },
            "math": {
                "command": "python",
                # Make sure to update to the full absolute path to your math_server.py file
                "args": ["/Users/dheerajnagpal/Projects/mcp_demo/src/server/mcp_math_server.py", "--transport", "stdio"],
                "transport": "stdio",
            },
        }

def get_server_subset(selected_server_names):
    """Get a subset of server configurations based on selected names"""
    return {k: MCP_REGISTRY[k] for k in selected_server_names if k in MCP_REGISTRY}


kg_chat = llm | StrOutputParser()

class CypherQuery(BaseModel):
    query: str

class FindChunk(BaseModel):
    search_term: str

class KG(BaseModel):
    message: str


# Define JSON schemas for your general tools
knowledge_graph_schema = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The query to get information about entities and relationships"
        }
    },
    "required": ["query"]
}

lesson_content_schema = {
    "type": "object",
    "properties": {
        "search_term": {
            "type": "string",
            "description": "The term to search for in the lesson content"
        }
    },
    "required": ["search_term"]
}

general_chat_schema = {
    "type": "object",
    "properties": {
        "message": {
            "type": "string",
            "description": "Your message for general knowledge graph discussion"
        }
    },
    "required": ["message"]
}

# Create your general tools with JSON args_schema
general_tools = [
    StructuredTool.from_function(
    func=run_cypher,
    name="KnowledgeGraphInformation",
    description="Use this to run Cypher queries on the knowledge graph.",
),
    StructuredTool.from_function(
        name="LessonContentSearch",
        description="For when you need to find information in the lesson content",
        func=find_chunk,
    ),
    StructuredTool.from_function(
        name="GeneralChat",
        description="For general knowledge graph chat not covered by other tools",
        func=kg_chat.invoke,  # This should be defined elsewhere in your code
    )
]

# --- AGENT PROMPT ---
# Modify the prompt to clarify the input format for tools with JSON schema
agent_prompt = PromptTemplate.from_template("""
You are a Neo4j, Knowledge graph, and generative AI expert.
Be as helpful as possible and return as much information as possible.
Only answer questions that relate to Neo4j, graphs, cypher, generative AI, or associated subjects.
        
Always use a tool and only use the information provided in the context.

TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
""")

# --- MAIN INVOKE FUNCTION ---
async def invoke_with_mcp(user_question, mcp_servers=None, session_id=None):
    server_subset = get_server_subset(mcp_servers) if mcp_servers else {}
    # Use secure session id and environment variables for credentials
    stdio_server_params = StdioServerParameters(
        command="uv",
        args=[
            "run", "mcp-neo4j-cypher",
            "--db-url", os.environ.get("NEO4J_URL", "bolt://localhost:7687"),
            "--username", os.environ.get("NEO4J_USER", "neo4j"),
            "--password", os.environ.get("NEO4J_PASSWORD", "password"),
        ]
    )

    async with MultiServerMCPClient(
            server_subset
        ) as client:
    #async with stdio_client(stdio_server_params) as (read, write):
        #async with ClientSession(read, write) as session:
        #    await session.initialize()
            all_tools = []
        #    session_tools = await load_mcp_tools(session)
            for tool in client.get_tools():
                all_tools.append(tool)  # Add MCP tools

            if general_tools:  # If additional tools exist
                all_tools.extend(general_tools)

            for t in all_tools:
                print("\n === TOOL ===")
                print(t)
                print("=== TOOL ===\n")

            session_id =  session_id or get_session_id()
            def get_chat_history(session_id: str) -> Neo4jChatMessageHistory:
                chat_history = chats_by_session_id.get(session_id)
                if chat_history is None:
                    chat_history = Neo4jChatMessageHistory(session_id=session_id, graph=graph)
                return chat_history

            chat_history_data = get_chat_history(session_id)
            def format_for_model(state):
                return chat_prompt.invoke({"messages": state["messages"], "chat_history": chat_history_data})
            
            agent = create_react_agent(llm, all_tools, state_modifier=format_for_model, pre_model_hook=summarization_node) #prompt=chat_prompt)
            chat_history = get_chat_history(session_id)
            tools = []
            try:
                humam_msg, ai_response, tools = await generate_response(agent, session_id, user_question)
                chat_history.add_user_message(humam_msg)
                chat_history.add_ai_message(ai_response)

            except Exception as e:
                logger.error(f"Agent invocation failed: {e}")
                ai_response = f"Sorry, there was an error processing your request: {str(e)}"
    return ai_response, tools

def extract_tool_calls(response):
    """Extract tool calls from the complex response structure"""
    print(f"Extracting tool calls from response... {response}")
    tool_calls = []
    
    if not isinstance(response, list):
        return tool_calls
        
    for message_idx, message in enumerate(response):  # Added enumerate to get index
        msg_type = message.__class__.__name__
        
        # Handle AI messages with tool calls
        if msg_type == 'AIMessage' and hasattr(message, 'tool_calls') and message.tool_calls:
            # Get the tool calls from this message
            for tool_call in message.tool_calls:
                tool_name = tool_call.get('name', 'Unknown Tool')
                tool_input = tool_call.get('args', {})
                
                # Find the corresponding tool result in the next message (if any)
                tool_result = "No result found"
                if message_idx < len(response) - 1:
                    next_msg = response[message_idx + 1]
                    if next_msg.__class__.__name__ == 'ToolMessage' and next_msg.tool_call_id == tool_call.get('id'):
                        tool_result = next_msg.content
                
                tool_calls.append((
                    f"{tool_name}: {tool_input}", 
                    tool_result
                ))
    print(f"Extracted tool calls from response... {tool_calls}")
    return tool_calls

def get_ai_answer(response):
    """Extract the final answer from the response"""
    print("Extracting final answer from response...")
    if not isinstance(response, list):
        return str(response)
        
    # The final answer is usually in the last AIMessage
    for message in reversed(response):
        if message.__class__.__name__ == 'AIMessage':
            # Simple content
            if isinstance(message.content, str):
                return message.content
            # Complex content with text items
            elif isinstance(message.content, list):
                return "".join([item.get('text', '') for item in message.content if item.get('type') == 'text'])
            
def get_human_msg(response):
    for message in reversed(response):
        if message.__class__.__name__ == 'HumanMessage':
            return message.content

# --- GENERATE RESPONSE FUNCTION ---
async def generate_response(chat_agent, session_id, user_input):
    """
    Calls the Conversational agent and returns a response.
    """

    agent_response = await chat_agent.ainvoke(
        {"messages": user_input, "session_id":session_id},
        config={"configurable": {"thread_id":session_id},
                "recursion_limit": 20
                }
        )
    
    human_message = get_human_msg(agent_response['messages'])
    ai_message = get_ai_answer(agent_response['messages'])
    tool_calls = extract_tool_calls(agent_response['messages'])

    return human_message, ai_message, tool_calls

# --- MAIN ENTRY POINT ---
async def main():
    try:
        user_question = "in knowledge graph, what does it say about my giving conversational memory to llm?"
        print(f"Question: {user_question}")
        
        ai_message, tool_calls = await invoke_with_mcp(user_question, mcp_servers=["math"], session_id='default_session')
        print(f"Response: {ai_message}")
        print(f"tool_calls: {tool_calls}")

        user_question = "What was my last msg?"
        print(f"Question: {user_question}")
        
        ai_message, tool_calls = await invoke_with_mcp(user_question, session_id='default_session')
        print(f"Response: {ai_message}")
        print(f"tool_calls: {tool_calls}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())