import os
import asyncio

from llm import llm
from graph import graph
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j import Neo4jChatMessageHistory
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.prompts import PromptTemplate
from utils import get_session_id
from neo4j import GraphDatabase
from langchain.tools import StructuredTool
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from typing import Any
import traceback

from tools.vector import find_chunk

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- PROMPT SETUP ---
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", '''You are a Neo4j, Knowledge graph, and generative AI expert.
                      Be as helpful as possible and return as much information as possible.
                      Only answer questions that relate to Neo4j, graphs, cypher, generative AI, or associated subjects.
                      Always use a tool and only use the information provided in the context. But only use them if you have a reason to do so.
                      Using tools unnecessarily will be considered a bad practice.
                      You have access to below chat history: {chat_history}
        '''),
        ("placeholder", "{messages}"),
    ]
)

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
                "args": ["mcp_math_server.py", "--transport", "stdio"],
                "transport": "stdio",
            },
        }

def get_server_subset(selected_server_names):
    """Get a subset of server configurations based on selected names"""
    return {k: MCP_REGISTRY[k] for k in selected_server_names if k in MCP_REGISTRY}

# Create your general tools with JSON args_schema
general_tools = [
    StructuredTool.from_function(
        name="LessonContentSearch",
        description="For when you need to find information in the lesson content",
        func=find_chunk,
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

# Add this helper function to convert functions to proper tools
def ensure_proper_tool_format(tool_or_func):
    """Ensure a tool-like object is properly formatted with a name attribute"""
    if hasattr(tool_or_func, 'name'):
        # It's already a formatted tool
        return tool_or_func
    
    if callable(tool_or_func) and not hasattr(tool_or_func, 'name'):
        # It's a raw function, convert it to a Tool
        from langchain.tools import Tool
        func_name = tool_or_func.__name__
        func_desc = tool_or_func.__doc__ or f"Tool for {func_name}"
        return Tool.from_function(
            func=tool_or_func,
            name=func_name,
            description=func_desc
        )
    
    # If it's not callable or already has a name, return as is
    return tool_or_func

# --- MAIN INVOKE FUNCTION ---
async def invoke_with_mcp(user_question, mcp_servers=None, session_id=None):
    try:
        server_subset = get_server_subset(mcp_servers) if mcp_servers else {}
        
        async with MultiServerMCPClient(server_subset) as client:
            # Get properly formatted tools
            all_tools = []
            
            # Process MCP tools
            for tool in client.get_tools():
                # Convert MCP tools to proper format if needed
                formatted_tool = ensure_proper_tool_format(tool)
                all_tools.append(formatted_tool)
            
            # Process general tools if they exist
            if general_tools:
                for tool_item in general_tools:
                    # Skip commented tools (those that are strings)
                    if isinstance(tool_item, str):
                        continue
                    
                    # Process and add other tools
                    formatted_tool = ensure_proper_tool_format(tool_item)
                    all_tools.append(formatted_tool)
            
            # Debug output
            for t in all_tools:
                print(f"\n === TOOL: {getattr(t, 'name', 'Unknown')} ===")
                print(t)
                print("=== TOOL ===\n")
            
            # Get session and chat history
            session_id = session_id or get_session_id()
            def get_chat_history(session_id: str) -> Neo4jChatMessageHistory:
                chat_history = chats_by_session_id.get(session_id)
                if chat_history is None:
                    chat_history = Neo4jChatMessageHistory(session_id=session_id, graph=graph)
                return chat_history
            
            chat_history_data = get_chat_history(session_id)
            
            # Define format_for_model with error handling
            def format_for_model(state):
                try:
                    return chat_prompt.invoke({
                        "messages": state.get("messages", []), 
                        "chat_history": chat_history_data
                    })
                except Exception as e:
                    print(f"Error in format_for_model: {e}")
                    import traceback
                    traceback.print_exc()
                    # Return a minimal functional state to avoid crashing
                    return {"messages": state.get("messages", [])}
            
            # Create the agent with properly formatted tools
            agent = create_react_agent(llm, all_tools, state_modifier=format_for_model)
            
            chat_history = get_chat_history(session_id)
            tools = []
            
            try:
                human_msg, ai_response, tools = await generate_response(agent, session_id, user_question)
                chat_history.add_user_message(human_msg)
                chat_history.add_ai_message(ai_response)
            except Exception as e:
                logger.error(f"Agent invocation failed: {e}\n{traceback.format_exc()}")
                ai_response = f"Sorry, there was an error processing your request: {str(e)}"
                tools = []
                
        return ai_response, tools
        
    except Exception as outer_e:
        logger.error(f"Error in invoke_with_mcp: {outer_e}\n{traceback.format_exc()}")
        return f"Sorry, there was an error setting up the agent: {str(outer_e)}", []

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
        user_question = "how to use a language model to generate Cypher queries?"
        print(f"Question: {user_question}")
        
        ai_message, tool_calls = await invoke_with_mcp(user_question, mcp_servers=["math"], session_id='default_session')
        print(f"Response: {ai_message}")
        print(f"tool_calls: {tool_calls}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())