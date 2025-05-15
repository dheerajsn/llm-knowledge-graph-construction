import streamlit as st
import asyncio
from agent import invoke_with_mcp

# Page configuration for better layout
st.set_page_config("Ebert", page_icon="🎙️")

# Initialize session state for messages and server selection
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm the GraphAcademy Chatbot!  How can I help you?", "tool_calls": None},
    ]
if "selected_servers" not in st.session_state:
    st.session_state.selected_servers = ["neo4j-cypher"]  # Default selection

# Available MCP servers
mcp_servers = ["neo4j-cypher", "math"]

def process_user_question(question, selected_servers=None):
    """Process user question through the agent and return the response"""
    return asyncio.run(invoke_with_mcp(question, mcp_servers=selected_servers))
    
    return "No final answer found"

def write_assistant_message(content, save=True, tool_calls=None):
    """Display a message in the Streamlit UI and optionally save it to session state"""
    with st.chat_message("assistant"):
        st.markdown(content)
        # Add info button for assistant messages with tool calls
        if tool_calls and len(tool_calls) > 0:
            with st.expander("ℹ️ Details of Tool Calls"):
                st.markdown(f"**Total Tool Calls:** {len(tool_calls)}")
                for i, (tool, result) in enumerate(tool_calls, start=1):
                    st.markdown(f"**{i}. {tool}**")
                    st.markdown("**Result:**")
                    st.code(result, language="json")
                    st.markdown("---")  # Add a separator between tool calls
    
    # Save the message to session state if required
    if save:
        save_message("assistant", content, tool_calls)

def write_user_message(content, save=True):
    """Display a user message in the Streamlit UI and optionally save it to session state"""
    with st.chat_message("user"):
        st.markdown(content)
    
    # Save the message to session state if required
    if save:
        save_message("user", content)

def save_message(role, content, tool_calls=None):
    """Save a message to the session state"""
    message_data = {"role": role, "content": content}
    if role == 'assistant' and tool_calls:
        message_data["tool_calls"] = tool_calls
    st.session_state.messages.append(message_data)

# Submit handler
def handle_submit(message):
    """
    Submit handler:

    You will modify this method to talk with an LLM and provide
    context using data from Neo4j.
    """

    # Handle the response
    with st.spinner('Thinking...'):
        # Call the agent
        ai_response, tools = process_user_question(message, st.session_state.selected_servers)
                # Handle different response formats
        write_assistant_message(ai_response, tool_calls=tools)


# CHAT CONTAINER
for message in st.session_state.messages:
    # Only pass tool_calls for assistant messages
    if message['role'] == 'assistant':
        write_assistant_message(
            message['content'], 
            save=False, 
            tool_calls=message.get('tool_calls')
        )
    if message['role'] == 'user':
        write_user_message(
            message['content'], 
            save=False  # Always None for user messages
        )

# Sidebar for server selection
st.sidebar.header("Select MCP Servers")
st.sidebar.write("Select the MCP servers you want to use:")
# Server selection using multiselect
st.session_state.selected_servers = st.sidebar.multiselect(
     options=mcp_servers,
     label="Select MCP Servers",
)
    # chat input on right
prompt = st.chat_input("What is up?")
if prompt:
    write_user_message(prompt)
    handle_submit(prompt)


