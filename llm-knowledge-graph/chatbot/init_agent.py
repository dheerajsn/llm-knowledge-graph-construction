import asyncio
import json
import nest_asyncio
from typing import Type, Any, Dict
from pydantic import BaseModel, Field

# LangChain imports
from langchain.tools import BaseTool
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
from langchain.schema import AgentAction, AgentFinish
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult

# MCP imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Apply nest_asyncio for Jupyter compatibility
nest_asyncio.apply()

class MCPMathTool(BaseTool):
    """Wrapper for MCP math tools (add and multiply)"""
    name: str = Field(...)
    description: str = Field(...)
    mcp_tool_name: str = Field(...)
    mcp_client: Any = Field(...)
    
    class Config:
        arbitrary_types_allowed = True
    
    def _run(self, tool_input: str) -> str:
        """Execute the MCP tool synchronously"""
        try:
            # Create new event loop if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            return loop.run_until_complete(self._arun(tool_input))
        except Exception as e:
            return f"Error executing tool: {str(e)}"
    
    async def _arun(self, tool_input: str) -> str:
        """Execute the MCP tool asynchronously"""
        try:
            # Simple space-separated number parsing
            tool_input = tool_input.strip()
            
            # Handle space-separated numbers (most common case)
            parts = tool_input.split()
            if len(parts) == 2:
                try:
                    args = {"a": float(parts[0]), "b": float(parts[1])}
                except ValueError:
                    return f"Invalid numbers: {parts}. Please provide two valid numbers separated by space."
            else:
                # Fallback: extract any numbers from the input
                import re
                numbers = re.findall(r'-?\d+(?:\.\d+)?', tool_input)
                if len(numbers) >= 2:
                    args = {"a": float(numbers[0]), "b": float(numbers[1])}
                elif len(numbers) == 1:
                    return f"Only one number found: {numbers[0]}. Please provide TWO numbers separated by space like '5 3'."
                else:
                    return f"No numbers found. Please provide input like '5 3' to {self.mcp_tool_name} 5 and 3."
            
            # Call MCP tool
            result = await self.mcp_client.call_tool(
                self.mcp_tool_name, 
                args
            )
            
            # Return clean result
            if result.content and len(result.content) > 0:
                answer = result.content[0].text
                return str(answer)  # Just return the number result
            else:
                return "No result returned"
                
        except Exception as e:
            return f"Error: {str(e)}. Input format should be two numbers separated by space like '5 3'"

class DetailedAgentCallback(BaseCallbackHandler):
    """Callback to capture detailed agent execution information"""
    
    def __init__(self):
        self.execution_log = []
        self.current_step = {}
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        """Called when LLM starts"""
        self.current_step = {
            "type": "llm_call",
            "prompt": prompts[0] if prompts else "",
            "timestamp": asyncio.get_event_loop().time() if hasattr(asyncio, 'get_event_loop') else 0
        }
    
    def on_llm_end(self, response: LLMResult, **kwargs):
        """Called when LLM ends"""
        if self.current_step.get("type") == "llm_call":
            self.current_step["response"] = response.generations[0][0].text if response.generations else ""
            self.execution_log.append(self.current_step.copy())
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        """Called when tool starts"""
        tool_step = {
            "type": "tool_call",
            "tool_name": serialized.get("name", "unknown"),
            "input": input_str,
            "timestamp": asyncio.get_event_loop().time() if hasattr(asyncio, 'get_event_loop') else 0
        }
        self.execution_log.append(tool_step)
    
    def on_tool_end(self, output, **kwargs):
        """Called when tool ends"""
        if self.execution_log and self.execution_log[-1]["type"] == "tool_call":
            self.execution_log[-1]["output"] = str(output)
    
    def on_agent_action(self, action: AgentAction, **kwargs):
        """Called when agent takes an action"""
        action_step = {
            "type": "agent_action",
            "tool": action.tool,
            "tool_input": action.tool_input,
            "log": action.log,
            "timestamp": asyncio.get_event_loop().time() if hasattr(asyncio, 'get_event_loop') else 0
        }
        self.execution_log.append(action_step)
    
    def on_agent_finish(self, finish: AgentFinish, **kwargs):
        """Called when agent finishes"""
        finish_step = {
            "type": "agent_finish",
            "output": finish.return_values,
            "log": finish.log,
            "timestamp": asyncio.get_event_loop().time() if hasattr(asyncio, 'get_event_loop') else 0
        }
        self.execution_log.append(finish_step)
    
    def get_execution_summary(self):
        """Get a formatted summary of the execution"""
        summary = {
            "total_steps": len(self.execution_log),
            "llm_calls": len([step for step in self.execution_log if step["type"] == "llm_call"]),
            "tool_calls": len([step for step in self.execution_log if step["type"] == "tool_call"]),
            "agent_actions": len([step for step in self.execution_log if step["type"] == "agent_action"]),
            "execution_log": self.execution_log,
            "conversation_flow": self._format_conversation_flow()
        }
        return summary
    
    def _format_conversation_flow(self):
        """Format the conversation as a readable flow"""
        flow = []
        for i, step in enumerate(self.execution_log):
            if step["type"] == "llm_call":
                flow.append({
                    "step": i + 1,
                    "type": "AI Thinking",
                    "content": step.get("response", ""),
                    "prompt_snippet": step.get("prompt", "")[:200] + "..." if len(step.get("prompt", "")) > 200 else step.get("prompt", "")
                })
            elif step["type"] == "tool_call":
                flow.append({
                    "step": i + 1,
                    "type": "Tool Usage",
                    "tool": step.get("tool_name", "unknown"),
                    "input": step.get("input", ""),
                    "output": step.get("output", "")
                })
            elif step["type"] == "agent_action":
                flow.append({
                    "step": i + 1,
                    "type": "Agent Action",
                    "tool": step.get("tool", ""),
                    "input": step.get("tool_input", ""),
                    "reasoning": step.get("log", "")
                })
            elif step["type"] == "agent_finish":
                flow.append({
                    "step": i + 1,
                    "type": "Final Answer",
                    "output": step.get("output", {}),
                    "reasoning": step.get("log", "")
                })
class MCPAgentManager:
    """Manages MCP client connection and LangChain agent"""
    
    def __init__(self, mcp_server_command: str):
        self.mcp_server_command = mcp_server_command
        self.session = None
        self.agent = None
        self.tools = []
        self.callback_handler = DetailedAgentCallback()
    
    async def setup_mcp_client(self):
        """Set up MCP client connection to the math server"""
        try:
            # Setup server parameters for your fast MCP math server
            server_params = StdioServerParameters(
                command=self.mcp_server_command,
                args=[]
            )
            
            # Connect to MCP server
            self.read, self.write = await stdio_client(server_params).__aenter__()
            self.session = await ClientSession(self.read, self.write).__aenter__()
            
            # Initialize the session
            await self.session.initialize()
            
            # List available tools
            tools_response = await self.session.list_tools()
            print(f"Available MCP tools: {[tool.name for tool in tools_response.tools]}")
            
            return tools_response.tools
            
        except Exception as e:
            print(f"Error setting up MCP client: {e}")
            raise
    
    async def create_langchain_tools(self):
        """Create LangChain tools from MCP math tools"""
        mcp_tools = await self.setup_mcp_client()
        
        # Create tool wrappers for add and multiply
        tool_configs = {
            "add": {
                "name": "add",
                "description": "Adds two numbers together. Always input exactly two numbers separated by a space. Example: to add 5 and 3, input '5 3'. To add 10.5 and 2.8, input '10.5 2.8'."
            },
            "multiply": {
                "name": "multiply", 
                "description": "Multiplies two numbers together. Always input exactly two numbers separated by a space. Example: to multiply 4 and 7, input '4 7'. To multiply 3.5 and 2, input '3.5 2'."
            }
        }
        
        langchain_tools = []
        for mcp_tool in mcp_tools:
            if mcp_tool.name in tool_configs:
                config = tool_configs[mcp_tool.name]
                wrapper = MCPMathTool(
                    name=config["name"],
                    description=config["description"],
                    mcp_tool_name=mcp_tool.name,
                    mcp_client=self.session
                )
                langchain_tools.append(wrapper)
                print(f"Created LangChain tool: {config['name']}")
        
        self.tools = langchain_tools
        return langchain_tools
    
    def create_agent(self, llm_api_key: str = None):
        """Create LangChain zero-shot agent with MCP tools"""
        # Initialize LLM (you can replace with other LLMs)
        if llm_api_key:
            llm = OpenAI(temperature=0, openai_api_key=llm_api_key)
        else:
            # You can use other LLMs like Anthropic, Cohere, etc.
            from langchain.llms import FakeListLLM
            responses = [
                "I need to add these numbers. Let me use the math_add tool.",
                "I need to multiply these numbers. Let me use the math_multiply tool.",
                "The calculation is complete."
            ]
            llm = FakeListLLM(responses=responses)
        
        # Create zero-shot agent with detailed callbacks
        self.agent = initialize_agent(
            tools=self.tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            callbacks=[self.callback_handler],
            return_intermediate_steps=True,
            agent_kwargs={
                "prefix": """You are a calculator assistant with access to add and multiply tools. 
For any math calculation, immediately use the appropriate tool.

IMPORTANT: Tool input format is always two numbers separated by a space (e.g., '5 3' or '10.5 2.8').

Available tools:""",
                "suffix": """Begin! Remember: for tool inputs, always use format 'number1 number2' like '5 3'.

Question: {input}
Thought:{agent_scratchpad}"""
            }
        )
        
        return self.agent
    
    async def run_query(self, query: str):
        """Run a query through the agent and return detailed execution info"""
        try:
            # Reset callback handler for new query
            self.callback_handler = DetailedAgentCallback()
            self.agent.callback_manager.handlers = [self.callback_handler]
            
            # Run the agent
            result = self.agent.run(query)
            
            # Get detailed execution information
            execution_summary = self.callback_handler.get_execution_summary()
            
            return {
                "query": query,
                "final_answer": result,
                "execution_details": execution_summary,
                "conversation_flow": execution_summary["conversation_flow"]
            }
            
        except Exception as e:
            return {
                "query": query,
                "error": str(e),
                "execution_details": self.callback_handler.get_execution_summary() if hasattr(self, 'callback_handler') else {},
                "conversation_flow": []
            }
    
    def format_execution_report(self, result):
        """Format the execution result into a readable report"""
        if "error" in result:
            return f"Error: {result['error']}"
        
        report = []
        report.append(f"QUERY: {result['query']}")
        report.append(f"FINAL ANSWER: {result['final_answer']}")
        report.append(f"\nEXECUTION SUMMARY:")
        report.append(f"- Total steps: {result['execution_details']['total_steps']}")
        report.append(f"- LLM calls: {result['execution_details']['llm_calls']}")
        report.append(f"- Tool calls: {result['execution_details']['tool_calls']}")
        report.append(f"\nCONVERSATION FLOW:")
        
        for step in result['conversation_flow']:
            report.append(f"\nStep {step['step']}: {step['type']}")
            if step['type'] == 'AI Thinking':
                report.append(f"  Response: {step['content']}")
            elif step['type'] == 'Tool Usage':
                report.append(f"  Tool: {step['tool']}")
                report.append(f"  Input: {step['input']}")
                report.append(f"  Output: {step['output']}")
            elif step['type'] == 'Agent Action':
                report.append(f"  Tool: {step['tool']}")
                report.append(f"  Input: {step['input']}")
                report.append(f"  Reasoning: {step['reasoning']}")
            elif step['type'] == 'Final Answer':
                report.append(f"  Output: {step['output']}")
        
        return "\n".join(report)
    
    async def cleanup(self):
        """Clean up MCP connections"""
        if self.session:
            try:
                await self.session.__aexit__(None, None, None)
            except:
                pass

# Main execution functions
async def main():
    """Main function to demonstrate the integration"""
    
    # Initialize the MCP agent manager
    # Replace with your actual MCP server command/path
    mcp_server_command = "python your_mcp_math_server.py"  # or path to your server
    
    manager = MCPAgentManager(mcp_server_command)
    
    try:
        # Setup MCP tools
        print("Setting up MCP connection...")
        await manager.create_langchain_tools()
        
        # Create agent (replace with your OpenAI API key if using OpenAI)
        print("Creating LangChain agent...")
        agent = manager.create_agent()  # Add your API key here if needed
        
        # Test queries with clear expectations
        test_queries = [
            "Calculate 15 + 27",
            "What is 8 * 6?",
            "Add 3.5 and 2.8",
            "Multiply 12 by 9"
        ]
        
        print("\n" + "="*50)
        print("Testing MCP Math Tools with LangChain Agent")
        print("="*50)
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            print("-" * 50)
            result = await manager.run_query(query)
            
            # Print detailed report
            report = manager.format_execution_report(result)
            print(report)
            print("=" * 50)
    
    except Exception as e:
        print(f"Error in main execution: {e}")
    
    finally:
        # Cleanup
        await manager.cleanup()

def run_agent_sync():
    """Synchronous wrapper to run the async agent"""
    return asyncio.run(main())

# Example usage functions
# Example usage functions
async def detailed_single_query():
    """Example of running a single query with detailed output"""
    mcp_server_command = "python your_mcp_math_server.py"
    manager = MCPAgentManager(mcp_server_command)
    
    try:
        await manager.create_langchain_tools()
        agent = manager.create_agent()
        
        # Run query with detailed tracking
        result = await manager.run_query("What is 25 multiplied by 4?")
        
        # Print structured output
        print("=== DETAILED EXECUTION REPORT ===")
        print(manager.format_execution_report(result))
        
        # Access raw data if needed
        print("\n=== RAW EXECUTION DATA ===")
        print(f"LLM Calls: {result['execution_details']['llm_calls']}")
        print(f"Tool Calls: {result['execution_details']['tool_calls']}")
        
        # Get just the conversation messages
        messages = []
        for step in result['conversation_flow']:
            if step['type'] == 'AI Thinking':
                messages.append({"role": "assistant", "content": step['content']})
            elif step['type'] == 'Tool Usage':
                messages.append({"role": "tool", "tool": step['tool'], "input": step['input'], "output": step['output']})
        
        print("\n=== CONVERSATION MESSAGES ===")
        for msg in messages:
            print(f"Role: {msg['role']}")
            if msg['role'] == 'tool':
                print(f"  Tool: {msg['tool']}, Input: {msg['input']}, Output: {msg['output']}")
            else:
                print(f"  Content: {msg['content']}")
        
    finally:
        await manager.cleanup()

async def quick_math_test():
    """Quick test function for math operations with detailed output"""
    mcp_server_command = "python your_mcp_math_server.py"
    manager = MCPAgentManager(mcp_server_command)
    
    try:
        await manager.create_langchain_tools()
        agent = manager.create_agent()
        
        # Test multiple queries
        queries = ["Add 10 and 5", "Multiply 3 by 7"]
        
        for query in queries:
            print(f"\n{'='*60}")
            print(f"TESTING: {query}")
            print('='*60)
            
            result = await manager.run_query(query)
            print(manager.format_execution_report(result))
            
    finally:
        await manager.cleanup()

if __name__ == "__main__":
    print("Starting LangChain Agent with MCP Math Server Integration")
    print("Make sure your MCP math server is available!")
    
    # Run the main demo with detailed output
    run_agent_sync()
    
    # Or run detailed single query
    # asyncio.run(detailed_single_query())
    
    # Or run quick test with details
    # asyncio.run(quick_math_test())