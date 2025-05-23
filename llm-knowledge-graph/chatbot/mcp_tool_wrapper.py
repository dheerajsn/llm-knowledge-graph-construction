from langchain.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import json
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPToolWrapper(BaseTool):
    name: str = Field(...)
    description: str = Field(...)
    mcp_tool_name: str = Field(...)
    mcp_client: object = Field(...)
    
    def _run(self, tool_input: str) -> str:
        """Execute the MCP tool synchronously"""
        return asyncio.run(self._arun(tool_input))
    
    async def _arun(self, tool_input: str) -> str:
        """Execute the MCP tool asynchronously"""
        try:
            # Parse input if it's JSON
            if isinstance(tool_input, str):
                try:
                    args = json.loads(tool_input)
                except json.JSONDecodeError:
                    args = {"input": tool_input}
            else:
                args = tool_input
            
            # Call MCP tool
            result = await self.mcp_client.call_tool(
                self.mcp_tool_name, 
                args
            )
            
            return str(result.content[0].text if result.content else "No result")
        except Exception as e:
            return f"Error calling MCP tool: {str(e)}"
        
async def setup_mcp_client(server_path: str):
    """Set up MCP client connection"""
    server_params = StdioServerParameters(
        command=server_path,
        args=[]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the session
            await session.initialize()
            
            # List available tools
            tools_response = await session.list_tools()
            
            return session, tools_response.tools

async def create_mcp_tools(server_path: str):
    """Create LangChain tools from MCP server"""
    session, mcp_tools = await setup_mcp_client(server_path)
    
    langchain_tools = []
    for tool in mcp_tools:
        wrapper = MCPToolWrapper(
            name=tool.name,
            description=tool.description,
            mcp_tool_name=tool.name,
            mcp_client=session
        )
        langchain_tools.append(wrapper)
    
    return langchain_tools, session