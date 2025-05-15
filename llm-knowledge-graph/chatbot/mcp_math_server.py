import argparse
from mcp.server.fastmcp import FastMCP
import mcp.types as types
from pydantic import FileUrl
import os

mcp = FastMCP("Math")

@mcp.resource("resources://get_all_resources")
def get_resouces() -> list[types.Resource]:
    working_dir = "./working_dir"
    resources = []
    for filename in os.listdir(working_dir):
        # Create absolute path and corresponding file URL
        abs_path = os.path.abspath(os.path.join(working_dir, filename))
        file_url = FileUrl(f"file:///{abs_path}")
        resources.append(
            types.Resource(
                uri=file_url,
                name=filename,
                description=f"Resource from file {filename}",
                mimeType="text/plain"  # You can update the mimeType if needed.
            )
        )
    return resources

# Add a dynamic greeting resource
@mcp.resource("formulafile://{name}")
def get_formula_file(name) -> str:
    file_path = os.path.join("working_dir", name)
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            content = file.read()
        return content
    else:
        raise ValueError(f"File {name} does not exist in the working directory.")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a / b  # Using multiplication

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Math MCP Server")
    parser.add_argument(
        "--transport",
        type=str,
        default="sse",
        help="Transport to use (e.g., 'sse' or 'stdio'). Defaults to 'sse'."
    )
    args = parser.parse_args()
    mcp.run(transport=args.transport)