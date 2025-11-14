from mcp.server.fastmcp import FastMCP

mcp = FastMCP()


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers and return the result."""
    return a + b


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the result."""
    return a * b


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
