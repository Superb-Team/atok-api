"""
MCP Agent Helper - Functions to create MCP client and agent
Connects to GitHub MCP server via Streamable HTTP
"""

from strands import Agent
from strands.tools.mcp import MCPClient
from mcp.client.streamable_http import streamablehttp_client
from model import llm
import os
from dotenv import load_dotenv

load_dotenv()

MCP_AGENT_SYSTEM_PROMPT = """You are a specialized GitHub operations assistant using Model Context Protocol (MCP).

Your expertise includes:
- Repository operations (search, read, create)
- Issue management (create, read, update, close)
- Pull request operations
- File operations (read, write, update)
- User and organization queries

Always use the appropriate MCP tools to help users with their GitHub needs.
Be technical but accessible, and provide clear operation confirmations!
"""

def create_mcp_agent():
    """
    Create MCP client for GitHub MCP server connection.
    Must be used within a context manager.
    
    Returns:
        MCPClient instance
    """
    github_pat = os.getenv('GITHUB_PAT')
    if not github_pat:
        raise ValueError("GITHUB_PAT environment variable not set")
    
    # Create MCP client for GitHub
    mcp_client = MCPClient(
        lambda: streamablehttp_client(
            url="https://api.githubcopilot.com/mcp/",
            headers={"Authorization": f"Bearer {github_pat}"}
        )
    )
    
    return mcp_client


def get_mcp_agent_with_tools(mcp_client):
    """
    Get MCP Agent with tools loaded from MCP server.
    Must be called within MCP client context.
    
    Args:
        mcp_client: Active MCPClient instance
    
    Returns:
        Agent configured with MCP tools
    """
    # Get tools from MCP server
    tools = mcp_client.list_tools_sync()
    
    # Create agent with MCP tools
    agent = Agent(
        name="MCPAgent",
        model=llm,
        system_prompt=MCP_AGENT_SYSTEM_PROMPT,
        tools=tools
    )
    
    return agent
