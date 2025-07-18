import os
from typing import Optional, List, Dict, Any
import chainlit as cl
from dotenv import load_dotenv
from mistralai import Mistral
from mistralai.extra.run.context import RunContext
from mistralai.extra.mcp.sse import (
    MCPClientSSE,
    SSEServerParams,
)
from mcp import StdioServerParameters
from mistralai.extra.mcp.stdio import (
    MCPClientSTDIO,
)
from mcp import ClientSession
from pydantic import BaseModel
from agent.config import DEV_MODEL


class MCPTool(BaseModel):
    clientType: str
    url: Optional[str] = None
    command: Optional[str] = None
    args: Optional[List[str]] = None


load_dotenv()


class GitHubAgent:
    def __init__(self) -> None:
        self.MODEL = DEV_MODEL
        self.api_key = os.environ["MISTRAL_API_KEY"]
        self.client = Mistral(api_key=self.api_key)
        self.MCP_TOOLS: list[MCPTool] = []
        self.github_agent = self.client.beta.agents.create(
            model=self.MODEL,
            name="github agent",
            instructions=(
                "You are a GitHub assistant for repositories owned by JadyLiu. "
                "You can handle issues, pull requests, and other repository management tasks. "
                "The repository owner is JadyLiu. "
                "You have access to GitHub's API and can use your tools to help with tasks. "
                "Provide detailed information about GitHub features and best practices when requested."
            ),
            description=(
                "A GitHub assistant that helps manage repositories for JadyLiu. "
                "Can handle issues, pull requests, and provide GitHub-related information."
            ),
        )

    def format_messages(
        self, all_cl_messages: list[cl.Message]
    ) -> list[dict[str, str]]:
        """
        Format messages for API consumption, preserving chat history context.
        The entire conversation history is included to provide context for the agent.
        """
        api_input_list = []
        if not all_cl_messages:
            return []

        # If this isn't the first message, include previous context
        if len(all_cl_messages) > 1:
            # Add all historical messages as context, using only valid roles
            for msg in all_cl_messages[:-1]:
                # Ensure we only use 'user' or 'assistant' roles
                api_role = "user" if msg.author == "User" else "assistant"
                api_input_list.append({"role": api_role, "content": msg.content})

        # Add the current message
        current_message = all_cl_messages[-1]
        api_input_list.append({"role": "user", "content": current_message.content})
        print(api_input_list)
        return api_input_list

    async def on_mcp_connect(
        self, connection: Dict[str, Any], session: ClientSession
    ) -> None:
        # Convert the connection dict to our MCPTool model
        tool = MCPTool(**connection)
        self.MCP_TOOLS.append(tool)

    async def on_mcp_disconnect(self, name: str, session: ClientSession) -> None:
        self.MCP_TOOLS.remove(session)

    async def run(self, messages: list[cl.Message]) -> str:
        async with RunContext(agent_id=self.github_agent.id) as run_ctx:
            if self.MCP_TOOLS:
                for tool in self.MCP_TOOLS:
                    if tool.clientType == "sse":
                        temp_mcp_client = MCPClientSSE(
                            sse_params=SSEServerParams(url=tool.url, timeout=100)
                        )
                    elif tool.clientType == "stdio":
                        server_params = StdioServerParameters(
                            command=tool.command, args=tool.args, env=None
                        )
                        temp_mcp_client = MCPClientSTDIO(stdio_params=server_params)
                    await run_ctx.register_mcp_client(mcp_client=temp_mcp_client)

            server_params = StdioServerParameters(
                command="docker",
                args=[
                    "run",
                    "-i",
                    "--rm",
                    "-e",
                    "GITHUB_PERSONAL_ACCESS_TOKEN",
                    "ghcr.io/github/github-mcp-server",
                ],
                env={
                    "GITHUB_PERSONAL_ACCESS_TOKEN": os.environ[
                        "GITHUB_PERSONAL_ACCESS_TOKEN"
                    ]
                },
            )
            temp_mcp_client = MCPClientSTDIO(stdio_params=server_params)
            await run_ctx.register_mcp_client(mcp_client=temp_mcp_client)

            inputs_messages = self.format_messages(messages)
            response = await self.client.beta.conversations.run_async(
                run_ctx=run_ctx,
                inputs=inputs_messages,
            )

            output = ""
            for entry in response.output_entries:
                if hasattr(entry, "type") and entry.type == "function.call":
                    if hasattr(entry, "name") and hasattr(entry, "arguments"):
                        output += f"⚙️ Used tool: **{entry.name}**\nArguments: `{entry.arguments}`\n\n"

            if response.output_as_text:
                output += response.output_as_text

            return output     