import os
from mistralai import Mistral, MessageInputEntry
from typing import List, Optional, Generator, Dict, Any, AsyncGenerator
from langchain_core.documents import Document
from agent.rag import RetrievalAgent
from agent.config import CODE_MODEL


CODE_GEN_PROMPT = """
You are a coding assistant that generates Python code based on provided codebase context.

Guidelines:
- Follow the existing code patterns and style from the retrieved documents
- Use the same naming conventions, type hints, and class structures
- Generate working, tested code that integrates with the existing codebase
- If code has errors, use the code_interpreter tool to test and fix them
- Always include proper docstrings and error handling
"""


class CodeGeneratorAgent:
    def __init__(self, retrieval_agent: Optional[RetrievalAgent] = None) -> None:
        self.client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        self.agent = self.client.beta.agents.create(
            name="code_gen_agent",
            model=CODE_MODEL,
            instructions=CODE_GEN_PROMPT,
            description="Generates code based on codebase context with self-correction",
            tools=[{"type": "code_interpreter"}],
        )
        self.retrieval_agent = retrieval_agent

    def __call__(self, input_data: dict) -> dict:
        """Sync version that returns dict format."""
        query = input_data.get("query", "")
        context_docs = []

        if self.retrieval_agent:
            context_docs = self.retrieval_agent.retrieve(query)

        return self.generate_code(query, context_docs)

    async def acall(self, input_data: dict) -> dict:
        """Async version that returns dict format."""
        query = input_data.get("query", "")
        context_docs = []

        if self.retrieval_agent:
            context_docs = self.retrieval_agent.retrieve(query)

        return await self.agenerate_code(query, context_docs)

    async def agenerate_code(self, query: str, context_docs: List[Document]) -> dict:
        """Async version for backwards compatibility that returns dict format."""
        code = ""
        async for chunk in self.agenerate_code_stream(query, context_docs):
            if chunk["type"] == "content":
                code += chunk["data"]
            elif chunk["type"] == "tool_output":
                code += chunk["data"]
            elif chunk["type"] == "error":
                return {"messages": [{"role": "assistant", "content": chunk["data"]}]}

        return {
            "messages": [
                {"role": "assistant", "content": code if code else "No code generated"}
            ]
        }

    async def agenerate_code_stream(
        self, query: str, context_docs: List[Document]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Async version of generate_code_stream for Chainlit."""
        context = self._format_context(context_docs)

        prompt = f"""
Based on this codebase context:
{context}

Generate code for: {query}

Test the code using the code_interpreter tool and fix any errors.
Return only the final working code.
"""

        try:
            stream = self.client.beta.conversations.start_stream(
                agent_id=self.agent.id,
                inputs=[MessageInputEntry(role="user", content=prompt)],
                store=False,
            )

            for event in stream:
                if event.event == "message.output.delta":
                    content = getattr(event.data, "content", "")
                    if content:
                        # Handle different content types
                        if hasattr(content, "text"):
                            content_str = content.text
                        elif isinstance(content, list):
                            content_str = "".join(
                                str(item.text) if hasattr(item, "text") else str(item)
                                for item in content
                            )
                        else:
                            content_str = str(content)

                        yield {"type": "content", "data": content_str}

                elif event.event == "tool.execution.done":
                    output = getattr(event.data, "output", None)
                    if output:
                        yield {
                            "type": "tool_output",
                            "data": f"\n# Tool Output:\n{output}",
                        }

                elif event.event == "conversation.completed":
                    yield {"type": "done", "data": "Code generation completed"}

        except Exception as e:
            yield {"type": "error", "data": f"Error generating code: {str(e)}"}

    def generate_code_stream(
        self, query: str, context_docs: List[Document]
    ) -> Generator[Dict[str, Any], None, None]:
        """Generate code using agent with retrieved context, streaming response."""
        context = self._format_context(context_docs)

        prompt = f"""
Based on this codebase context:
{context}

Generate code for: {query}

Test the code using the code_interpreter tool and fix any errors.
Return only the final working code.
"""

        try:
            stream = self.client.beta.conversations.start_stream(
                agent_id=self.agent.id,
                inputs=[MessageInputEntry(role="user", content=prompt)],
                store=False,
            )

            for event in stream:
                if event.event == "message.output.delta":
                    content = getattr(event.data, "content", "")
                    if content:
                        # Handle different content types
                        if hasattr(content, "text"):
                            content_str = content.text
                        elif isinstance(content, list):
                            content_str = "".join(
                                str(item.text) if hasattr(item, "text") else str(item)
                                for item in content
                            )
                        else:
                            content_str = str(content)

                        yield {"type": "content", "data": content_str}

                elif event.event == "tool.execution.done":
                    output = getattr(event.data, "output", None)
                    if output:
                        yield {
                            "type": "tool_output",
                            "data": f"\n# Tool Output:\n{output}",
                        }

                elif event.event == "conversation.completed":
                    yield {"type": "done", "data": "Code generation completed"}

        except Exception as e:
            yield {"type": "error", "data": f"Error generating code: {str(e)}"}

    def generate_code(self, query: str, context_docs: List[Document]) -> dict:
        """Non-streaming version for backwards compatibility."""
        code = ""
        for chunk in self.generate_code_stream(query, context_docs):
            if chunk["type"] == "content":
                code += chunk["data"]
            elif chunk["type"] == "tool_output":
                code += chunk["data"]
            elif chunk["type"] == "error":
                return {"messages": [{"role": "assistant", "content": chunk["data"]}]}

        return {
            "messages": [
                {"role": "assistant", "content": code if code else "No code generated"}
            ]
        }

    def _format_context(self, docs: List[Document]) -> str:
        context = ""
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            context += (
                f"## File {i}: {source}\n```python\n{doc.page_content[:1000]}\n```\n\n"
            )
        return context
