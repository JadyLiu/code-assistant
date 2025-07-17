import os
import chainlit as cl
from typing import List, Optional
from langchain_core.documents import Document
from agent.rag import RetrievalAgent  # Import your RetrievalAgent wrapper
from mistralai import Mistral, MessageInputEntry
from agent.config import CODE_MODEL

CODE_EXPLAIN_PROMPT = """
You are a helpful assistant that explains Python code snippets with context.

Guidelines:
- Provide clear, concise explanations of the code functionality.
- Reference the provided codebase context for accuracy.
- Explain naming conventions, logic, and any complex parts.
- Use comments and docstrings as part of your explanation.
"""


class CodeExplainerAgent:
    def __init__(self, retrieval_agent: Optional[RetrievalAgent] = None) -> None:
        self.client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        self.agent = self.client.beta.agents.create(
            name="code_explainer_agent",
            model=CODE_MODEL,
            instructions=CODE_EXPLAIN_PROMPT,
            description="Explains Python code snippets using provided context",
        )
        self.retrieval_agent = retrieval_agent

    async def __call__(self, input_data: dict) -> dict:
        query = input_data.get("query", "")
        context_docs = []

        if self.retrieval_agent:
            async with cl.Step(name="Retrieving Context", type="retrieval") as step:
                context_docs = self.retrieval_agent.retrieve(query)

                # Display the context documents
                if context_docs:
                    context_summary = (
                        f"Found {len(context_docs)} relevant documents:\n\n"
                    )
                    for i, doc in enumerate(context_docs, 1):
                        source = doc.metadata.get("source", "unknown")
                        preview = doc.page_content
                        context_summary += f"**Document {i}:** `{source}`\n```python\n{preview}\n```\n\n"

                    step.output = context_summary
                else:
                    step.output = "No relevant context documents found."

        return self.explain_code(query, context_docs)

    def explain_code(self, query: str, context_docs: List[Document]) -> dict:
        """Generate code explanation using agent with retrieved context."""
        context = self._format_context(context_docs)

        prompt = f"""
Based on this codebase context:
{context}

Explain the following code or concept: {query}
"""

        print(f"Prompt: {prompt}")
        stream = self.client.beta.conversations.start_stream(
            agent_id=self.agent.id,
            inputs=[MessageInputEntry(role="user", content=prompt)],
            store=True,
        )

        explanation = ""
        for event in stream:
            if hasattr(event.data, "content") and event.data.content:
                explanation += event.data.content

        return {"messages": [{"role": "assistant", "content": explanation}]}

    def _format_context(self, docs: List[Document]) -> str:
        context = ""
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            context += (
                f"## File {i}: {source}\n```python\n{doc.page_content[:1000]}\n```\n\n"
            )
        return context
