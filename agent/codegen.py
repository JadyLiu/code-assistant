import os
from mistralai import Mistral, MessageInputEntry
from typing import List, Optional, Dict, Any, AsyncGenerator
from langchain_core.documents import Document
from agent.rag import RetrievalAgent
from agent.config import CODE_MODEL


CODE_GEN_PROMPT = """
You are a coding assistant specialized in writing unit tests based on a provided codebase context.
## Instructions:
1. **Code Generation**:
   - Generate functional, well-integrated code that aligns with the existing codebase.
   - Ensure the code is fully tested and functional.
   - Ensure the generated code can be executed from any subdirectory by dynamically appending the project root to `sys.path`:
     ```python
     from pathlib import Path
     import sys
     sys.path.append(str(Path(__file__).parent.parent))
     ```
3. **Testing and Validation**:
   - Use the `code_interpreter` tool to test the generated code.
   - If errors or issues are identified, analyze the feedback, make necessary corrections, and retest.
   - Repeat the testing and correction process iteratively until the code is free of errors and fully functional.
   - Return ONLY the final, working version of the code.
4. **Code Consistency**:
   - Maintain the same naming conventions, type hints, and class structures as used in the existing codebase.
   - Ensure the code follows the project's coding standards and practices.
5. **Documentation**:
   - Include concise comments and docstrings to explain the purpose and functionality of the generated code.
"""


class CodeGeneratorAgent:
    def __init__(self, retrieval_agent: Optional[RetrievalAgent] = None) -> None:
        self.client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        self.agent = self.client.beta.agents.create(
            name="code_gen_agent",
            model=CODE_MODEL,
            instructions=CODE_GEN_PROMPT,
            description=CODE_GEN_PROMPT,
            tools=[{"type": "code_interpreter"}],
        )
        self.retrieval_agent = retrieval_agent

    def __call__(self, input_data: dict) -> dict:
        """Simple callable interface for RunnableLambda compatibility."""
        return {"messages": [{"role": "assistant", "content": "Use generate_code_stream instead"}]}

    async def generate_code_stream(
        self, query: str, context_docs: List[Document]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate code with streaming response."""
        context = self._format_context(context_docs)
        print(f"Context: {context}")
        print(query)
        prompt = f"""
{query}
This source code is provided as context for the code generation: {context}
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
                        content_str = self._extract_content_string(content)
                        yield {"type": "content", "data": content_str}

                elif event.event == "tool.execution.delta":
                    print(event)
                    output = getattr(event.data, "output", None)

                    if output:
                        yield {
                            "type": "tool_output",
                            "data": f"\n# Tool Output:\n{output}",
                        }

                elif event.event == "conversation.response.done":
                    yield {"type": "done", "data": "Code generation completed"}

        except Exception as e:
            yield {"type": "error", "data": f"Error generating code: {str(e)}"}

    def _extract_content_string(self, content) -> str:
        """Extract string content from various content types."""
        if hasattr(content, "text"):
            return content.text
        elif isinstance(content, list):
            return "".join(
                str(item.text) if hasattr(item, "text") else str(item)
                for item in content
            )
        else:
            return str(content)

    def _format_context(self, docs: List[Document]) -> str:
        """Format retrieved documents as context."""
        context = ""
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            context += (
                f"## File {i}: {source}\n```python\n{doc.page_content[:1000]}\n```\n\n"
            )
        return context