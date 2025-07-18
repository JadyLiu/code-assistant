import os
from typing import List, Optional
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from pathlib import Path
from agent.config import EMBEDDING_MODEL, SOURCE_CODE
from dotenv import load_dotenv


load_dotenv()
api_key = os.environ.get("MISTRAL_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")


class VectorStoreOperations:
    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self.embeddings = MistralAIEmbeddings(model=EMBEDDING_MODEL)
        self.vector_store = InMemoryVectorStore(embedding=self.embeddings)

    def load_code_and_readme_files(self) -> List[Document]:
        print(f"Loading files from: {SOURCE_CODE}")
        source_path = Path(SOURCE_CODE)
        target_extensions = [".py", ".md"]
        files = [
            f for ext in target_extensions
            for f in source_path.rglob(f"*{ext}")
            if "tests" not in f.parts
        ]

        documents = []
        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                documents.append(
                    Document(page_content=content, metadata={"source": str(file_path)})
                )
            except Exception as e:
                print(f"Failed to read {file_path}: {e}")
        return documents

    def add_documents(self, documents: List[Document]) -> None:
        try:
            self.vector_store.add_documents(documents=documents)
            print(
                f"Added {len(documents)} documents to vector store for user {self.user_id}"
            )
        except Exception as e:
            print(f"Error adding documents: {str(e)}")
            raise


class RetrievalAgent:
    def __init__(self, retriever: Optional[object]) -> None:
        self.retriever = retriever

    def retrieve(self, query: str) -> List[Document]:
        if not self.retriever:
            print("Retriever not initialized, returning empty document list.")
            return []
        try:
            # Type checking shows retriever might not have invoke method
            # We'll check at runtime and handle the AttributeError
            if hasattr(self.retriever, "invoke"):
                docs = self.retriever.invoke(query)
            else:
                print("Retriever does not have invoke method")
                return []
            print(f"Retrieved {len(docs)} documents for query: {query}")
            return docs
        except Exception as e:
            print(f"Error during retrieval for query '{query}': {e}")
            return []
