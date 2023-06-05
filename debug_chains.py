import logging
from typing import Any, Dict, List
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.docstore.document import Document
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()  # take environment variables from .env.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class DebugRetrievalQAWithSourcesChain(RetrievalQAWithSourcesChain):
    debug: bool = Field(False, description="Whether or not to print debug info.")
    delimiter: str = Field("\n=== End of Row ===\n", description="The delimiter to add at the end of each retrived document.")
    logger: Any = Field(logging.getLogger(__name__), description="Logger instance")

    def __init__(self, *args, debug=False, delimiter="\n=== End of Row ===\n", **kwargs):
        """
        Initialize the DebugRetrievalQAWithSourcesChain.

        Args:
            debug (bool): Whether or not to print debug info.
            delimiter (str): The delimiter to add at the end of each document.
        """
        super().__init__(*args, **kwargs)
        self.debug = debug
        self.delimiter = delimiter
        self.logger = logging.getLogger(__name__)

    def _get_docs(self, inputs: Dict[str, Any]) -> List[Document]:
        """
        Retrieve documents synchronously and add delimiters to each document.

        Args:
            inputs (Dict[str, Any]): The inputs.

        Returns:
            List[Document]: The retrieved documents with delimiters added.
        """
        try:
            question = inputs[self.question_key]
            docs = self.retriever.get_relevant_documents(question)
            # Add delimiters and log the documents
            modified_docs = self._add_delimiters_and_log(docs, question)
            return self._reduce_tokens_below_limit(modified_docs)
        except Exception as e:
            # Log the exception and return an empty list if there was an error
            self.logger.error("Failed to retrieve documents: %s", e)
            return []

    async def _aget_docs(self, inputs: Dict[str, Any]) -> List[Document]:
        """
        Retrieve documents asynchronously and add delimiters to each document.

        Args:
            inputs (Dict[str, Any]): The inputs.

        Returns:
            List[Document]: The retrieved documents with delimiters added.
        """
        try:
            question = inputs[self.question_key]
            docs = await self.retriever.aget_relevant_documents(question)
            # Add delimiters and log the documents
            modified_docs = self._add_delimiters_and_log(docs, question)
            return self._reduce_tokens_below_limit(modified_docs)
        except Exception as e:
            # Log the exception and return an empty list if there was an error
            self.logger.error("Failed to retrieve documents: %s", e)
            return []

    def _add_delimiters_and_log(self, docs: List[Document], question: str) -> List[Document]:
        """
        Add delimiters to the end of each document and log the contents of the documents.

        Args:
            docs (List[Document]): The documents to add delimiters to and log.
            question (str): The question that was asked.

        Returns:
            List[Document]: The documents with delimiters added.
        """
        modified_docs = []
        for doc in docs:
            # Create a new document with the delimiter added to the end
            modified_doc = Document(page_content=doc.page_content + self.delimiter, metadata=doc.metadata)
            modified_docs.append(modified_doc)
            # Log the document's content if debugging is enabled
            if self.debug:
                self.logger.info("Document retrieved for question '%s': %s%s", question, doc.page_content, self.delimiter)
        return modified_docs


