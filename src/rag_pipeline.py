from src.retriever import BaseRetriever
from src.llm_agent import BaseAgent
from src.configs import RAGConfig


class RAGPipeline:
    def __init__(
        self,
        retriever: BaseRetriever,
        llm_generator: BaseAgent,
        config: RAGConfig,
    ):
        self.retriever = retriever
        self.llm_generator = llm_generator

        self.config = config

    def rerank_results(self, documents: dict):
        # Task: Implement reciprocal reranking if needed
        pass

    def query(self, query: str):
        retrieved_documents = self.retriever.retrieve(query, top_k=self.config.top_k) 
        documents_content_string = "\n\n".join(
            f"Source: {doc.metadata['source']}\nContent: {doc.page_content}" for doc in retrieved_documents
        )

        rag_messages = [
            ("system", self.config.system_message),
            ("human", f"Question: {query}"),
            ("human", f"Context: {documents_content_string}")
        ]

        response = self.llm_generator.generate(rag_messages) 
        return response.content, retrieved_documents