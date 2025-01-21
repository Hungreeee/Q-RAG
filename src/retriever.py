import os
from typing import List
from abc import ABC, abstractmethod

from qdrant_client import QdrantClient, models

from langchain.schema.document import Document
from langchain_qdrant import QdrantVectorStore
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.configs import RetrieverConfig

from dotenv import load_dotenv

load_dotenv()


class BaseRetriever(ABC):
    def __init__(self, config: RetrieverConfig):
        super().__init__()

        self.config = config

        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,
            model_kwargs={"device": self.config.device},
        )

    @abstractmethod
    def retrieve(self, query: str, top_k: int):
        pass

    @abstractmethod
    def ingest(self, documents: List[str]):
        pass

    @abstractmethod
    def reset(self):
        pass


class SimpleVectorRetriever(BaseRetriever):
    def __init__(
        self, 
        config: RetrieverConfig, 
        vectorstore_path: str = None,
    ):
        super().__init__(config)

        self.vectorstore = None

        if vectorstore_path and os.path.exists(vectorstore_path):
            self.vectorstore = FAISS.load_local(
                folder_path=vectorstore_path, 
                embeddings=self.embedding_model,
                allow_dangerous_deserialization=True,
            )

    def ingest(
        self, 
        documents: List[dict], 
        vectorstore_path: str = "./vectorstore/"
    ):
        documents_langchain = []

        for doc in documents:
            metadata = {key: value for key, value in doc.items() if key != "page_content"}
            documents_langchain.append(Document(
                page_content=doc["page_content"], 
                metadata=metadata,
            ))

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size, 
            chunk_overlap=self.config.chunk_overlap,
        )
        document_chunks = text_splitter.split_documents(documents_langchain)

        self.vectorstore = FAISS.from_documents(
            documents=document_chunks, 
            embedding=self.embedding_model
        )
        self.vectorstore.save_local(vectorstore_path)
        
    def retrieve(self, query: str, top_k: int = 5):
        relevant_docs = self.vectorstore.similarity_search(query=query, k=top_k)
        return relevant_docs
    
    def reset(self):
        pass


class QdrantRetriever(BaseRetriever):
    def __init__(
        self, 
        config: RetrieverConfig, 
        base_url: str = "http://localhost:6333"
    ):
        super(config).__init__()

        self.client = QdrantClient(url=base_url)
        self._ensure_collection_exists()

        self.vectorstore = QdrantVectorStore(
            client=self.client, 
            collection_name=self.config.collection_name,
            embedding=self.embedder,
        )

    def _ensure_collection_exists(self):
        if not self.client.collection_exists(collection_name=self.config.collection_name):
            self.client.create_collection(
                collection_name=self.config.collection_name, 
                vectors_config=models.VectorParams(
                    size=self.config.embedding_dim, 
                    distance=models.Distance.COSINE,
                ),
            )

    def ingest(self, documents: List[dict]):
        documents_langchain = []

        for doc in documents:
            metadata = {key: value for key, value in doc.items() if key != "page_content"}
            documents_langchain.append(Document(
                page_content=doc["page_content"], 
                metadata=metadata,
            ))

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size, 
            chunk_overlap=self.config.chunk_overlap,
        )

        document_chunks = text_splitter.split_documents(documents_langchain)
        self.vectorstore.add_documents(document_chunks)
    
    def delete(self, node_ids: List):
        self.vectorstore.delete(node_ids)

    def reset(self):
        self.client.delete_collection(collection_name=self.config.collection_name)
        self._ensure_collection_exists()

    def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        filter: models.Filter = None,
    ):
        retrieve_documents = self.vectorstore.similarity_search(
            query=query, 
            k=top_k, 
            filter=filter,
        )
        formatted_docs = [f"Applicant ID: {str(docs.metadata['id'])}\n{docs.page_content}" for docs in retrieve_documents]
        return formatted_docs


class GraphRetriever(BaseRetriever):
    def __init__(self, config: RetrieverConfig):
        super().__init__()
        pass
    
    def ingest(self, documents: List[dict]): 
        # Task: Design graph parser engine
        pass

    def retrieve(self, query: str, top_k: int = 5): 
        # Task: Design graph retriever engine
        pass

    def reset(self): 
        pass

    