import os
import uuid
from typing import List, Dict

from qdrant_client import QdrantClient, models
from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchText

from langchain.schema.document import Document
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.configs import RetrieverConfig

from dotenv import load_dotenv

load_dotenv()


class BaseRetriever:
    def __init__(
        self, 
        collection_name: str, 
        config: RetrieverConfig = RetrieverConfig.default()
    ):
        self.config = config
        self.collection_name = collection_name

        self.embedder = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,
            model_kwargs={"device": self.config.device},
        )

    def retrieve(self, query: str, top_k: int):
        raise NotImplementedError()

    def ingest(self, documents: List[Dict]):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()


class QdrantRetriever(BaseRetriever):
    def __init__(
        self, 
        collection_name: str = "default",
        config: RetrieverConfig = RetrieverConfig.default(), 
        base_url: str = "http://localhost:6333",
    ):
        super().__init__(collection_name, config)

        self.client = QdrantClient(url=base_url)
        self._ensure_collection_exists()

        self.vectorstore = QdrantVectorStore(
            client=self.client, 
            collection_name=self.collection_name,
            embedding=self.embedder,
        )

    def _ensure_collection_exists(self):
        if not self.client.collection_exists(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name, 
                vectors_config=models.VectorParams(
                    size=self.config.embedding_dim, 
                    distance=models.Distance.COSINE,
                ),
            )

    def ingest(self, documents: List[Dict], chunking: bool = True):
        documents_langchain = []
        chunk_counters = {}

        for doc in documents:
            parent_id = str(doc["parent_id"])
            metadata = {key: value for key, value in doc.items() if key != "page_content"}
            metadata["parent_id"] = parent_id
            documents_langchain.append(Document(
                page_content=doc["page_content"], 
                metadata=metadata,
            ))

        if chunking:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size, 
                chunk_overlap=self.config.chunk_overlap,
            )
            documents_langchain = text_splitter.split_documents(documents_langchain)

            for chunk in documents_langchain:
                parent_id = chunk.metadata["parent_id"]
                
                if parent_id not in chunk_counters:
                    chunk_counters[parent_id] = 0
                
                chunk.metadata["chunk_id"] = f"{parent_id}:{chunk_counters[parent_id]}"
                chunk_counters[parent_id] += 1 

        self.vectorstore.add_documents(documents_langchain)
    
    def delete(self, node_ids: List):
        self.vectorstore.delete(node_ids)

    def reset(self):
        self.client.delete_collection(collection_name=self.collection_name)
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
        return retrieve_documents
    
    def retrieve_exact(
        self, 
        query: str, 
        filter: models.Filter = None,
    ):
        base_filter = Filter(
            must=[
                FieldCondition(key="page_content", match=MatchText(text=query)),
            ]
        )

        if filter:
            base_filter.must.extend(filter.must)

        retrieve_documents = self.vectorstore.similarity_search(
            query="*", 
            k=1, 
            filter=base_filter,
        )
        return retrieve_documents

    def retrieve_ids(
        self, 
        id_list: List,
    ):
        if not id_list:
            return []
        
        retrieve_documents = self.vectorstore.similarity_search(
            query="*",
            filter=Filter(
                must=[
                    FieldCondition(key="metadata.chunk_id", match=MatchAny(any=id_list)),
                ]
            ),
            k=len(id_list),
        )
        return retrieve_documents

