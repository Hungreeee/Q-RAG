import os
from typing import List
from configs import RetrieverConfig
from abc import ABC, abstractmethod
from langchain.schema.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()


class BaseRetriever(ABC):
    def __init__(self, config: RetrieverConfig):
        super.__init__()
        self.config = config

    @abstractmethod
    def retrieve(self, query: str, k: int):
        pass

    @abstractmethod
    def ingest(self, documents: List[str]):
        pass

    @abstractmethod
    def reset(self):
        pass


class VectorRetriever(BaseRetriever):
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,
            model_kwargs={"device": self.config.device},
        )

        self.vectorstore_path = os.getenv("VECTORSTORE_PATH")
        if os.path.exists(self.vectorstore_path):
            self.vectorstore = FAISS.load_local(
                folder_path=self.vectorstore_path, 
                embeddings=self.embedding_model,
                allow_dangerous_deserialization=True,
            )
        else:
            self.vectorstore = FAISS.from_documents([], self.embedding_model)
            self.vectorstore.save_local(self.vectorstore_path)

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
        embeddings = self.embedding_model.embed_documents(document_chunks)

        self.vectorstore.add_documents(document_chunks, embeddings)
        self.vectorstore.save_local(self.vectorstore_path)
        
    def retrieve(self, query: str, k: int):
        relevant_docs = self.vectorstore.similarity_search(query=query, k=k)
        return relevant_docs


class GraphRetriever(BaseRetriever):
    def __init__(self):
        pass
    
    def ingest(self, documents: List[dict]): 
        # Task: Design graph parser engine
        pass

    