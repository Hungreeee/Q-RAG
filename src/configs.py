from typing import List
from dataclasses import dataclass, field


@dataclass
class RetrieverConfig:
    chunk_size: int = 1500
    chunk_overlap: int = 1000
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    device: str = "cpu"
    collection_name: str = "default"

    @classmethod
    def default(cls):
        return cls()


@dataclass
class LLMConfig:
    model: str = "gpt-4-turbo"
    temperature: int = None
    top_p: int = None

    @classmethod
    def default(cls):
        return cls()


@dataclass
class RAGConfig:
    top_k: int = 5
    system_message: str = """
    You are a question-answering chatbot acting as a virtual teaching assistant. Your role is to help students by providing accurate and concise answers based on the given context.
    - Use the provided documents to formulate your response.
    - Do not mention the presence of these documents in your response. Just extract relevant facts you can use to build up your answer.
    - Respond with "no answer" if you cannot find enough information to answer a question.
    """

    @classmethod
    def default(cls):
        return cls()
