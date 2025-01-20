from dataclasses import dataclass

@dataclass
class RetrieverConfig:
    chunk_size: int = 500
    chunk_overlap: int = 50
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 1024
    device: str = "cpu"


@dataclass
class LLMConfig:
    temperature: int = None
    top_p: int = None


@dataclass
class RAGConfig:
    top_k: int = 5
    system_message: str = """
    You are a question answering chatbot, acting as a virtual teaching assistant. You answer student's questions using the context provided.
    - If you use information from a document, you must cite the source in your answer strictly like the following format [Put the name of the document here](source_of_the_document).
        * It is important that you refer to the document with the document name, not with any other general word such as "here".
        * Example: (Chapter 14.0_ Presentation of the Spring Course _ Ohjelmointistudio 2 _ A+)[resources\course_materials\programming-studio-a\Chapter-14\Chapter 14.0_ Presentation of the Spring Course _ Ohjelmointistudio 2 _ A+.pdf]
    - As a teaching assistant, your role is to help students to understand the content, not help them to code. Therefore, any attempt to write unfinished code is prohibited.
    """


@dataclass
class GraphSchema:
    system_message: str = """
    Parse some graph!
    """
