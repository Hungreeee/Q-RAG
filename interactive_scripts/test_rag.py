# %%
import glob
import uuid
from tika import parser

from src.llm_agent import OpenAIAgent
from src.retriever import QdrantRetriever
from src.rag_pipeline import RAGPipeline, QRAGPipeline

from qdrant_client import QdrantClient, models
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny 

# %%
# Read .pdf into database
# ! IMPORTANT: Run this only for the first time to ingest data into database
files = glob.glob("./data/course_materials/statistical-nlp/**/*.pdf", recursive=True)

documents = []

count = 1

for file_path in files:
    raw_text = parser.from_file(file_path)["content"].splitlines()
    raw_text = "\n".join([line for line in raw_text if line != ""])
    documents.append({
        "page_content": raw_text,
        "parent_id": count,
        "source": file_path,
    })
    count += 1

# %%
chunk_retriever = QdrantRetriever(collection_name="chunks")
question_retriever = QdrantRetriever(collection_name="questions")

llm_agent = OpenAIAgent()

rag_pipeline = RAGPipeline(
    retriever=chunk_retriever,
    llm_generator=llm_agent,
)

qrag_pipeline = QRAGPipeline(
    question_retriever=question_retriever,
    chunk_retriever=chunk_retriever,
    llm_generator=llm_agent,
)

# %%
chunk_retriever.ingest([documents[0]])

# %%
# Ingest question - Answer pairs
qrag_pipeline.loop_qrag({
        "How are you?": ["1:9"],
        "What is up?": ["1:17"],
    },
    force_replace=False,
)

# %%
question_retriever.retrieve("*")

# %%
qrag_pipeline.query("How are you?")
