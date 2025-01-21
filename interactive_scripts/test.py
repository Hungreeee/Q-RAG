# %%
import glob
import uuid
from tika import parser

from src.configs import LLMConfig, RetrieverConfig, RAGConfig
from src.llm_agent import OpenAIAgent
from src.retriever import SimpleVectorRetriever
from src.rag_pipeline import RAGPipeline

# %%
llm_config = LLMConfig()
retriever_config = RetrieverConfig()
rag_config = RAGConfig()

# %%
# Read .pdf into database
# ! IMPORTANT: Run this only for the first time to ingest data into database
files = glob.glob("./data/course_materials/statistical-nlp/**/*.pdf", recursive=True)

documents = []

for file_path in files:
    raw_text = parser.from_file(file_path)["content"].splitlines()
    raw_text = "\n".join([line for line in raw_text if line != ""])
    documents.append({
        "page_content": raw_text,
        "id": uuid.uuid4(),
        "source": file_path,
    })

retriever = SimpleVectorRetriever(config=retriever_config)
retriever.ingest(documents)

# %%
retriever = SimpleVectorRetriever(
    config=retriever_config, 
    vectorstore_path="./vectorstore/"
)

llm_agent = OpenAIAgent(config=llm_config)

rag_pipeline = RAGPipeline(
    retriever=retriever,
    llm_generator=llm_agent,
    config=rag_config,
)

# %%
retriever.retrieve("hi")

# %%
response, retrieved_documents = rag_pipeline.query("What is POS tagging?")

# %%
response
