# %%
import os
import glob
import re
import httpx
from dotenv import load_dotenv
from tika import parser

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer

load_dotenv()

# API_KEY = os.environ.get("OPENAI_API_KEY")
API_KEY = os.environ.get("AALTO_OPENAI_API_KEY")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")

# %%
def update_base_url(request: httpx.Request):
  if request.url.path == "/chat/completions":
    request.url = request.url.copy_with(path="/v1/openai/gpt4-turbo/chat/completions")

llm = ChatOpenAI(
    default_headers={
        "Ocp-Apim-Subscription-Key": API_KEY
    },
    base_url="https://aalto-openai-apigw.azure-api.net",
    api_key=None,
    http_client=httpx.Client(
    event_hooks={
        "request": [update_base_url],
    }),
)

llm.invoke(["Hello!"])

# %%
# llm_ollama = ChatOllama(
#     temperature=0, 
#     model="llama3.1:8b", 
#     format="json",
#     num_ctx=10000,
# )

graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
)

# llm = ChatOpenAI(
#     temperature=0.5, 
#     model="gpt-4o-mini", 
#     api_key=API_KEY,
# )

llm_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Concept", "Explanation", "Formula", "Resource"],
    allowed_relationships=[
        ("Concept", "RELATED_TO", "Concept"),
        ("Concept", "ILLUSTRATED_BY", "Explanation"),
        ("Concept", "ILLUSTRATED_BY", "Formula"),
        ("Resource", "SUPPORTS", "Concept"),
    ],
    node_properties=["description"],
)

# %%
def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\n+', '\n', text)  # Replace multiple newlines with a single newline
    text = re.sub(r'\s+', ' ', text)   # Replace multiple spaces with a single space
    text = re.sub(r'p\.\d+/\d+', '', text)
    text = re.sub(r'\b\d+(?:\.\d+)?(?:\s+\d+(?:\.\d+)?)*\b', '', text)  
    return text.strip()

files = glob.glob("./data/course_materials/data-mining/lecture-3/*.pdf", recursive=True)

for file_path in files:
    raw_xml = parser.from_file(file_path, xmlContent=True)
    body = raw_xml['content'].split('<body>')[1].split('</body>')[0]
    body_without_tag = body.replace("<p>", "").replace("</p>", "").replace("<div>", "").replace("</div>","").replace("<p />","")
    text_pages = body_without_tag.split("""<div class="page">""")[1:]

text_pages = [clean_text(page) for page in text_pages]

text_pages

# %%
from langchain_core.documents import Document

documents = [Document(page_content=page) for page in text_pages]
graph_documents = llm_transformer.convert_to_graph_documents(documents)

print(graph_documents)

# %%
for doc in graph_documents:
    print(doc.nodes)
    print(doc.relationships)
    print()

# %%
graph.add_graph_documents(graph_documents)

# %%
