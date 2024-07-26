import argparse, os

from core.database import Neo4jDatabase
from core.data_classes import Node
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents.base import Document

from dotenv import load_dotenv
load_dotenv()


def read_datasets(directory_path: str):
    loader = DirectoryLoader(directory_path, glob="**/*.pdf")
    documents = loader.load()
    return documents


def construct_graph_documents(docs: list[Document]) -> list[Node]:
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-path", default="../datasets/", type=str)

    database = Neo4jDatabase(
        host=os.environ[""],
        user=os.environ[""],
        password=os.environ[""],
        embedding_model=os.environ[""]
    )

    database.initiate_vector_indexing(
        node_type="chunk",
        index_name="chunk_similarity",
        dimension=os.environ[""],
        similarity_metric=os.environ[""],
    )

    args = parser.parse_args()
    docs = read_datasets(args.docs_path)
    graph_docs = construct_graph_documents(docs)

    for graph_doc in graph_docs:
        database.add_node(graph_doc)
        database.generate_vector_index(graph_doc)