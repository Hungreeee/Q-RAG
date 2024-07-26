import argparse
import sys

from core.database import Neo4jDatabase
from core.clients import OllamaClient, OpenAIClient
from core.rag_agent import RAGAgent


def main(args):
    test_set_path = args.test_set_path
    corpus_path = args.test_set_path
    client = args.client
    top_k = args.top_k
    threshold = args.threshold
    run_name = args.run_name

    if client == "openai":
        client = OpenAIClient()
    elif client == "ollama":
        client = OllamaClient()
    
    database = Neo4jDatabase(
        host="bolt://localhost:7687",
        user="neo4j",
        password="admin@123",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )

    rag_agent = RAGAgent(client, database)

    # Write results generation + save into folder here


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default="../datasets/", help='Dataset file')
    parser.add_argument('--prediction_path', type=str, default="../results/prediction/", help='Prediction file')
    parser.add_argument('--evaluation_path', type=str, default="../results/evaluation/", help='Evaluation output file')
    parser.add_argument("--client", required=True, type=str)
    parser.add_argument("--top_k", default=10, type=int)
    parser.add_argument("--threshold", default=0.0, type=float)
    parser.add_argument("--run_id", required=True, type=str)

    args = parser.parse_args()
    main(args)