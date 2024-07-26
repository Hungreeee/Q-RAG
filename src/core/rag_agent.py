from langchain_core.prompts import ChatPromptTemplate

from clients import BaseClient
from database import Neo4jDatabase
from data_classes import Question, Chunk

from prompts.rag_agent import answer_with_context_prompt

import random

from typing import List, Dict, Any

CHUNK_INDEX_NAME = "chunk_similarity"
QUESTION_INDEX_NAME = "question_similarity"
RELA_NAME = "HAS_ANSWER"


class RAGAgent():
    def __init__(
        self,
        client: BaseClient,
        database: Neo4jDatabase
    ):
        self.llm = client
        self.database = database


    def retrieve_context(
            self, 
            query: str, 
            top_k: int, 
            top_k_question: int = 1, 
            threshold: float = 0.0
    ):
        """
        Retrieve and handle the raw retrieval result from Neo4j
        """

        raw_context = self.database.retrieve_similar_nodes(query, top_k, CHUNK_INDEX_NAME)
        raw_q_context = self.database.retrieve_similar_nodes_neighbours(query, RELA_NAME, top_k_question, QUESTION_INDEX_NAME)

        raw_context = [node for node in raw_context if node["score"] >= threshold]
        # Only take 1 question
        raw_q_context = [node["neighbours"] for node in raw_q_context if node["score"] >= threshold][0]

        raw_context_str = self.construct_context(raw_context)
        raw_q_context_str = self.construct_context(raw_q_context)

        combined_context = f"{raw_context_str}\n\n{raw_q_context_str}"
        return combined_context


    def generate_response(
        self, 
        question: str, 
        context: str,
        **krawgs
    ):
        """
        Generate response to user question and cprovided ontext
        """

        chat_message = answer_with_context_prompt.invoke({
            "question": question,
            "context": context
        })

        response = self.llm.run(chat_message, **krawgs)
        return response
    

    def loop_qrag(
        self, 
        ground_truth_answer: str,
        question_info: Dict[Any], 
        top_k: int,
        threshold: float = 0.0
    ):
        """
        Retrieve the relevant docs given the ground truth answer and augment the question-chunk connections to the database
        """

        chunk_info_list = self.database.retrieve_similar_nodes(
            query=ground_truth_answer, 
            top_k=top_k, 
            index_name="Chunk", 
            threshold=threshold
        )

        question_node = Question(
            id=question_info["id"], 
            metadata={
                "text": question_info["text"]
            }
        )

        self.database.add_node(question_node)
        
        for chunk_info in chunk_info_list:
            chunk_node = self.database.get_node_with_id(chunk_info["id"], "Chunk")
            self.database.add_relationship(question_node, chunk_node, "HAS_ANSWER")
        
    
    @staticmethod
    def construct_context(chunk_node_list: List[str]):
        """
        Construct context string from provided chunk nodes 
        """

        doc_list = []
        for node in chunk_node_list:
            if "header" in node:
                doc_list.append("Title: " + node["header"] + "\n" + node["text"])
            else: 
                doc_list.append(node["text"])
    
        doc_list = random.shuffle(doc_list)
        context_str = "\n\n".join(doc_list)

        return context_str