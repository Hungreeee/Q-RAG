from typing import List, Dict
from uuid import uuid4

from qdrant_client.models import Filter, FieldCondition, MatchValue

from src.retriever import BaseRetriever
from src.llm_agent import BaseAgent
from src.configs import RAGConfig


class RAGPipeline:
    def __init__(
        self,
        retriever: BaseRetriever,
        llm_generator: BaseAgent,
        config: RAGConfig = RAGConfig.default(),
    ):
        self.retriever = retriever
        self.llm_generator = llm_generator

        self.config = config

    def query(self, query: str, syllabus_filter: str):
        retrieved_documents = self.retriever.retrieve(
            query, 
            filter=Filter(
                must=[
                    FieldCondition(key="metadata.syllabus", match=MatchValue(value=syllabus_filter)),
                ]
            ), 
            top_k=self.config.top_k
        )

        documents_content_string = "\n\n".join(
            f"Source: {doc.metadata['source']}\nContent: {doc.page_content}" for doc in retrieved_documents
        )

        rag_messages = [
            ("system", self.config.system_message),
            ("human", f"Question: {query}"),
            ("human", f"Context: {documents_content_string}")
        ]

        response = self.llm_generator.generate(rag_messages) 
        
        retrieved_documents = list({chunk.metadata["chunk_id"]: chunk for chunk in retrieved_documents}.values())
        
        return response.content, retrieved_documents
    

class QRAGPipeline:
    def __init__(
        self,
        question_retriever: BaseRetriever,
        chunk_retriever: BaseRetriever,
        llm_generator: BaseAgent,
        config: RAGConfig = RAGConfig.default(),
    ):
        self.question_retriever = question_retriever
        self.chunk_retriever = chunk_retriever
        self.llm_generator = llm_generator

        self.config = config

    def loop_qrag(self, train_set: List[Dict], force_replace: bool = False):
        question_dataset = []

        for datapoint in train_set:
            question = datapoint["question"]
            syllabus = datapoint["syllabus"]
            related_chunks = datapoint["related_chunks"]

            exist_questions = self.question_retriever.retrieve_exact(
                query=question, 
                filter=Filter(
                    must=[
                        FieldCondition(key="metadata.syllabus", match=MatchValue(value=syllabus)),
                    ]
                ),
            )

            if exist_questions:
                exist_question = exist_questions[0]
                self.question_retriever.delete([exist_question.metadata["_id"]])
                question_dataset.append({
                    "page_content": question,
                    "related_chunks": list(set(exist_question.metadata["related_chunks"] + related_chunks)) if not force_replace else related_chunks,
                    "parent_id": exist_question.metadata["parent_id"],
                })
            else:
                question_dataset.append({
                    "page_content": question,
                    "related_chunks": related_chunks,
                    "parent_id": str(uuid4()),
                })

        self.question_retriever.ingest(question_dataset, chunking=False)

    def query(self, query: str, syllabus_filter: str):
        # Retrieve similar chunks
        retrieved_documents = self.chunk_retriever.retrieve(
            query, 
            filter=Filter(
                must=[
                    FieldCondition(key="metadata.syllabus", match=MatchValue(value=syllabus_filter)),
                ]
            ), 
            top_k=self.config.top_k
        )
        # Format prompt string
        documents_content = "\n\n---\n\n".join(f"Document: {doc.page_content}" for doc in retrieved_documents)

        # Retrieve similar questions
        retrieved_questions = self.question_retriever.retrieve(query, top_k=self.config.qrag_top_k) 
        question_chunk_list = []

        for question in retrieved_questions:
            # Retrieve chunks linked to questions
            question_content = question.page_content
            relevant_chunk_ids = question.metadata["related_chunks"] 
            related_documents = self.chunk_retriever.retrieve_ids(relevant_chunk_ids)
            retrieved_documents.extend(related_documents)

            # Format prompt string
            documents_content = "\n\n".join(f"Document: {doc.page_content}" for doc in related_documents)
            question_chunk_list.append(f"Example Question: {question_content}\nRelated Information:\n{documents_content}")

        question_document_content = "\n\n---\n\n".join(quest_doc for quest_doc in question_chunk_list)

        # # Format prompt
        rag_messages = [
            ("system", self.config.system_message),
            ("human", f"User Question: {query}"),
            ("human", f"Similar Answered Questions: {question_document_content}"),
            ("human", f"Other Related Documents: {documents_content}"),
        ]

        response = self.llm_generator.generate(rag_messages) 

        retrieved_documents = list({chunk.metadata["chunk_id"]: chunk for chunk in retrieved_documents}.values())

        return response.content, retrieved_documents