# Q-RAG: Learning from Feedback to Improve Context Retrieval

## Introduction

Retrieval Augmented Generation (RAG) has gained popularity as a method to incorporate non-parametric memory space to LLM agents, allowing them to gain access to more knowledge while avoiding the computational overhead from fine-tuning. RAG operates by retrieving relevant documents and augmenting them as grounding context to every user input, thereby helping to enhance LLM's response accuracy and relevance. However, RAG systems may fail to obtain relevant context when human queries become too complex, resulting in poor answers. The core problem is that semantic search engines rely on the contextual similarity between questions and documents rather than verifying the presence of answers. As a result, a document might be contextually relevant but lack the information needed to answer the question. 

To address these issues, we introduce Q-RAG, a novel corrective RAG paradigm that enables feedback to prevent repeating poor retrieval performance. This method has a very simple setup and can be easily adapted to any existing RAG pipelines. 

## Overview

The idea of Q-RAG is straightforward. In the database, we first create "model questions" and link them to chunks that contain their answers. During the retrieval process, Q-RAG then search for model questions similar to the input query, returning the connected chunks. This question-based approach allows user questions to be directly matched with examples of questions that could already be answered.

<div align="center">
  <img src="https://github.com/Hungreeee/Q-RAG/blob/main/images/q-rag-idea.png" width=65%>
</div>

_How are these questions created and linked to relevant chunks in the first place?_ Whenever bad retrieval is detected, the retrieval engine allows feedback from an external source, for example, a human annotator providing the corrected answers. These corrected answers can be utilized to retrieve higher-quality and more relevant chunks. These "corrected" chunks can then be linked to the original question, then appended to the database. 

<div align="center">
  <img src="https://github.com/Hungreeee/Q-RAG/blob/main/images/q-rag-process.png">
</div>

Here's one scenario of how this "corrective process" might play out:
- The RAG retriever fails to retrieve relevant context from the database. 
- Poor context leads to poor answers. These can be detected by some evaluation method such as metric tracking, low user rating, etc. 
- An external source (e.g., human evaluator) provides the corrected context chunks. This can be achieved by providing the corrected answer, then use it to search for relevant chunks (using semantic search, BM-25, etc.). The intuition is that having the ground truth answers for retrieval will result in chunks with considerably higher chances of containing the true answers. 
- Connect the initial question to the corrected chunks, then append them to the database.
- Next time, when similar questions are asked, the retriever can match the input question with the new model question, returning the connected chunks (plus chunks from vanilla RAG).

It is possible to observe that the agent can somewhat "learn" from corrected question-chunks to retrieve better when similar queries are provided next time. With this mechanism, it is possible to simulate some kind of "training" for the RAG retriever. For instance, before deploying a Q-RAG chatbot, one can start out by ingesting a model question-answering dataset on their documents into the corrective process. This enables the Q-RAG retriever to "learn" question-chunk connections for many model questions, allowing it to correctly handle similar problem patterns that may appear in the future. 

## Demo

A simple Q-RAG demo can be found in this [notebook](https://github.com/Hungreeee/Q-RAG/blob/main/demo/demo_notebook_squad.ipynb).

## Run Q-RAG

Nothing here yet...

### Installation

### Environment set up

### Run Neo4j instance

### Prepare dataset

### Run evaluation 

## Contributions

Nothing here yet...
