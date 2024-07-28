# Q-RAG

## Overview

Retrieval Augmented Generation (RAG) has gained popularity as a method to incorporate non-parametric memory space to LLM agents, allowing them to gain access to more knowledge while avoiding the computational overhead from fine-tuning. RAG operates by retrieving relevant documents and augmenting them as the grounding context to every user input, thereby helping to enhance LLM's response accuracy and relevance. This is particularly powerful in knowlege-intensive tasks such as question answering.

However, RAG systems may fail to obtain relevant context when human queries become too complex, resulting in poor answers. The core problem is that semantic search engines rely on the contextual similarity between the question and the documents rather than verifying the presence of answers. As a result, a document might be contextually relevant but lack the information needed to answer the question. In light of this, there has been a research direction aiming to develop corrective methods to address scenarios where the retrieval performance is poor. This includes utilizing web search when the retrieved context is incorrect to the query, or forming a query rewriting loop until context retrieval is satisfied, etc. However, in a sense, most of these are not truly "corrective":

- Methods similar to utilizing knowledge base extension such as web search is not truly corrective. They merely allow agents to gain access to a larger knowledge base, which clearly can still result in bad context retrieval.
- Generally, these methods will go through another "corrective process" everytime the same complex question (or similar variants of it) is provided. In other words, they will try to solve the same problem again rather than leaving some kind of trajectory for future similar problems to follow, which can be quite inefficient.

To deal with these issues, we introduce Q-RAG, a corrective RAG paradigm helping to address the core problem of context retrieval. 

#### General Idea of Q-RAG:

Q-RAG revolves around using "model questions" for context retrieval. These "model questions" are examples of questions that can be answered by certain chunks. To express this relationships, the model questions are linked to chunks containing the answer. During the retrieval process, new input questions can then be matched with similar model questions, returning the connected chunks. 

<div align="center">
  <img src="https://github.com/Hungreeee/Q-RAG/blob/main/images/q-rag-idea.png" width=65%>
</div>

Q-RAG allows user questions to be directly matched with examples of questions that could already be answered. Intuitively, this approach is better than solely relying on generic chunk retrieval, which may fail to obtain high-quality chunks when the question becomes too complex. 

#### Corrective Process:

<div align="center">
  <img src="https://github.com/Hungreeee/Q-RAG/blob/main/images/q-rag-process.png">
</div>

With this unique retrieval process, Q-RAG allows poor answers to be corrected. Here is one scenario of how this might play out:

- The RAG retriever fails to retrieve relevant context from the database. 
- Poor context leads to poor answers. These cases can be detected by some evaluation method such as metric tracking, poor user rating, etc. 
- An external source (e.g., human evaluator) provides the corrected context chunks. This can be achieved by providing corrected answer, then use it to search for relevant chunks (using semantic search, BM-25, etc.). The intuition is that utilizing the ground truth answers (instead of the query) for retrieval will result in chunks with a considerably higher chance of containing true answers. 
- Connect the initial question to the corrected chunks, then append them to the database.
- Next time, when users ask similar questions, the retriever can match the input question with the new model question, returning the connected chunks.

In a realistic QnA setting, new questions would start to appear less with time. As such, with Q-RAG, the situation will eventually somewhat converge to the scenario where every complex question can be handled by this method. Moreover, with a simple approach, Q-RAG can easily be integrated into any existing RAG pipeline.

## Demo

A small Q-RAG demo developed using langchain can be found in this [notebook](https://github.com/Hungreeee/Q-RAG/blob/main/demo/demo_notebook_squad.ipynb).

## Run Q-RAG

Nothing here yet...

### Installation

### Environment set up

### Run Neo4j instance

### Prepare dataset

### Run evaluation 

## Contributions

Nothing here yet...
