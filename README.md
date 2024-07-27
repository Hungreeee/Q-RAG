# Q-RAG

## Overview

Retrieval Augmented Generation (RAG) has gained popularity as a method to incorporate non-parametric memory space to LLM agents, allowing them to gain access to more knowledge while avoiding the computational overhead from fine-tuning. RAG operates by retrieving relevant documents and augmenting them as the grounding context to every user input, thereby helping to enhance LLM's response accuracy and relevance. This is particularly powerful in knowlege-intensive tasks such as question answering.

However, whether LLMs can accurately answer a question is highly dependent on the quality of the retrieved documents. RAG systems, however, may fail to obtain relevant context when human queries become too complex, resulting in poorly-formed answers. The core problem lies in the way semantic searching is being used: matching human queries with documents does not necessarily mean that those documents will contain details related to the answers. As such, there has been a research direction to improve the context retrieval process by developing advanced corrective methods to address scenarios where the retrieval performance is poor. This includes utilizing web search when the retrieved context is incorrect to the query, or forming a query rewriting loop until context retrieval is satisfied, etc. However, such methods are not truly "corrective" in a sense:

- Methods similar to utilizing knowledge base extension such as web search is not truly corrective. They merely allow agents to gain access to a larger knowledge base, which clearly can still result in bad context retrieval.
- Generally, these methods will go through the very same "advanced retrieval process" everytime the same complex question (or similar variants of it) is provided. In other words, they will try to solve the very same problem again, which can be highly inefficient.

To deal with these issues, we introduce Q-RAG, a corrective RAG paradigm helping to address the core problem of context retrieval. Q-RAG utilizes a very simple approach and can be integrated into any existing RAG pipeline.

#### General Idea of Q-RAG:

Q-RAG revolves around using "model questions" for context retrieval besides chunks. To explain, these "model questions" are examples of questions that can be answered by the chunks. During the retrieval process, the input question can be matched with these model questions, returning relevant chunks connected to them. 

<div align="center">
  <img src="https://github.com/Hungreeee/Q-RAG/blob/main/images/q-rag-idea.png" width=65%>
</div>

This approach allows user questions to be directly matched with examples of questions that could already be answered, which is better than solely relying on generic chunk retrieval, which may fail to obtain high-quality chunks when the question becomes too complex. 

#### Corrective Process:

<div align="center">
  <img src="https://github.com/Hungreeee/Q-RAG/blob/main/images/q-rag-process.png">
</div>

The architecture of Q-RAG allows answer correction to take place, which works as following:

- The RAG retriever did not manage to retrieve relevant context from the database. 
- Poor context leads to poor answers. These cases can be detected by some evaluation method such as metric tracking, poor user rating, etc. 
- An external source (e.g., human evaluator) provides the corrected context chunks. This can be done simply by providing corrected answer, then use it to search for relevant chunks (using semantic search, BM-25, etc.). The intuition is that utilizing the ground truth answers for retrieval will result in chunks that have a considerably higher chance to contain the true answers. 
- Connect the initial question to the corrected chunks, then add them to the database.
- Next time, when users asks similar questions, the retriever can match the input question with the new model question, returning chunks with answers.

To justify for this process, the intuition is that, in a realistic QnA setting, new questions would start to appear less with time. As such, the situation will eventually somewhat converge to the scenario where every complex question can be corrected by this method. 

## Demo

A small Q-RAG demo written with langchain can be found in this [notebook](https://github.com/Hungreeee/Q-RAG/blob/main/demo/demo_notebook_squad.ipynb). It demonstrates how question-chunk connections can be simply replicated using metadata.

## Run Q-RAG

Nothing here yet...

### Installation

### Environment set up

### Run Neo4j instance

### Prepare dataset

### Run evaluation 

## Contributions

Nothing here yet...
