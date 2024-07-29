# Q-RAG

## Overview

Retrieval Augmented Generation (RAG) has gained popularity as a method to incorporate non-parametric memory space to LLM agents, allowing them to gain access to more knowledge while avoiding the computational overhead from fine-tuning. RAG operates by retrieving relevant documents and augmenting them as the grounding context to every user input, thereby helping to enhance LLM's response accuracy and relevance. This is particularly powerful in knowlege-intensive tasks such as question answering.

However, RAG systems may fail to obtain relevant context when human queries become too complex, resulting in poor answers. The core problem is that semantic search engines rely on the contextual similarity between the question and the documents rather than verifying the presence of answers. As a result, a document might be contextually relevant but lack the information needed to answer the question. In light of this, there has been a research direction aiming to develop corrective methods to address scenarios where the retrieval performance is poor. This includes utilizing web search when the retrieved context is incorrect to the query, or forming a query rewriting loop until context retrieval is satisfied, etc. However, in a sense, most of these are not truly "corrective":

- Methods similar to utilizing knowledge base extension such as web search is not truly corrective. They merely allow agents to gain access to a larger knowledge base, which clearly can still result in bad context retrieval.
- Generally, these methods will go through another "corrective process" everytime the same complex question (or similar variants of it) is provided. In other words, they will try to solve the same problem again rather than leaving some kind of trajectory for future similar problems to follow, which can be quite inefficient in terms of cost and time. 

To address these issues, we introduce Q-RAG, a corrective RAG paradigm that enables external intervention (e.g., human annotators) to help prevent the system from repeating bad answers.

#### Core Mechanism:

Q-RAG revolves around using "model questions" for context retrieval. These "model questions" are examples of questions that can be answered by certain chunks. To express this relationships, the model questions are linked to chunks containing the answer. During the retrieval process, new input questions can be matched with similar model questions, returning the connected chunks. 

<div align="center">
  <img src="https://github.com/Hungreeee/Q-RAG/blob/main/images/q-rag-idea.png" width=65%>
</div>

In addition to prodive relevant chunks (using vanilla retrieval), Q-RAG allows user questions to be directly matched with examples of questions that could already be answered. This approach ensures that the LLM generator will not only recieve contextually relevant chunks but also chunks that contains answers. 

#### Corrective Process:

<div align="center">
  <img src="https://github.com/Hungreeee/Q-RAG/blob/main/images/q-rag-process.png">
</div>

With this unique retrieval process, Q-RAG allows LLM's answers to be corrected with a human-in-the-loop process. Here is one scenario of how this might play out:

- The RAG retriever fails to retrieve relevant context from the database. 
- Poor context leads to poor answers. These cases can be detected by some evaluation method such as metric tracking, poor user rating, etc. 
- An external source (e.g., human evaluator) provides the corrected context chunks. This can be achieved by providing corrected answer, then use it to search for relevant chunks (using semantic search, BM-25, etc.). The intuition is that having the ground truth answers for retrieval will result in chunks with considerably higher chances of containing the true answers. 
- Connect the initial question to the corrected chunks, then append them to the database.
- Next time, when similar questions are asked, the retriever can match the input question with the new model question, returning the connected chunks (plus chunks from vanilla RAG).

#### Benefits:

- With a simple set-up, Q-RAG is efficient in terms of cost/time and can be easily adapted to any existing RAG pipelines.
- Q-RAG allows a sort of "continual learning" mechanism by allowing humans to correct poor answers, helping the agent not making the same mistake. 
- In a realistic QnA setting, new questions would start to appear less with time. Therefore, intuitively, the situation will somewhat converge to the scenario where every complex question can be handled by Q-RAG.

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
