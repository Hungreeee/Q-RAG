# Q-RAG: Improving Educational QA with Query-to-query Retrieval and Feedback Loop

## Introduction

Retrieval-Augmented Generation (RAG) has emerged as a powerful technique for enhancing Large Language Models (LLMs) with access to external knowledge without requiring costly fine-tuning. By retrieving relevant documents and incorporating them into the context for each query, RAG improves the factual accuracy and relevance of responses—an especially promising approach in educational question-answering (QA) systems.

However, conventional RAG systems face two major limitations in this domain. 
- First, the retrieval step often fails to surface the most relevant information due to semantic mismatches between natural language queries and information chunks. Human questions are concise and target specific information, while document chunks are general statements without the goal to address any direct question. This structural difference leads to mismatching in the semantic space, making retrieval less accurate, resulting in superficially relevant but unhelpful content.
- Second, most RAG pipelines operate as closed systems, lacking mechanisms for expert feedback. This makes it difficult for instructors or teaching assistants to correct retrieval failures or guide the model’s future outputs—leading to repeated misinformation and a growing burden on educators.

To address these issues, we introduce Q-RAG, a novel RAG-based framework specifically designed for educational contexts. Q-RAG introduces two key features:
- *Query-to-query Retrieval*: Indexing document chunks via representative "model questions" to match with similar queries - essentially converting to query-to-query matching.
- *Feedback-Loop Learning*: Allowing direct intervention to address questions leading to poor retrieval, preventing them from occuring again.

![qrag-structure](https://github.com/user-attachments/assets/39392036-2f7b-4475-87e0-b2f955553d75)

This project builds Q-RAG proof-of-concept in the hope to propose a more robust RAG-based paradigms, contributing to ongoing research in retrieval-based LLMs in closed-domain QA tasks. The directory `src` contains the core components for RAG and Q-RAG. The folder `interactive_scripts` contains the Python scripts used for result generation, evaluation, and analysis. The research report is found in `Research-Report.pdf`.

## Installations and Setup

To clone the project to your local machine:
```
git clone https://github.com/Hungreeee/Educational-Knowledge-Graph-RAG.git
cd Educational-Knowledge-Graph-RAG
```

Inside the terminal of the project directory, set up a virtual environment using the `venv` package:
```
python -m venv .venv 
.venv/Scripts/activate
```

Install the required dependencies:
```
pip install -r requirements.txt
```

Make sure to download [Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/). This is important to host our database / local LLM.

After this, simply do this to start up the database:
```
docker-compose up
```

You are all set!
