# Q-RAG: Improving Educational QA with Query-to-query Retrieval and Feedback Loop

## Introduction

Retrieval-Augmented Generation (RAG) has emerged as a powerful technique for enhancing Large Language Models (LLMs) with access to external knowledge without requiring costly fine-tuning. By retrieving relevant documents and incorporating them into the context for each query, RAG improves the factual accuracy and relevance of responses—an especially promising approach in educational question-answering (QA) systems.

However, conventional RAG systems face two major limitations in this domain. First, the retrieval step often fails to surface the most relevant information due to semantic mismatches between natural language queries and information chunks. This can result in partially relevant but ultimately insufficient responses, which is particularly problematic in educational settings where precision is critical. Second, most RAG pipelines operate as closed systems, lacking mechanisms for expert feedback. This makes it difficult for instructors or teaching assistants to correct retrieval failures or guide the model’s future outputs—leading to repeated misinformation and a growing burden on educators.

To address these issues, we introduce Q-RAG, a novel RAG-based framework specifically designed for educational contexts. Q-RAG introduces two key features:
- *Question-Based Retrieval*: Indexing content via representative model questions to better match student queries.
- *Feedback-Loop Learning*: Integrating expert corrections to iteratively improve retrieval quality.

![qrag-structure](https://github.com/user-attachments/assets/39392036-2f7b-4475-87e0-b2f955553d75)

This project builds Q-RAG to create a more accurate, feedback-driven educational chatbot, while contributing to ongoing research in retrieval-based LLMs in closed-domain QA tasks. The directory `src` contains the core components for RAG and Q-RAG. The folder `interactive_scripts` contains the Python scripts used for result generation, evaluation, and analysis.

## Installations

To clone the project to local:
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
