# Educational-Knowledge-Graph-RAG

## Introduction

Retrieval-Augmented Generation (RAG) enhances LLMs by retrieving external documents as context, making it valuable for educational QA systems. However, traditional RAG often struggles with mismatched query-chunk semantics and lacks effective human feedback mechanisms—leading to incomplete or unhelpful answers.

Q-RAG addresses these challenges with:
- *Question-Based Retrieval*: Indexing content via representative model questions to better match student queries.
- *Feedback-Loop Learning*: Integrating expert corrections to iteratively improve retrieval quality.

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
