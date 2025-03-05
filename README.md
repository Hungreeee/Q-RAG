# Educational-Knowledge-Graph-RAG

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
