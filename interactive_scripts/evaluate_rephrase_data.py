# %%
import pickle
import pandas as pd
import numpy as np

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

# %%
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

# %%
with open("data/saved_data/df_test", "rb") as f:  
    df_test = pickle.load(f)

with open("data/saved_data/df_test_paraphrase", "rb") as f:  
    df_test_paraphrase = pickle.load(f)

# %%
questions = df_test["question"].to_list()
paraphrased_questions = df_test_paraphrase["question"].to_list()

# %%
scores = []

for i in range(len(questions)):
    question_embeddings = embedder.embed_query(questions[i])
    paraphrased_question_embeddings = embedder.embed_query(paraphrased_questions[i])

    score = cosine_similarity(
        np.array(question_embeddings).reshape(1, -1), 
        np.array(paraphrased_question_embeddings).reshape(1, -1),
    )
    scores.append(score)

sum(scores) / len(scores)

# %%
df = pd.DataFrame({
    "Question": questions,
    "Rephrased Question": paraphrased_questions,
    "score": scores, 
})

df = df.sort_values("score")

# df[["Question", "Rephrased Question"]].to_csv("data/saved_data/rephased_questions.csv", index=False)
df

# %%
plt.hist([score[0][0] for score in scores])
plt.xlabel("Semantic Score")
plt.ylabel("Frequency")
plt.title("Distribution of semantic scores")
plt.rcParams["figure.figsize"] = (7, 4)
