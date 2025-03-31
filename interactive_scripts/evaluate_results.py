# %%
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict

import pandas as pd
from dotenv import load_dotenv

from deepeval.evaluate import EvaluationResult
from deepeval.metrics import FaithfulnessMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

load_dotenv()

# %%
def score_map(scores: list):
    """Map scores based on threshold"""
    return [1 if score > 0.5 else 0 for score in scores]

def calculate_mean_metrics(result_df: EvaluationResult):
    """Calculate mean metrics"""
    mean_scores = defaultdict(list)

    for result in result_df:
        for metric in result.metrics_data:
            mean_scores[metric.name].append(metric.score)

    mean_scores_dict = {metric: sum(score_map(scores)) / len(scores) for metric, scores in mean_scores.items()}
    return mean_scores_dict

def plot_score_dist(result_df: EvaluationResult, metric_name: str):
    """Calculate mean metrics"""
    scores = []

    for result in result_df:
        for metric in result.metrics_data:
            if metric_name == metric.name:
                scores.append(metric.score)

    
    fig = plt.hist(scores)
    return fig

# %%
with open("data/saved_data/zero_iter_result_df", "rb") as f:  
    zero_iter_result_df = pickle.load(f)

zero_iter_result_df.test_results

# %%
# Obtain results from Q-RAG
with open("data/saved_data/qrag_answer_dataset", "rb") as f:  
    qrag_answer_dataset = pickle.load(f)

qrag_answer_dataset

# %%
with open("data/saved_data/qrag_result_df", "rb") as f:  
    qrag_result_df = pickle.load(f)

qrag_result_df.test_results

# %% Obtain results from RAG
with open("data/saved_data/naive_rag_answer_dataset", "rb") as f:  
    naive_rag_answer_dataset = pickle.load(f)

naive_rag_answer_dataset

# %%
with open("data/saved_data/naive_rag_result_df", "rb") as f:  
    naive_rag_result_df = pickle.load(f)

naive_rag_result_df.test_results

# %%
calculate_mean_metrics(qrag_result_df.test_results)

# %%
calculate_mean_metrics(naive_rag_result_df.test_results)

# %%
calculate_mean_metrics(zero_iter_result_df.test_results)

# %%
plot_score_dist(zero_iter_result_df.test_results, "Context Recall")

# %%
plot_score_dist(naive_rag_result_df.test_results, "Context Recall")

# %%
index = [int(doc.name.split("_")[-1]) for doc in zero_iter_result_df.test_results if doc.success == False]
plt.hist(df_test.iloc[index]["question_type"])
# len(index)

# %%
naive_rag_result_df.test_results
