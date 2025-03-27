# %%
import glob
import pickle
from tqdm import tqdm
from collections import defaultdict

import pandas as pd
from dotenv import load_dotenv

from typing import List

from qdrant_client.models import Filter, FieldCondition, MatchValue

from src.llm_agent import AzureAIAgent, LLMJudge, update_base_url
from src.retriever import QdrantRetriever
from src.rag_pipeline import RAGPipeline, QRAGPipeline
from src.metrics import ContextRecall, ContextPrecision
from src.configs import LLMConfig

from deepeval import evaluate 
from deepeval.evaluate import EvaluationResult
from deepeval.metrics import FaithfulnessMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

load_dotenv()

# %%
def preprocess_data(df):
    """Preprocess QnA dataframes"""
    # Remove reasoning questions (not within the scope of our project)
    df = df[~df["question_type"].isin(["multi reasoning", "single reasoning", "no answer"])]

    # Construct ground-truth chunk list
    df["answer_spans"] = df[[f"answer_span_{idx}" for idx in range(1, 6)]].values.tolist()
    df["answer_spans"] = df["answer_spans"].apply(lambda row: [x for x in row if pd.notna(x)])
    df["answer_spans"] = df["answer_spans"].apply(lambda row: [x.replace("\r", " ") for x in row])

    # Remove questions without evidence to support
    df = df[~((pd.isna(df["answer_span_1"])) & (df["question_type"] != "no answer"))]

    # Drop redundant columns
    df = df.drop([f"reasoning_step_{idx}" for idx in range(1, 6)], axis="columns")
    df = df.drop([f"answer_span_{idx}" for idx in range(1, 6)], axis="columns")

    df["chunks"] = df["answer_spans"].apply(lambda row: [])
    df = df.reset_index(drop=True)
    return df


def preprocess_curate_chunks(retriever: QdrantRetriever, df_dataset: pd.DataFrame, truncation_threshold=5):
    """Converts the clues given as answer spans into actual chunks"""
    df_dataset_ = df_dataset.copy()

    for idx, row in tqdm(df_dataset_.iterrows(), total=len(df_dataset_)):
        ground_truth_clues = row["answer_spans"]
        ground_truth_chunks = []
        syllabus = row["syllabus_name"]

        syllabus_filter = Filter(
            must=[
                FieldCondition(key="metadata.syllabus", match=MatchValue(value=syllabus)),
            ]
        )

        for clue in ground_truth_clues:
            chunk = retriever.retrieve_exact(clue, syllabus_filter)
            while not chunk and len(clue.split()) > truncation_threshold:
                clue = clue[:len(clue) // 2]
                chunk = retriever.retrieve_exact(clue, syllabus_filter)
            ground_truth_chunks.extend(chunk)

        ground_truth_chunks = list({chunk.metadata["chunk_id"]: chunk for chunk in ground_truth_chunks}.values())
        df_dataset_.at[idx, "chunks"] = ground_truth_chunks

    df_dataset_ = df_dataset_[df_dataset_["chunks"].apply(len) > 0]
    df_dataset_ = df_dataset_.reset_index(drop=True)
    return df_dataset_


def construct_answer_dataset(rag_pipeline: RAGPipeline, df_dataset: pd.DataFrame):
    """Collect answers into a dataset for evaluation"""
    df_dataset_ = df_dataset.copy()
    answer_list = []

    for idx, row in tqdm(df_dataset_.iterrows(), total=len(df_dataset_)):
        syllabus = row["syllabus_name"]
        question = row["question"]
        ground_truth_answer = row["answer"]
        ground_truth_chunks = row["chunks"]
        ground_truth_chunks_string = [chunk.page_content for chunk in ground_truth_chunks]

        # Execute the query
        answer, retrieved_chunks = rag_pipeline.query(question, syllabus)

        # Process retrieved chunks to remove duplicates by 'chunk_id'
        retrieved_chunks = list({chunk.metadata["chunk_id"]: chunk for chunk in retrieved_chunks}.values())
        retrieved_chunks_string = [chunk.page_content for chunk in retrieved_chunks]

        # Add the result to the answer list
        answer_list.append(LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=ground_truth_answer,
            context=ground_truth_chunks_string,
            retrieval_context=retrieved_chunks_string,
            additional_metadata={
                "syllabus": syllabus,
                "raw_ground_truth_chunks": ground_truth_chunks,
                "raw_retrieved_chunks": retrieved_chunks,
            }
        ))

    return answer_list


def construct_rephrased_dataset(llm_agent: AzureAIAgent, df_dataset: pd.DataFrame):
    """Construct rephrased dataset based on original dataset"""
    rephrased_dataset = df_dataset.copy()
    question = []

    for idx, row in tqdm(rephrased_dataset.iterrows(), total=len(rephrased_dataset)):
        syllabus = row["syllabus_name"]
        question = row["question"]
        ground_truth_answer = row["answer"]

        paraphrase_messages = [
            ("system", """
            You are provided with a QUESTION of a student. Your task is to transform the QUESTION into a REPHRASED QUESTION by rephrasing it differently.
            * Do a strong paraphrasing. 
            * Do not add any redundant text to your response besides the rephrased question. For example, do not add the "REPHRASED QUESTION" text to your response.
            * Make sure to leave enough information in the REPHRASED QUESTION so that the ANSWER to it do not change.
            """),
            ("user", f"""
            QUESTION: {question}
            ANSWER: {ground_truth_answer}
            """)
        ]
        
        response = llm_agent.generate(paraphrase_messages)
        rephrased_dataset.at[idx, "question"] = response.content

    return rephrased_dataset


def construct_qrag_train_set(result_df: EvaluationResult, answer_dataset: List[LLMTestCase]):
    """Construct learning dataset for Q-RAG loop"""
    train_set = []
    result_df_sorted = sorted(result_df.test_results, key=lambda x: int(x.name.split("_")[-1]))

    for idx, test_result in enumerate(result_df_sorted):
        if not test_result.success:
            question = test_result.input
            ground_truth_chunks = answer_dataset[idx].additional_metadata["raw_ground_truth_chunks"]
            syllabus = answer_dataset[idx].additional_metadata["syllabus"]
            ground_truth_chunks_ids = [chunk.metadata["chunk_id"] for chunk in ground_truth_chunks]

            train_set.append({
                "question": question,
                "syllabus": syllabus,
                "related_chunks": ground_truth_chunks_ids,
            })

    return train_set


def calculate_mean_metrics(result_df: EvaluationResult):
    """Calculate mean metrics"""
    mean_scores = defaultdict(list)

    for result in result_df:
        for metric in result.metrics_data:
            mean_scores[metric.name].append(metric.score)

    mean_scores_dict = {metric: sum(scores) / len(scores) for metric, scores in mean_scores.items()}
    return mean_scores_dict

# %%
# Read context documents
files = glob.glob("./data/SyllabusQA/syllabi/**/*.txt", recursive=True)

documents = []

for idx, file_path in enumerate(files):
    with open(file_path, "r", encoding="latin-1") as f:
        raw_text = f.read()

    course_name = file_path.split("\\")[-1][:-4].strip()

    documents.append({
        "page_content": raw_text,
        "parent_id": idx,
        "syllabus": course_name,
        "source": file_path,
    })

documents

# %%
# Read QA test set
df_test = pd.read_csv("./data/SyllabusQA/data/dataset_split/test.csv")
df_test = preprocess_data(df_test)

df_train = pd.read_csv("./data/SyllabusQA/data/dataset_split/train.csv")
df_train = preprocess_data(df_train)

df_val = pd.read_csv("./data/SyllabusQA/data/dataset_split/val.csv")
df_val = preprocess_data(df_val)

# %%
df_test

# %%
# Set up models
chunk_retriever = QdrantRetriever(collection_name="chunks")
question_retriever = QdrantRetriever(collection_name="questions")

llm_agent = AzureAIAgent()

naive_rag_pipeline = RAGPipeline(
    retriever=chunk_retriever,
    llm_generator=llm_agent,
)

qrag_pipeline = QRAGPipeline(
    question_retriever=question_retriever,
    chunk_retriever=chunk_retriever,
    llm_generator=llm_agent,
)

# %%
# Refresh database
# chunk_retriever.reset()
# chunk_retriever.ingest(documents)

# question_retriever.reset()

# %%
# Convert clues into chunks
df_test = preprocess_curate_chunks(chunk_retriever, df_test)
df_test

# %%
# zero_iter_answer_dataset = construct_answer_dataset(naive_rag_pipeline, df_test)

# with open("data/saved_data/zero_iter_answer_dataset", "wb") as f:  
#     pickle.dump(zero_iter_answer_dataset, f)

with open("data/saved_data/zero_iter_answer_dataset", "rb") as f:  
    zero_iter_answer_dataset = pickle.load(f)

zero_iter_answer_dataset

# %%
# Define metrics
llm_judge_config = LLMConfig(
    model="gpt-4o-mini",
    temperature=0.
)
llm_judge = LLMJudge(config=llm_judge_config)

context_recall = ContextRecall(threshold=0.8)

context_precision = ContextPrecision(threshold=0.8)

faithfulness = FaithfulnessMetric(model=llm_judge, threshold=0.5)

correctness = GEval(
    name="Correctness",
    model=llm_judge,
    threshold=0.5,
    evaluation_steps=[
        "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
        "You should also heavily penalize omission of detail",
        "You should not penalize redundant details in 'actual output'",
        "Vague language, or contradicting OPINIONS, are OK",
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT, 
        LLMTestCaseParams.ACTUAL_OUTPUT, 
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
)

# %%
# Run evaluation pipeline (zero-iteration)
# zero_iter_result_df = evaluate(
#     zero_iter_answer_dataset, 
#     metrics=[
#         context_recall,
#     ], 
#     ignore_errors=False,
#     show_indicator=True,
# )

# with open("data/saved_data/zero_iter_result_df", "wb") as f:  
#     pickle.dump(zero_iter_result_df, f)

with open("data/saved_data/zero_iter_result_df", "rb") as f:  
    zero_iter_result_df = pickle.load(f)

zero_iter_result_df.test_results

# %%
# Detect poor cases based on metrics and prompt Q-RAG to learn them
# qrag_train_set = construct_qrag_train_set(zero_iter_result_df, zero_iter_answer_dataset)
# qrag_pipeline.loop_qrag(qrag_train_set, force_replace=True)

# %%
# df_test_paraphrase = construct_rephrased_dataset(llm_agent, df_test)

# with open("data/saved_data/df_test_paraphrase", "wb") as f:  
#     pickle.dump(df_test_paraphrase, f)

with open("data/saved_data/df_test_paraphrase", "rb") as f:  
    df_test_paraphrase = pickle.load(f)

df_test_paraphrase

# %%
# Obtain results from Q-RAG
# qrag_answer_dataset = construct_answer_dataset(qrag_pipeline, df_test_paraphrase)

# with open("data/saved_data/qrag_answer_dataset", "wb") as f:  
#     pickle.dump(qrag_answer_dataset, f)

with open("data/saved_data/qrag_answer_dataset", "rb") as f:  
    qrag_answer_dataset = pickle.load(f)

qrag_answer_dataset

# %%
# Run evaluation pipeline (Q-RAG)
# qrag_result_df = evaluate(
#     qrag_answer_dataset, 
#     metrics=[
#         faithfulness,
#         correctness,
#         context_recall,
#         context_precision,
#     ], 
#     ignore_errors=False,
#     show_indicator=True,
# )

# with open("data/saved_data/qrag_result_df", "wb") as f:  
#     pickle.dump(qrag_result_df, f)

with open("data/saved_data/qrag_result_df", "rb") as f:  
    qrag_result_df = pickle.load(f)

qrag_result_df.test_results

# %% Obtain results from RAG
# naive_rag_answer_dataset = construct_answer_dataset(naive_rag_pipeline, df_test_paraphrase)

# with open("data/saved_data/naive_rag_answer_dataset", "wb") as f:  
#     pickle.dump(naive_rag_answer_dataset, f)

with open("data/saved_data/naive_rag_answer_dataset", "rb") as f:  
    naive_rag_answer_dataset = pickle.load(f)

naive_rag_answer_dataset

# %%
# Run evaluation pipeline (RAG)
# naive_rag_result_df = evaluate(
#     naive_rag_answer_dataset, 
#     metrics=[
#         faithfulness,
#         correctness,
#         context_recall,
#         context_precision,
#     ], 
#     ignore_errors=False,
#     show_indicator=True,
# )

# with open("data/saved_data/naive_rag_result_df", "wb") as f:  
#     pickle.dump(naive_rag_result_df, f)

with open("data/saved_data/naive_rag_result_df", "rb") as f:  
    naive_rag_result_df = pickle.load(f)

naive_rag_result_df.test_results

# %%
calculate_mean_metrics(qrag_result_df.test_results)

# %%
calculate_mean_metrics(naive_rag_result_df.test_results)

# %%
calculate_mean_metrics(zero_iter_result_df.test_results)
