# %%
import glob
import httpx
from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv

from src.llm_agent import AzureAIAgent, LLMJudge, update_base_url
from src.retriever import QdrantRetriever
from src.rag_pipeline import RAGPipeline, QRAGPipeline

from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase

load_dotenv()

# %%
def preprocess_data(df):
    """Preprocess QnA dataframes"""
    # Remove reasoning questions (not within the scope of our project)
    df = df[~df["question_type"].isin(["multi reasoning", "single reasoning"])]

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
    return df


def preprocess_curate_chunks(retriever: QdrantRetriever, df_dataset: pd.DataFrame, truncation_threshold=5):
    """Converts the clues given as answer spans into actual chunks"""
    df_dataset_ = df_dataset.copy()

    for idx, row in tqdm(df_dataset_.iterrows()):
        ground_truth_clues = row["answer_spans"]
        ground_truth_chunks = []

        for clue in ground_truth_clues:
            chunk = retriever.retrieve_exact(clue)
            while not chunk and len(clue.split()) > truncation_threshold:
                clue = clue[:len(clue) // 2]
                chunk = retriever.retrieve_exact(clue)
            ground_truth_chunks.extend(chunk)

        ground_truth_chunks = list({chunk.metadata["chunk_id"]: chunk for chunk in ground_truth_chunks}.values())
        df_dataset_.at[idx, "chunks"] = ground_truth_chunks

    df_dataset_ = df_dataset_.reset_index(drop=True)
    return df_dataset_


def construct_answer_dataset(rag_pipeline: RAGPipeline, df_dataset: pd.DataFrame):
    """Collect answers into a dataset for evaluation"""
    df_dataset_ = df_dataset.copy()
    answer_list = []

    for _, row in tqdm(df_dataset_.iterrows()):
        syllabus = row["syllabus_name"]
        question = row["question"]
        ground_truth_answer = row["answer"]
        ground_truth_chunks = row["chunks"]
        ground_truth_chunks_string = [chunk.page_content for chunk in ground_truth_chunks]

        answer, retrieved_chunks = rag_pipeline.query(question, syllabus)
        retrieved_chunks = list({chunk.metadata["chunk_id"]: chunk for chunk in retrieved_chunks}.values())
        retrieved_chunks_string = [chunk.page_content for chunk in retrieved_chunks]

        answer_list.append(LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=ground_truth_answer,
            context=ground_truth_chunks_string,
            retrieval_context=retrieved_chunks_string,
        ))

    return answer_list


def construct_rephrased_dataset(llm_agent: AzureAIAgent, df_dataset: pd.DataFrame):
    """Construct rephrased dataset based on original dataset"""
    rephrased_dataset = df_dataset.copy()
    question = []

    for _, row in tqdm(rephrased_dataset.iterrows()):
        syllabus = row["syllabus_name"]
        question = row["question"]
        ground_truth_answer = row["answer"]
        ground_truth_chunks = row["chunks"]

        paraphrase_messages = [
            ("system", """
            You are provided with a QUESTION of a student. Your task is to transform the QUESTION into a REPHRASED QUESTION by rephrasing it differently.
            * Make sure to leave enough information in the REPHRASED QUESTION so that the ANSWER to it do not change.
            * You may use additional information such as the COURSE name to add noise to the REPHRASED QUESTION.
            """),
            ("user", f"""
            QUESTION: {question}
            COURSE: {syllabus}
            ANSWER: {ground_truth_answer}
            """)
        ]
        
        response = llm_agent.generate(paraphrase_messages)
        rephrased_dataset.at[idx, "question"] = response.content

    return rephrased_dataset

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
df_test_slice = df_test.tail(5)
df_test_slice

# %%
# Set up models
chunk_retriever = QdrantRetriever(collection_name="chunks")
question_retriever = QdrantRetriever(collection_name="questions")

llm_agent = AzureAIAgent()
llm_judge = LLMJudge()

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
# Ingest documents into database
chunk_retriever.reset()
chunk_retriever.ingest(documents)

# %%
# Convert clues into chunks
df_test_slice = preprocess_curate_chunks(chunk_retriever, df_test_slice)
df_test_slice

# %%
answer_dataset = construct_answer_dataset(naive_rag_pipeline, df_test_slice)
answer_dataset

# %%
# Metric calculation
faithfulness = FaithfulnessMetric(
    model=llm_judge,
)

context_recall = ContextualRecallMetric(
    model=llm_judge,
)

result_df = evaluate(
    answer_dataset, 
    metrics=[faithfulness, context_recall], 
    ignore_errors=False,
    show_indicator=True,
)

result_df.test_results

# %%
# Detect poor cases based on metrics and prompt Q-RAG to learn them
curated_loop_qrag_dict = {}
qrag_pipeline.loop_qrag(curated_loop_qrag_dict)

# %%
