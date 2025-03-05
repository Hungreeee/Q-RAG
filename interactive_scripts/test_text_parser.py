# %%
import sys, os
sys.path.append("/src/")

import glob
import uuid
from pypdf import PdfReader, PdfWriter

from src.configs import LLMConfig, RetrieverConfig, RAGConfig
from src.llm_agent import OpenAIAgent
from src.retriever import SimpleVectorRetriever
from src.rag_pipeline import RAGPipeline

# %%
llm_config = LLMConfig()
retriever_config = RetrieverConfig()
rag_config = RAGConfig()

# Input PDFs location
pdf_folder = "./data/course_materials/data-mining/lecture-3/*.pdf"

# Output folder
output_folder = "./extracted_content/"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Get all PDF files
files = glob.glob(pdf_folder, recursive=True)

# Process each PDF
for file_path in files:
    reader = PdfReader(file_path)

    # Get base name for saving outputs
    pdf_name = os.path.splitext(os.path.basename(file_path))[0]
    pdf_output_dir = os.path.join(output_folder, pdf_name)
    os.makedirs(pdf_output_dir, exist_ok=True)  # Folder for this PDF

    for page_num, page in enumerate(reader.pages, start=1):
        # Create a folder for each page inside the PDF folder
        page_output_dir = os.path.join(pdf_output_dir, f"page_{page_num}")
        os.makedirs(page_output_dir, exist_ok=True)

        # Extract text and save to a text file
        text = page.extract_text()
        if text:
            text_file_path = os.path.join(page_output_dir, "text.txt")
            with open(text_file_path, "w", encoding="utf-8") as text_file:
                text_file.write(text)

        # Extract images and save
        for img_idx, image in enumerate(page.images):
            image_ext = os.path.splitext(image.name)[-1] or ".jpg"  # Default to JPG
            image_file_name = f"img_{img_idx}{image_ext}"
            image_path = os.path.join(page_output_dir, image_file_name)

            with open(image_path, "wb") as img_file:
                img_file.write(image.data)

print("Extraction complete. Text and images saved in:", output_folder)

