# %%
import os
import glob
import cv2
import numpy as np
from pdf2image import convert_from_path
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"F:\Program Files\Tesseract-OCR\tesseract.exe"

# %%
# Input PDFs location
pdf_folder = "./data/course_materials/statistical-nlp/lecture-1/*.pdf"
output_folder = "./ocr_extracted_content"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Get all PDF files
files = glob.glob(pdf_folder, recursive=True)

for file_path in files:
    pdf_name = os.path.splitext(os.path.basename(file_path))[0]
    pdf_output_dir = os.path.join(output_folder, pdf_name)
    os.makedirs(pdf_output_dir, exist_ok=True)  

    images = convert_from_path(file_path, dpi=300)

    for page_num, image in enumerate(images, start=1):
        page_image_path = os.path.join(pdf_output_dir, f"page_{page_num}.png")
        image.save(page_image_path, "PNG")

        # Extract text from the page using OCR
        text = pytesseract.image_to_string(image)
        text_file_path = os.path.join(pdf_output_dir, f"page_{page_num}.txt")
        with open(text_file_path, "w", encoding="utf-8") as text_file:
            text_file.write(text)

        # Convert PIL image to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Convert to grayscale and threshold to find objects
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # Find contours (potential image regions)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        img_idx = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Ignore small regions (likely noise)
            if w * h < 5000:
                continue  

            # Crop region
            cropped_image = image_cv[y:y+h, x:x+w]

            # Run OCR to check if it's text
            extracted_text = pytesseract.image_to_string(cropped_image)

            # If OCR detects mostly empty text, save it as an image
            if len(extracted_text.strip()) < 5:  
                cropped_image_path = os.path.join(pdf_output_dir, f"page_{page_num}_img_{img_idx}.png")
                cv2.imwrite(cropped_image_path, cropped_image)
                img_idx += 1

print("OCR and image extraction complete. Text and non-text images saved in:", output_folder)


# %%
