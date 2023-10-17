# !pip install torch transformers pandas requests PyPDF2 huggingface datasets

import prefect
from prefect import task, Flow
from prefect.engine.executors import LocalExecutor
import os

@task
def process_pdf(url):
    import io
    import os
    import PyPDF2
    import pandas as pd
    import requests
    import torch
    import json
    import csv
    import re
    import math
    from tqdm import tqdm
    from bs4 import BeautifulSoup
    import logging
    from requests.exceptions import RequestException
    from time import sleep
    from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
    from torch.utils.data import TensorDataset, DataLoader

    # Initialize a logger
    logger = logging.getLogger(__name__)

    # Set up the output directory for extracted text
    extracted_text_file = "websocket.txt"

    text_data = []

    # Get the most recent server destination address
    def get_new_dest_ip(url):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                html = response.text
                soup = BeautifulSoup(html, 'html.parser')
                download_div = soup.find("div", id="download")
                if download_div:
                    h2_element = download_div.find("h2")
                    link_tag = h2_element.find("a")
                    href = link_tag.get("href")
                    # print(href)
                    match = re.search(r'http://(\d+\.\d+\.\d+\.\d+)/', href)
                    if match:
                        ip_address = match.group(1)
                        return ip_address
                    else:
                        print("IP address not found in the URL.")
                else:
                    print("Download Div not found!")
        except Exception as e:
            print(f"An error occurred while making the request: {e}")

    # Function to extract text from a PDF URL and save it to the file
    def extract_text_from_pdf(pdf_url, extracted_text_file):
        try:
            response = requests.get(pdf_url, stream=True, timeout=30)
            response.raise_for_status()

            # Create a file-like object from the response content
            pdf_file = PyPDF2.PdfReader(io.BytesIO(response.content))

            text = ""
            total_pages = len(pdf_file.pages)

            # Use tqdm for a progress bar while iterating through pages
            with tqdm(total=total_pages, desc="Extracting", unit="page") as pbar:
                for page_num in range(total_pages):
                    # Encode the text as UTF-8 and replace problematic characters
                    page_text = pdf_file.pages[page_num].extract_text()
                    page_text = page_text.encode("utf-8", errors="replace").decode("utf-8")
                    text += page_text
                    pbar.update(1)  # Update progress bar

            # Save the extracted text to the file
            with open(extracted_text_file, "a", encoding="utf-8") as f:
                f.write(text)

            return text
        except RequestException as e:
            logger.error(f"Error while extracting text from PDF: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred: {str(e)}")
            return None

    # Load your combined CSV file with PDF links
    input_csv = "/content/API's.csv"  # Update with your CSV file name

    # Initialize the GPT-2 tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Move the model to the specified device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Set the padding token
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize lists to store tokenized input and attention masks
    input_ids = []
    attention_masks = []

    max_sequence_length = 1200

    # Adjust max_sequence_length based on the model's maximum token limit
    max_sequence_length = model.config.max_position_embeddings

    # Text extraction and cleaning
    def preprocess_text(text):
        # Remove metadata and unwanted content
        content_start = re.search(r"Chapter", text)  # Adjust this pattern as needed
        if content_start:
            text = text[content_start.start():]

        # Tokenize into paragraphs or sentences
        paragraphs = re.split(r'\n\n+|\.\s|\!\s|\?\s', text)

        cleaned_text = []
        for paragraph in paragraphs:
            # Remove special characters, non-ASCII characters, and excessive whitespace
            cleaned_paragraph = re.sub(r'[^\x00-\x7F]+', '', paragraph)
            cleaned_paragraph = re.sub(r'\s+', ' ', cleaned_paragraph)
            cleaned_paragraph = cleaned_paragraph.strip()

            # Convert to lowercase
            cleaned_paragraph = cleaned_paragraph.lower()

            # Skip empty paragraphs
            if cleaned_paragraph:
                cleaned_text.append(cleaned_paragraph)

        # Join the cleaned paragraphs back together
        cleaned_text = ' '.join(cleaned_text)

        return cleaned_text

    # Load the dataset from the CSV file
    df = pd.read_csv(input_csv)

    # Check if the extracted text file already exists, and if so, remove it
    if os.path.exists(extracted_text_file):
        os.remove(extracted_text_file)

    # Iterate through the PDF links and extract text
    for _, row in df.iterrows():
        title = row["Title"]
        pdf_url = row["PDF Link"]

        url = "http://library.lol/main/B3842962B35629FF07ACE128FDBCD1DB"
        server_address = get_new_dest_ip(url)

        # Replace the destination IP address in the PDF URL with server_address
        pdf_url = re.sub(r'http://(\d+\.\d+\.\d+\.\d+)/', f'http://{server_address}/', pdf_url)

        # Retry downloading the PDF with a maximum of 3 retries
        max_retries = 3
        for retry in range(max_retries):
            pdf_text = extract_text_from_pdf(pdf_url, extracted_text_file)

            if pdf_text is not None:
                # Preprocess the text to remove unwanted content
                preprocessed_text = preprocess_text(pdf_text)

                print(f"Title: {title}")
                print(f"Text from PDF: {preprocessed_text[:8000]}...")  # Print a portion of the extracted text
                print("-" * 50)
                text_data.append(preprocessed_text)
                break  # Successfully downloaded and processed the PDF
            else:
                # Retry after a short delay
                if retry < max_retries - 1:
                    logger.warning(f"Retrying {pdf_url} in 5 seconds...")
                    sleep(5)
                else:
                    logger.error(f"Max retries exceeded for {pdf_url}. Moving to the next URL.")
                    break  # Move to the next URL

with Flow("PDF Processing Flow") as flow:
    csv_path = "/path/to/your/csv/file.csv"  # Update with your CSV file path
    urls = task(pd.read_csv)(csv_path)
    pdf_urls = task(lambda df: df['PDF Link'].tolist())(urls)

    process_pdf.map(pdf_urls)

# Use a LocalExecutor to run tasks on your local machine
executor = LocalExecutor()
flow_state = flow.run(executor=executor)


