from elsapy.elsclient import ElsClient
from elsapy.elsprofile import ElsAuthor, ElsAffil
from elsapy.elsdoc import FullDoc, AbsDoc
from elsapy.elssearch import ElsSearch
import json
import pandas as pd
import requests
import csv
import time
from dotenv import load_dotenv
load_dotenv(override = True)

def fetch_article_pdf(doi, api_key):
    url = f"https://api.elsevier.com/content/article/doi/{doi}?APIKey={api_key}&httpAccept=application/pdf&view=FULL"
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        print(f"Failed to fetch data for DOI: {doi}, Status Code: {response.status_code}")
        log_failed_request(doi, response.status_code)
        return None

def save_pdf(data, filename):
    with open(filename, 'wb') as file:
        file.write(data)

def log_failed_request(doi, status_code):
    with open(failed_log_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([doi, status_code])
        
api_key = os.getenv('elsevier_api_key') 
failed_log_file = '../Data/Metadata_open_access/failed_requests.csv'
doi_column = "DOI"
df = pd.read_csv("../Data/Metadata_open_access/extracted_bib_info_open_access.csv")
df_no_duplicates = df.drop_duplicates()
df_no_duplicates.reset_index(inplace = True, drop=True)
dois = df_no_duplicates[doi_column].tolist()

for doi in dois:
    pdf_data = fetch_article_pdf(doi, api_key)
    if pdf_data:
        pdf_filename = f"{doi.replace('/', '_')}.pdf"
        save_path = f"../Data/pdfs/{pdf_filename}"
        save_pdf(pdf_data, save_path)
        print(f"Saved PDF for DOI: {doi} as {pdf_filename}")
    time.sleep(10)
