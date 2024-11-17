import bibtexparser
import csv
import os

def extract_bib_info(file_path):
    with open(file_path, 'r') as bib_file:
        bib_database = bibtexparser.load(bib_file)
        entries = bib_database.entries
        extracted_data = []

        for entry in entries:
            doi = entry.get('doi', '').replace('https://doi.org/', '')
            title = entry.get('title', '')
            url = entry.get('url', '')
            author = entry.get('author', '')
            year = entry.get('year', '')
            extracted_data.append({
                'DOI': doi,
                'Title': title,
                'URL': url,
                'Author': author,
                'Year' : year
            })
        
        return extracted_data

def save_to_csv(data, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['DOI', 'Title', 'URL', 'Author', 'Year']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in data:
            writer.writerow(row)

def main(root_folder, output_file):
    all_data = []
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.bib'):
                file_path = os.path.join(dirpath, filename)
                all_data.extend(extract_bib_info(file_path))
    
    save_to_csv(all_data, output_file)
    
root_folder = '../Data/Metadata_open_access'
output_file = '../Data/Metadata_open_access/extracted_bib_info_open_access.csv'
main(root_folder, output_file)
