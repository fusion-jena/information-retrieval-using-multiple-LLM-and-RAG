import glob
import re
import pandas as pd
import os
from datetime import datetime
from helper_functions import load_cqs, read_txt
from langchain_core.prompts import PromptTemplate
from LLM_loader import llm_loader

def log_elapsed_time(file_path, elapsed_time, entry_type, uid=None, cq_idx=None):
    with open(file_path, 'a') as log_file:
        if entry_type == 'UID':
            log_file.write(f"Time elapsed for UID {uid}: {elapsed_time}\n")
        elif entry_type == 'UID-CQ':
            log_file.write(f"Time elapsed for UID {uid} and CQ {cq_idx}: {elapsed_time}\n")

def bin_res(config):
    pdf_dir = config['Paths']['pdfs_path']
    output_csv = config['Paths']['output_csv']
    cqs_dir = config['Paths']['Ans_to_cq_path_processed']
    uid_log_file = config['Paths']['time_logs_doc_binres_path']
    uid_cq_log_file = config['Paths']['time_logs_doc_CQ_binres_path']
    CQs = load_cqs(config.get('Paths', 'CQs_path'))
    template = read_txt(config.get('Paths', 'Binary_reponses_prompt'))
    prompt_template = PromptTemplate(input_variables=["Question","Answer"],template=template)
    llm = llm_loader()

    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
    else:
        df = pd.DataFrame()

    if 'UID' not in df.columns:
        df['UID'] = pd.NA
    if 'Status' not in df.columns:
        df['Status'] = pd.NA

    pdf_files = glob.glob(f"{pdf_dir}*.pdf")
    pdf_uids = [d[13:] for d in pdf_files]

    if not df.empty:
        processed_uids = df[df['Status'] == 'Complete']['UID'].dropna().tolist()
        pdf_uids = [uid for uid in pdf_uids if uid not in processed_uids]

    for i, uid in enumerate(pdf_uids, len(df)):
        if uid not in df['UID'].values:
            df.at[i, "UID"] = uid
            df.at[i, "Status"] = "In Progress"

        uid_start_time = datetime.now()

        for idx, cq in enumerate(CQs, 1):
            txt_file = f'{cqs_dir}Publication_{uid}_CQ{idx}.txt'
            cq_start_time = datetime.now()
            
            if os.path.exists(txt_file):
                with open(txt_file, 'r') as f:
                    context = f.read().replace('\n', ' ')
                final_prompt = prompt_template.format(Question=cq, Answer=context)
                pattern = r'(?:Answer|Response|the\s+answer\s+is|response\s+is)(?::::|:)?\s*["\']?(Yes|No)["\']?\s*'
                groq_use = config['Models']['use_groq']
                if groq_use == "true":
                    output = llm.invoke(final_prompt)
                    final_response = re.findall(pattern, output.content, re.IGNORECASE) 
                else:
                    output = llm.invoke(final_prompt, return_only_outputs=True)
                    final_response = re.findall(pattern, output, re.IGNORECASE) 
                response = final_response[-1] if final_response else "No response found"
                CQ_num = f"CQ{idx}"
                df.at[i, CQ_num] = response
            else:
                print(f"File {txt_file} not found, skipping.")
            
            # Log elapsed time for CQ
            cq_end_time = datetime.now()
            cq_elapsed_time = cq_end_time - cq_start_time
            log_elapsed_time(uid_cq_log_file, cq_elapsed_time, entry_type='UID-CQ', uid=uid, cq_idx=idx)
        
        df.at[i, "Status"] = "Complete"
        
        uid_end_time = datetime.now()
        uid_elapsed_time = uid_end_time - uid_start_time
        log_elapsed_time(uid_log_file, uid_elapsed_time, entry_type='UID', uid=uid)
        
        df.to_csv(output_csv, index=False)

    df.to_csv(output_csv, index=False)


