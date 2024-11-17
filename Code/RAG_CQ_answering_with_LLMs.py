from helper_functions import load_cqs, read_txt, get_embeddings, log_time_and_execute
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import textwrap
import numpy as np
from LLM_loader import llm_loader
from dotenv import load_dotenv
import os
import time

def RAG_CQ_answering(config):
    CQs = load_cqs(config.get('Paths', 'CQs_path'))
    embedding_model_id = config.get('Models', 'embedding_model_id')
    template = read_txt(config.get('Paths', 'RAG_question_answering_prompt'))
    prompt_template = PromptTemplate(input_variables=["query"], template=template)
    #pdfs_path = config.get('Paths', 'pdfs_path')
    ans_to_cq_path = config.get('Paths', 'Ans_to_cq_path')
    log_path_file = config.get('Paths', 'time_logs_doc_CQans_path')
    log_path_file_cq = config.get('Paths', 'time_logs_doc_CQ_CQans_path')
    llm = llm_loader()
    for d in os.listdir("../Data/pdfs/"):
        file_check = f"Publication_{d}_CQ28.txt"
        print(file_check)
        if os.path.exists(os.path.join(ans_to_cq_path, file_check)):
            print(f"Skipping {d} as {file_check} already exists.")
            continue
        #print(d)
        doc_start_time = time.time()
        file_path = "../Data/pdfs/" + d
        #print(file_path)
        loader = PDFMinerLoader(file_path)
        documents = loader.load()
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        text_chunks=text_splitter.split_documents(documents)
        vectorstore=FAISS.from_documents(text_chunks, get_embeddings(embedding_model_id))
        chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", return_source_documents=False, retriever=vectorstore.as_retriever())                                    
        for cq, p in zip(CQs, np.arange(1, len(CQs) + 1)):
            log_time_and_execute(cq, d, p, chain, prompt_template, ans_to_cq_path, log_path_file_cq)
        doc_end_time = time.time()
        total_doc_time = doc_end_time - doc_start_time
        with open(log_path_file, 'a') as doc_log_file:
            doc_log_file.write(f"Document {d} processing time: {total_doc_time} seconds\n")
