import os
os.environ['HF_HOME'] = 'path_where_huggingface_models_can_save'
from langchain_core.prompts import PromptTemplate
import numpy as np
import pandas as pd
from langchain_groq import ChatGroq
import re
from configparser import ConfigParser
from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from helper_functions import read_txt, get_embeddings
from LLM_loader import llm_loader
load_dotenv(override = True)
groq_api_key = os.getenv('groq_api_key')

config = ConfigParser()
config.read('config.ini')
template = read_txt(config.get('Paths', 'Binary_reponses_prompt'))
llm = llm_loader()
embedding_model_id = config['Models']['embedding_model_id']
df = pd.DataFrame(os.listdir("../Data/pdfs/"), columns = ["UID"])

output_file = "../Data/DL_methods_identify.csv"
if os.path.exists(output_file):
    existing_df = pd.read_csv(output_file)
    df = df.merge(existing_df, on="UID", how="left", suffixes=('', '_existing'))


for i,x in df.iterrows():
    col = x["UID"]
    if pd.notnull(x.get("Response")):
        continue
    file_path = "../Data/pdfs/" + col
    loader = PDFMinerLoader(file_path)
    documents = loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    text_chunks=text_splitter.split_documents(documents)
    vectorstore=FAISS.from_documents(text_chunks, embeddings)
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", return_source_documents=False, retriever=vectorstore.as_retriever())
    output = chain.invoke(template)
    pattern = r'(?:Answer:::\s*|Response:\s*|the\s+answer\s+is|response\s+is)\s*["\']?(Yes|No)["\']?'
    final_response = re.findall(pattern, output["result"], re.IGNORECASE)
    response = final_response[-1] if final_response else "No response found"
    df.at[i, "Response"] = response
    df.to_csv(output_file,index=False)
