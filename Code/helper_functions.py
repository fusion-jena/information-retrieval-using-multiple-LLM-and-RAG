import os
os.environ['HF_HOME'] = 'path_where_huggingface_models_can_save'
import re
import transformers
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface.llms import HuggingFacePipeline
import torch
from huggingface_hub import login
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import time
import textwrap

load_dotenv(override = True)
access_token_read = os.getenv('access_token_read_hf')
login(token = access_token_read)
groq_api_key = os.getenv('groq_api_key')

def load_llm(model_id, embedding_model_id, use_groq):
    if use_groq:
        llm = ChatGroq(model_name=model_id, temperature=0, groq_api_key=groq_api_key)
    else:
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_id)
        model_config = transformers.AutoConfig.from_pretrained(model_id, cache_dir='path_where_huggingface_models_can_save')
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir='path_where_huggingface_models_can_save')
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map='auto',
            cache_dir='path_where_huggingface_models_can_save'
        )
        model.eval()
        pipe = pipeline(
            model=model,
            tokenizer=tokenizer,
            task='text-generation',
            temperature=0,
            max_new_tokens=1200,
            repetition_penalty=1.1,
            device_map="auto"
        )
        llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature': 0})
    
    return llm
def load_cqs(CQs_path):
    with open(CQs_path) as f:
        lines = f.readlines()
    CQs = [l[:-1] for l in lines]
    return CQs

def read_txt(txt_path):
    with open(txt_path,'r') as f:
        content = f.read()
    return content

def get_embeddings(embedding_model_id):
    return HuggingFaceEmbeddings(model_name=embedding_model_id)
    

def log_time_and_execute(cq, d, p, chain, prompt_template, ans_to_cq_path, log_path_file_cq):
    start_time = time.time()
    
    # Execute the query
    prompt = prompt_template.format(query=cq)
    result = chain.invoke({"query": prompt}, return_only_outputs=True)
    
    # Save the result
    with open(f"{ans_to_cq_path}Publication_{d}_CQ{p}.txt", 'w') as f:
        f.write(result['result'])
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Log the time taken
    with open(log_path_file_cq, 'a') as log_file:
        log_file.write(f"Document {d}, CQ {p}, {elapsed_time} seconds\n")
