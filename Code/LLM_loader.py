from helper_functions import load_llm
from configparser import ConfigParser

def llm_loader():
    config = ConfigParser()
    config.read('config.ini')
    
    model_id = config.get('Models', 'model_id')
    embedding_model_id = config.get('Models', 'embedding_model_id')
    use_groq = config.getboolean('Models', 'use_groq')
    
    # Load the appropriate LLM based on the configuration
    llm = load_llm(model_id, embedding_model_id, use_groq)
    
    return llm
