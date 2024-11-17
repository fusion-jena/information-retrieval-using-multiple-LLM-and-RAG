import os
from configparser import ConfigParser
from RAG_CQ_answering_with_LLMs import RAG_CQ_answering
from Binary_reponses import bin_res
from Process_text import process_text

def main():
    config = ConfigParser()
    config.read('config.ini')
    RAG_CQ_answering(config)
    process_text(config)
    bin_res(config)

if __name__ == "__main__":
    main()
