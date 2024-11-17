# Information Retrieval using multiple LLMs and RAG
This repository contains code, data, prompts and results related to information retrieval from PDFs using multiple LLMs and the RAG approach. Later, results from individual LLM were used to create a hard voting classifier, which enhances the overall quality of the results.
**LLMs used:** Llama-3 70B, Llama-3.1 70B, Mixtral-8x22B-Instruct-v0.1, Mixtral 8x7B, and Gemma 2 9B
## Overview
In this project, we present an innovative approach that harnesses the power of large language models (LLMs) to automate the extraction and processing of deep learning methodological information from scientific literature. LLMs, trained on extensive text datasets, have shown remarkable proficiency in natural language understanding and generation. In this work, we utilize five open-source LLMs: Llama-3 70B1, Llama-3.1 70B2, Mixtral-8x22B-Instruct-v0.13, Mixtral 8x7B4, and Gemma 2 9B5, in conjunction with a Retrieval-Augmented Generation (RAG) framework. Our goal is to achieve a higher level of accuracy and efficiency in extracting pertinent methodological details compared to traditional manual methods. Our methodology consists of three key components: identifying relevant research publications, employing the RAG approach for automatic information extraction, and transforming the extracted textual responses into categorical values for more straightforward evaluation. Then, the categorical values were compared with manually annotated data to evaluate. 
## Contents
* CQs/: Contains the competency questions.
* Code/: Contains code files for data extraction, RAG approach, for converting textual to categorical responses and data processing 
* Data/: Contains the metadata for publications, TDWG abstracts, and RAG-retrieved textual responses for all the CQ and publication combinations, time logs, and categorical responses.
* Evaluation/: Contains LLM and human evaluations of categorical responses
* Prompts/: Contains prompts used in all stages of the pipeline.
## Usage
Create a virtual environment and install the prerequisite using the requirements.txt file.

Update the .env file with your respective API keys. To use [groq models](https://console.groq.com/docs/models), update the [Models][use_groq] to true. To use [hugging face models](https://huggingface.co/models), update the variable to false. In addition to the groq usage, update the [Models][model_id] according to the desired model usage. [Paths][selected_model] is a variable and is also used as a placeholder in different paths in config.ini, in case of multiple model usage, it is important to update this variable name according to the model name

To run the whole pipeline, submit (execute) your job using main.py and use config.ini to customise the variables and paths

## License
The repository is licensed under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Publication
Kommineni, V.K., Konig-ries, B., & Samuel, S. (2024). Harnessing multiple LLMs for Information Retrieval: A case study on Deep Learning methodologies in Biodiversity publications. arXiv preprint arXiv:2411.09269. 
[https://doi.org/10.48550/arXiv.2411.09269](https://doi.org/10.48550/arXiv.2411.09269)





