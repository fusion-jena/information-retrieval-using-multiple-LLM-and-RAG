import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

base_path = "../Data/Ans_to_cqs/V1_processed/"
subfolders = [os.path.join(base_path, f) for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
df_dl_identify = pd.read_csv("../Data/DL_methods_identify.csv")
No_DL_method_list = df_dl_identify[df_dl_identify["DL_method_availability"]=="No"]["UID"].tolist()
print(len(No_DL_method_list))
file_contents = {}

for subfolder in subfolders:
    for filename in os.listdir(subfolder):
        base_filename = filename.rsplit('_CQ', 1)[0]
        base_filename = base_filename.replace('Publication_', '', 1)
        if filename.endswith('.txt') and base_filename not in No_DL_method_list:
            file_path = os.path.join(subfolder, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                file_contents.setdefault(filename, []).append(content)                
print(len(file_contents))
file_names = list(file_contents.keys())
similarity_dict = {}
for x in range(len(file_names)):
    texts = file_contents[file_names[x]]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    score = cosine_similarity(tfidf_matrix)
    texts = ["mixtral_8_7b", "llama-3.1_70b", "mixtral_8_22b", "llama-3_70b", "gemma2-9b-it"]
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            pair_key = f"{texts[i]} - {texts[j]}"
            if pair_key not in similarity_dict:
                similarity_dict[pair_key] = []
            similarity_dict[pair_key].append(score[i][j])
            
mean_similarity_dict = {}
for key, scores in similarity_dict.items():
    if scores: 
        mean_similarity = sum(scores) / len(scores)
        mean_similarity_dict[key] = mean_similarity

for key, mean_value in mean_similarity_dict.items():
    print(f'Mean Similarity for {key}: {mean_value:.4f}')



