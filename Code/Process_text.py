import os

def process_text(config):
    input_dir = config.get('Paths', 'Ans_to_cq_base_path')
    output_dir = config.get('Paths', 'Ans_to_cq_base_path_processed')
    model_to_process = config.get('Paths', 'selected_model')

    output_folder = os.path.join(output_dir, model_to_process)
    os.makedirs(output_folder, exist_ok=True)
    input_folder_path = os.path.join(input_dir, model_to_process)

    for filename in os.listdir(input_folder_path):
        input_file_path = os.path.join(input_folder_path, filename)
        
        if filename.endswith('.txt'):
            output_file_path = os.path.join(output_folder, filename)

            with open(input_file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            if content.count('Answer:::') >= 2:
                parts = content.split('Answer:::')
                extracted_text = parts[2].strip()
                modified_content = extracted_text.replace('Helpful Answer:', '').replace('Answer:', '').strip()
            else:
                modified_content = content.replace('Answer:::', '').replace('Helpful Answer:::', '').replace('Answer:', '').replace('Helpful Answer:', '').strip()

            with open(output_file_path, 'w', encoding='utf-8') as file:
                file.write(modified_content)
