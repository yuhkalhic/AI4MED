import json
from transformers import AutoTokenizer
from tqdm import tqdm

def analyze_data(file_path):
    tokenizer = AutoTokenizer.from_pretrained("/mnt/public/code/wangzr/yjy/Meta-Llama-3.1-8B-Instruct")
    
    golden_true = 0
    golden_false = 0
    max_length = 0
    max_text = ""
    
    with open(file_path, 'r') as f:
        for line in tqdm(f):
            data = json.loads(line)
            
            if data['golden']:
                golden_true += 1
            else:
                golden_false += 1
            
            tokens = tokenizer(data['instruction'])
            length = len(tokens['input_ids'])
            if length > max_length:
                max_length = length
                max_text = data['instruction']
    
    print(f"\nStatistics:")
    print(f"Golden True count: {golden_true}")
    print(f"Golden False count: {golden_false}")
    print(f"Maximum token length: {max_length}")
    print(f"\nLongest text:")
    print(max_text)

if __name__ == "__main__":
    analyze_data("processed_data.jsonl")