import argparse
import json
import os
from os.path import join as pjoin
import re
from typing import List, Dict, Tuple
import logging
import random

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http import models

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

template = """You are a professional medical expert to answer the # Question. Please first using the # Retrieved Documents and then answer the question. Your responses will be used for research purposes only, so please have a definite and direct answer.
You should respond in the format <answer>A/B/C/D</answer> (only one option can be chosen) at your response.

# Retrieved Documents
{documents}

# Question
{question}"""

class DocumentProcessor:
    def __init__(self, model_name: str = "BAAI/llm-embedder", p: float = 0.8, num_distract: int = 3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.p = p
        self.num_distract = num_distract
        logging.info(f"Using device: {self.device}")
        
        logging.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        self.qdrant = QdrantClient(":memory:")

    def extract_document_blocks(self, text: str) -> List[Dict[str, str]]:
        blocks = []
        lines = []
        current_source = None
        
        for line in text.split('\n'):
            if line.startswith('## source:'):
                if current_source and lines:
                    blocks.append({
                        "source": current_source,
                        "content": '\n'.join(lines)
                    })
                    lines = []
                current_source = line
                lines = [line]
            elif current_source is not None: 
                lines.append(line)
        
        if current_source and lines:
            blocks.append({
                "source": current_source,
                "content": '\n'.join(lines)
            })
            
        return blocks

    def get_embedding(self, text: str) -> np.ndarray:
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", 
                                  max_length=512, truncation=True, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings.cpu().numpy()

    def rank_blocks(self, blocks: List[Dict[str, str]], query: str) -> Tuple[List[str], List[str]]:
        if not blocks:
            return [], []

        block_contents = [block["content"] for block in blocks]
        doc_embeddings = np.vstack([self.get_embedding(content) for content in block_contents])
        query_embedding = self.get_embedding(query)
        
        vector_size = doc_embeddings.shape[1]
        
        collection_name = "temp_collection"
        self.qdrant.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )
        
        self.qdrant.upload_points(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=idx,
                    vector=embedding.tolist(),
                    payload={"content": content}
                )
                for idx, (embedding, content) in enumerate(zip(doc_embeddings, block_contents))
            ]
        )

        k = min(len(blocks), max(1, self.num_distract + 1))

        positive_results = self.qdrant.search(
            collection_name=collection_name,
            query_vector=query_embedding[0].tolist(),
            limit=k
        )
        positive_blocks = [hit.payload["content"] for hit in positive_results]
        
        negative_results = self.qdrant.search(
            collection_name=collection_name,
            query_vector=(-query_embedding[0]).tolist(),  
            limit=k
        )
        negative_blocks = [hit.payload["content"] for hit in negative_results]
        
        self.qdrant.delete_collection(collection_name)
        
        return positive_blocks, negative_blocks

    def process_example(self, content: str, question: str) -> Dict:
        try:
            blocks = self.extract_document_blocks(content)
            if not blocks:
                logging.warning("No document blocks found")
                return {
                    "instruction": "",
                    "output": "",
                    "golden": False
                }

            positive_blocks, negative_blocks = self.rank_blocks(blocks, question)
            is_golden = random.random() < self.p
            
            if is_golden:
                selected_blocks = positive_blocks[:1] + negative_blocks[:self.num_distract]
            else:
                selected_blocks = negative_blocks[:self.num_distract + 1]
            
            random.shuffle(selected_blocks)
            
            
            documents_str = '\n'.join(selected_blocks)
            instruction = template.format(
                documents=documents_str,
                question=question
            )

            return {
                "instruction": instruction,
                "output": "",
                "golden": is_golden
            }
            
        except Exception as e:
            logging.error(f"Error processing example: {str(e)}")
            raise

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, nargs="+", required=True,
                       help="List of datasets to process")
    parser.add_argument("--output_path", type=str, default="processed_data.jsonl")
    parser.add_argument("--p", type=float, default=0.8,
                       help="Probability of golden examples")
    parser.add_argument("--num_distract", type=int, default=3,
                       help="Number of distractor blocks for golden examples")
    return parser.parse_args()

def load_dataset(dataset_name: str) -> List[Dict]:
    plan_name = f"system=planner_addret,dataset={dataset_name},debug=False"
    output_all_path = pjoin("alog", plan_name, "output_all.json")
    logging.info(f"Loading data from {output_all_path}")
    
    with open(output_all_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    args = get_args()
    
    processor = DocumentProcessor(p=args.p, num_distract=args.num_distract)
    
    output_all = []
    for dataset in args.datasets:
        try:
            dataset_items = load_dataset(dataset)
            logging.info(f"Loaded {len(dataset_items)} items from {dataset}")
            output_all.extend(dataset_items)
        except Exception as e:
            logging.error(f"Error loading dataset {dataset}: {str(e)}")
            continue
    
    logging.info(f"Total items after combining datasets: {len(output_all)}")

    processed_data = []
    logging.info(f"Total examples to process: {len(output_all)}")
    
    for item in tqdm(output_all):
        try:
            if "pred" not in item or "doc_path" not in item["pred"]:
                logging.warning(f"Missing pred or doc_path in item: {item.get('id', 'NO_ID')}")
                continue
                
            doc_path = item["pred"]["doc_path"]
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            question = item.get("question", "")
            processed_item = processor.process_example(content, question)
            processed_item["output"] = f"<answer>{item.get('gold', '')}</answer>"
            
            processed_data.append(processed_item)
            
        except Exception as e:
            logging.error(f"Error processing item {item.get('id', 'NO_ID')}: {str(e)}")
            continue
    
    logging.info(f"Successfully processed {len(processed_data)} examples")
    
    logging.info(f"Saving processed data to {args.output_path}")
    with open(args.output_path, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    main()
