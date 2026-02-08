import json
import random
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm  # For progress bar

# --- Configuration ---
CORPUS_FILE = "rag_corpus.json"
OUTPUT_FILE = "test_dataset.json"
NUM_QUESTIONS = 10
MODEL_NAME = "google/flan-t5-base"

def load_corpus(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def generate_evaluation_dataset():
    # 1. Setup Model
    print(f"Loading {MODEL_NAME} for Q&A Generation...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Model loaded on {device}.")

    # 2. Load Data
    corpus = load_corpus(CORPUS_FILE)
    
    # Filter for chunks with enough substance (skip tiny chunks)
    viable_chunks = [c for c in corpus if len(c['text_content'].split()) > 50]
    
    # Randomly sample chunks to generate questions from
    # We sample slightly more than needed in case generation fails
    selected_chunks = random.sample(viable_chunks, min(len(viable_chunks), NUM_QUESTIONS + 20))
    
    dataset = []
    
    print(f"Generating {NUM_QUESTIONS} Q&A pairs...")
    
    for chunk in tqdm(selected_chunks):
        if len(dataset) >= NUM_QUESTIONS:
            break
            
        context = chunk['text_content']
        
        # --- Step A: Generate Question ---
        q_prompt = f"generate question: {context}"
        
        input_ids = tokenizer(q_prompt, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
        outputs = model.generate(input_ids, max_length=64, do_sample=True, temperature=0.7)
        question = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Filter out bad questions (too short or generic)
        if len(question) < 10 or "?" not in question:
            continue
            
        # --- Step B: Generate Ground Truth Answer ---
        # We ask the model to answer its own question using the context
        a_prompt = f"answer question: {question} context: {context}"
        
        input_ids = tokenizer(a_prompt, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
        outputs = model.generate(input_ids, max_length=128)
        ground_truth = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # --- Step C: Save ---
        dataset.append({
            "question_id": len(dataset) + 1,
            "question": question,
            "ground_truth": ground_truth,
            "source_url": chunk['source_url'],
            "source_chunk_id": chunk['chunk_id'],
            "metadata": {
                "type": "factual" # Placeholder for category
            }
        })
        
    # 3. Save to JSON
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(dataset, f, indent=4)
        
    print(f"\nSuccess! Generated {len(dataset)} pairs in {OUTPUT_FILE}")
    
    # Preview
    print("\n--- Example Entry ---")
    print(json.dumps(dataset[0], indent=2))

if __name__ == "__main__":
    generate_evaluation_dataset()