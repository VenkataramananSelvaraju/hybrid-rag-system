import json
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from hybrid_rag import HybridRAG

# --- Custom Metrics Libraries ---
try:
    from bert_score import score as bert_score
    BERT_SCORE_AVAILABLE = True
except ImportError:
    print("Warning: 'bert-score' library not found. Semantic Metric will be skipped.")
    BERT_SCORE_AVAILABLE = False

# --- Configuration ---
TEST_DATASET = "test_dataset.json"
RESULTS_FILE = "evaluation_report.csv"
TOP_K_RETRIEVAL = 50  # Search depth
TOP_N_CONTEXT = 5     # Final chunks used

def calculate_mrr_url(retrieved_docs, ground_truth_url):
    """
    Mandatory Metric: Mean Reciprocal Rank at URL Level.
    Formula: 1 / Rank_of_first_correct_URL
    """
    for rank, doc in enumerate(retrieved_docs):
        # Check if the source URL matches the ground truth
        if doc['source'] == ground_truth_url:
            return 1.0 / (rank + 1)
    return 0.0

def calculate_precision_at_k(retrieved_docs, ground_truth_url, k=5):
    """
    Custom Metric 2: Precision@K (URL Level)
    Formula: (Relevant Items in Top K) / K
    Measures how much of the retrieved context actually belongs to the correct source.
    """
    if not retrieved_docs:
        return 0.0
    
    # Slice to top K
    k_docs = retrieved_docs[:k]
    relevant_count = sum(1 for doc in k_docs if doc['source'] == ground_truth_url)
    
    return relevant_count / k

def evaluate_system():
    # 1. Initialize RAG
    print("Initializing RAG System for Evaluation...")
    rag = HybridRAG(corpus_file="rag_corpus.json")
    
    # 2. Load Questions
    with open(TEST_DATASET, 'r') as f:
        questions = json.load(f)
        
    print(f"Loaded {len(questions)} test questions.")
    
    results = []
    
    # 3. Evaluation Loop
    print("Starting Evaluation Loop...")
    for q_data in tqdm(questions):
        query = q_data['question']
        ground_truth_ans = q_data['ground_truth']
        ground_truth_url = q_data['source_url']
        
        # --- Run System ---
        start_time = time.time()
        pipeline_out = rag.run_pipeline(query, top_n_final=TOP_N_CONTEXT)
        end_time = time.time()
        
        generated_ans = pipeline_out['answer']
        retrieved_context = pipeline_out['context'] # These are the top N chunks
        
        # --- Calculate Metrics ---
        
        # Metric 1: MRR (Mandatory)
        # Note: We pass the 'dense_results' or 'sparse_results' if we want raw retrieval MRR,
        # but usually we check the final fused list (context). 
        # However, MRR is often calculated on the larger retrieval set (Top-K) before filtering.
        # Let's check the RRF sorted list (which represents the system's ranking).
        # We need access to the full sorted list. Let's re-run fusion strictly for ranking check if needed, 
        # but using the Top-N context is strict and acceptable for "Final Context" evaluation.
        # Ideally, we check the Top-K from RRF.
        
        # *Correction for best practice*: MRR should be checked against the specific Retrieval output.
        # Since 'run_pipeline' returns Top-N, let's assume if it's not in Top-N, rank is 0.
        mrr_score = calculate_mrr_url(retrieved_context, ground_truth_url)
        
        # Metric 2: Precision@K (Custom)
        precision_score = calculate_precision_at_k(retrieved_context, ground_truth_url, k=TOP_N_CONTEXT)
        
        # Metric 3: Response Time
        latency = end_time - start_time
        
        # Record Data
        results.append({
            "question_id": q_data['question_id'],
            "question": query,
            "generated_answer": generated_ans,
            "ground_truth": ground_truth_ans,
            "mrr_score": mrr_score,
            "precision_at_5": precision_score,
            "response_time": latency
        })

    # --- Batch Metrics (BERTScore) ---
    # BERTScore is faster if run in a batch at the end rather than inside the loop
    if BERT_SCORE_AVAILABLE:
        print("\nCalculating BERTScore (Semantic Similarity)...")
        preds = [r['generated_answer'] for r in results]
        refs = [r['ground_truth'] for r in results]
        
        # Calculate F1 measure (rescale_with_baseline is optional but recommended if set up)
        P, R, F1 = bert_score(preds, refs, lang="en", verbose=True)
        
        # Add to results
        for i, f1_val in enumerate(F1):
            results[i]['bert_f1'] = f1_val.item()
    else:
        for r in results:
            r['bert_f1'] = 0.0

    # 4. Save & Report
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_FILE, index=False)
    
    print("\n" + "="*30)
    print("EVALUATION REPORT")
    print("="*30)
    print(f"Total Questions: {len(df)}")
    print(f"Mean Reciprocal Rank (MRR): {df['mrr_score'].mean():.4f}")
    print(f"Avg Precision@5:           {df['precision_at_5'].mean():.4f}")
    print(f"Avg BERTScore F1:          {df['bert_f1'].mean():.4f}")
    print(f"Avg Response Time:         {df['response_time'].mean():.4f} sec")
    print(f"Report saved to: {RESULTS_FILE}")

if __name__ == "__main__":
    evaluate_system()