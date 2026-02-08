import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from hybrid_rag import HybridRAG

# --- Configuration ---
TEST_DATASET = "test_dataset.json"
TOP_K = 50
TOP_N = 5

def normalize_url(url):
    """
    Standardizes URLs for comparison.
    """
    if not url: return ""
    # Decode URL encoding (e.g. %20 -> space)
    try:
        from urllib.parse import unquote
        url = unquote(url)
    except:
        pass
        
    url = url.lower().strip()
    url = url.replace("https://", "").replace("http://", "").replace("www.", "")
    if url.endswith("/"):
        url = url[:-1]
    return url

def calculate_mrr(retrieved_docs, ground_truth_url, debug=False):
    """
    Calculates MRR with debug printing.
    """
    clean_truth = normalize_url(ground_truth_url)
    
    for rank, doc in enumerate(retrieved_docs):
        if 'source' not in doc:
            continue
            
        clean_retrieved = normalize_url(doc['source'])
        
        # Debug print for the first rank of the first failure
        if debug and rank == 0:
            print(f"   [DEBUG] Truth: '{clean_truth}' | Retrieved: '{clean_retrieved}'")

        # Fuzzy Match
        if clean_truth == clean_retrieved or clean_truth in clean_retrieved or clean_retrieved in clean_truth:
            return 1.0 / (rank + 1)
            
    return 0.0

def run_ablation():
    print("Initializing System for Ablation Study...")
    rag = HybridRAG(corpus_file="rag_corpus.json")
    
    with open(TEST_DATASET, 'r') as f:
        questions = json.load(f)

    results = []
    print(f"Running Ablation on {len(questions)} questions...")

    # Limit to first 5 questions for debugging if needed, or remove [:5] to run all
    for i, q_data in enumerate(tqdm(questions)):
        query = q_data['question']
        truth_url = q_data['source_url']
        
        # Only print debug info for the first 3 questions
        debug_mode = (i < 3) 
        if debug_mode:
            print(f"\n--- Question {i+1}: {query} ---")

        # --- Experiment A: Dense Only ---
        dense_raw = rag.search_dense(query, top_k=TOP_N)
        mrr_dense = calculate_mrr(dense_raw, truth_url, debug=debug_mode)
        
        # --- Experiment B: Sparse Only ---
        sparse_raw = rag.search_sparse(query, top_k=TOP_N)
        mrr_sparse = calculate_mrr(sparse_raw, truth_url, debug=False)
        
        # --- Experiment C: Hybrid RRF ---
        d_pool = rag.search_dense(query, top_k=TOP_K)
        s_pool = rag.search_sparse(query, top_k=TOP_K)
        rrf_out = rag.reciprocal_rank_fusion(d_pool, s_pool, k=60)
        hybrid_final = rrf_out[:TOP_N]
        mrr_hybrid = calculate_mrr(hybrid_final, truth_url, debug=False)
        
        results.append({
            "question_id": q_data.get('question_id', i),
            "Dense Only": mrr_dense,
            "Sparse Only": mrr_sparse,
            "Hybrid RRF": mrr_hybrid
        })

    # Aggregate Results
    df = pd.DataFrame(results)
    
    # Calculate Averages
    avg_scores = df[["Dense Only", "Sparse Only", "Hybrid RRF"]].mean()
    print("\n=== ABLATION RESULTS (Mean MRR) ===")
    print(avg_scores)
    
    if avg_scores['Hybrid RRF'] > 0:
        create_visualization(avg_scores)
    else:
        print("\n⚠️ WARNING: MRR is still 0.0. Check the DEBUG prints above.")

def create_visualization(scores):
    try:
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        ax = sns.barplot(x=scores.index, y=scores.values, palette="viridis")
        plt.title("Ablation Study: Retrieval Method Comparison", fontsize=16)
        plt.ylabel("Mean Reciprocal Rank (MRR)", fontsize=12)
        plt.xlabel("Method", fontsize=12)
        plt.ylim(0, 1.0)
        for i, v in enumerate(scores.values):
            ax.text(i, v + 0.02, f"{v:.4f}", ha='center', fontsize=12, fontweight='bold')
        plt.savefig("ablation_results.png")
        print("Chart saved as 'ablation_results.png'")
    except Exception as e:
        print(f"Visualization skipped: {e}")

if __name__ == "__main__":
    run_ablation()