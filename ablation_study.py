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

def calculate_mrr(retrieved_docs, ground_truth_url):
    """Calculates MRR for a list of docs."""
    for rank, doc in enumerate(retrieved_docs):
        if doc['source'] == ground_truth_url:
            return 1.0 / (rank + 1)
    return 0.0

def run_ablation():
    # 1. Initialize
    print("Initializing System for Ablation Study...")
    rag = HybridRAG(corpus_file="rag_corpus.json")
    
    with open(TEST_DATASET, 'r') as f:
        questions = json.load(f)

    results = []

    print(f"Running Ablation on {len(questions)} questions...")

    # 2. Loop through all questions
    for q_data in tqdm(questions):
        query = q_data['question']
        truth_url = q_data['source_url']
        
        # --- Experiment A: Dense Only ---
        dense_raw = rag.search_dense(query, top_k=TOP_N) # Get top N directly
        mrr_dense = calculate_mrr(dense_raw, truth_url)
        
        # --- Experiment B: Sparse Only ---
        sparse_raw = rag.search_sparse(query, top_k=TOP_N) # Get top N directly
        mrr_sparse = calculate_mrr(sparse_raw, truth_url)
        
        # --- Experiment C: Hybrid RRF ---
        # Get deep pools (Top K) then Fuse
        d_pool = rag.search_dense(query, top_k=TOP_K)
        s_pool = rag.search_sparse(query, top_k=TOP_K)
        rrf_out = rag.reciprocal_rank_fusion(d_pool, s_pool, k=60)
        
        # Slice to Top N
        hybrid_final = rrf_out[:TOP_N]
        mrr_hybrid = calculate_mrr(hybrid_final, truth_url)
        
        results.append({
            "question_id": q_data['question_id'],
            "Dense Only": mrr_dense,
            "Sparse Only": mrr_sparse,
            "Hybrid RRF": mrr_hybrid
        })

    # 3. Aggregate Results
    df = pd.DataFrame(results)
    
    # Calculate Averages
    avg_scores = df[["Dense Only", "Sparse Only", "Hybrid RRF"]].mean()
    print("\n=== ABLATION RESULTS (Mean MRR) ===")
    print(avg_scores)
    
    # 4. Visualize
    create_visualization(avg_scores)

def create_visualization(scores):
    """
    Creates a bar chart comparing the methods.
    """
    plt.figure(figsize=(10, 6))
    
    # Create Bar Chart
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x=scores.index, y=scores.values, palette="viridis")
    
    # Add labels
    plt.title("Ablation Study: Retrieval Method Comparison", fontsize=16)
    plt.ylabel("Mean Reciprocal Rank (MRR)", fontsize=12)
    plt.xlabel("Method", fontsize=12)
    plt.ylim(0, 1.0) # MRR is always between 0 and 1
    
    # Add values on top of bars
    for i, v in enumerate(scores.values):
        ax.text(i, v + 0.02, f"{v:.4f}", ha='center', fontsize=12, fontweight='bold')
    
    # Save plot
    plt.savefig("ablation_results.png")
    print("Chart saved as 'ablation_results.png'")
    plt.show()

if __name__ == "__main__":
    run_ablation()