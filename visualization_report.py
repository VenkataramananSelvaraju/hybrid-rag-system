import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_visualizations(csv_file="evaluation_report.csv"):
    """
    Reads the CSV report and creates a dashboard of 4 plots:
    1. MRR Distribution (How many queries got perfect retrieval?)
    2. Latency Histogram (Is the system fast enough?)
    3. BERT Score Spread (How good is the language quality?)
    4. Metrics Summary (Bar chart of averages)
    """
    if not os.path.exists(csv_file):
        print(f"❌ Error: '{csv_file}' not found. Please run evaluate_rag.py first.")
        return

    print(f"Reading {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Set global style
    sns.set_theme(style="whitegrid")
    
    # Create a figure with 2 rows and 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Hybrid RAG Evaluation Dashboard', fontsize=20, weight='bold')
    
    # --- Plot 1: MRR Score Distribution ---
    # Helps you see if retrieval is "All or Nothing" (0s and 1s) or mixed.
    sns.histplot(df['mrr_score'], bins=10, kde=True, ax=axes[0, 0], color='#3498db')
    axes[0, 0].set_title('Retrieval Quality (MRR) Distribution', fontsize=14)
    axes[0, 0].set_xlabel('MRR Score (0.0 to 1.0)')
    axes[0, 0].set_ylabel('Number of Queries')

    # --- Plot 2: Response Time (Latency) ---
    # Helps identify outliers that take too long.
    sns.histplot(df['response_time'], bins=20, kde=True, ax=axes[0, 1], color='#e74c3c')
    axes[0, 1].set_title('System Latency Distribution', fontsize=14)
    axes[0, 1].set_xlabel('Response Time (seconds)')
    
    # --- Plot 3: Semantic Accuracy (BERTScore) ---
    # Box plot shows the median and spread of answer quality.
    if 'bert_f1' in df.columns:
        sns.boxplot(x=df['bert_f1'], ax=axes[1, 0], color='#2ecc71', width=0.5)
        axes[1, 0].set_title('Answer Quality Spread (BERT F1)', fontsize=14)
        axes[1, 0].set_xlabel('BERT F1 Score')
    else:
        axes[1, 0].text(0.5, 0.5, "BERT Score not found in CSV", ha='center')

    # --- Plot 4: Executive Summary (Averages) ---
    metrics = {
        'Avg MRR': df['mrr_score'].mean(),
        'Avg Precision@5': df['precision_at_5'].mean(),
    }
    if 'bert_f1' in df.columns:
        metrics['Avg BERT F1'] = df['bert_f1'].mean()
    
    # Create Bar Chart
    names = list(metrics.keys())
    values = list(metrics.values())
    
    sns.barplot(x=names, y=values, ax=axes[1, 1], palette="viridis", hue=names, legend=False)
    axes[1, 1].set_title('Overall Performance Metrics', fontsize=14)
    axes[1, 1].set_ylim(0, 1.0)
    
    # Add values on top of bars
    for i, v in enumerate(values):
        axes[1, 1].text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    
    # Save
    output_file = "evaluation_visualization.png"
    plt.savefig(output_file)
    print(f"🎉 Dashboard saved as '{output_file}'")

if __name__ == "__main__":
    generate_visualizations()