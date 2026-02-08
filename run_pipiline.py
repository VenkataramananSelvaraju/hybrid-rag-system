import os
import subprocess
import pandas as pd
import datetime

# --- Configuration ---
REPORT_FILE = "final_rag_report.html"
CSS_STYLE = """
<style>
    body { font-family: sans-serif; margin: 40px; line-height: 1.6; }
    h1 { color: #2c3e50; border-bottom: 2px solid #2c3e50; }
    h2 { color: #34495e; margin-top: 30px; }
    table { border-collapse: collapse; width: 100%; margin: 20px 0; }
    th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
    th { background-color: #f2f2f2; }
    .metric-box { background: #e8f6f3; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 5px solid #1abc9c; }
    img { max-width: 800px; margin: 20px 0; border: 1px solid #ddd; padding: 5px; }
    .timestamp { color: #7f8c8d; font-size: 0.9em; }
</style>
"""

def run_step(script_name, description):
    print(f"\n{'='*60}")
    print(f"🚀 STEP: {description} ({script_name})")
    print(f"{'='*60}")
    
    # Using Subprocess to ensure clean memory for each heavy task
    # We pass the environment to handle the OMP issue on macOS
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    
    try:
        subprocess.run(["python", script_name], check=True, env=env)
        print(f"✅ {description} Completed Successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {script_name}: {e}")
        exit(1)

def generate_html_report():
    print(f"\n📝 Generating Comprehensive Report: {REPORT_FILE}...")
    
    # 1. Load Data
    try:
        df = pd.read_csv("evaluation_report.csv")
        metrics = {
            "mrr": df['mrr_score'].mean(),
            "precision": df['precision_at_5'].mean(),
            "bert": df['bert_f1'].mean() if 'bert_f1' in df.columns else 0.0,
            "latency": df['response_time'].mean()
        }
    except FileNotFoundError:
        print("Error: evaluation_report.csv not found.")
        return

    # 2. Build HTML Content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Hybrid RAG Evaluation Report</title>
        {CSS_STYLE}
    </head>
    <body>
        <h1>🧠 Hybrid RAG System Evaluation Report</h1>
        <p class="timestamp">Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <div class="metric-box">
            <h2>📊 Executive Summary</h2>
            <p><strong>Total Questions Evaluated:</strong> {len(df)}</p>
            <ul>
                <li><strong>Mean Reciprocal Rank (MRR):</strong> {metrics['mrr']:.4f} (Target: >0.5)</li>
                <li><strong>Precision@5:</strong> {metrics['precision']:.4f}</li>
                <li><strong>Semantic Accuracy (BERTScore F1):</strong> {metrics['bert']:.4f}</li>
                <li><strong>Avg Response Time:</strong> {metrics['latency']:.4f} sec</li>
            </ul>
        </div>

        <h2>📈 Innovation: Ablation Study</h2>
        <p>Comparison of retrieval methods (Dense vs. Sparse vs. Hybrid).</p>
        <img src="ablation_results.png" alt="Ablation Study Chart">
        
        <h2>📋 Detailed Failure Analysis (Bottom 5 Queries)</h2>
        <p>The following queries received the lowest scores and represent areas for improvement:</p>
        {df.sort_values(by='mrr_score').head(5)[['question', 'ground_truth', 'generated_answer', 'mrr_score']].to_html(index=False)}
        
    </body>
    </html>
    """
    
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"🎉 Report generated successfully! Open '{REPORT_FILE}' in your browser.")

if __name__ == "__main__":
    # --- 1. Check/Run Generation ---
    if not os.path.exists("test_dataset.json"):
        run_step("generate_qa.py", "Generating Test Dataset")
    else:
        print("ℹ️  Using existing test_dataset.json")

    # --- 2. Run Evaluation ---
    run_step("evaluate_rag.py", "Running Core Evaluation Metrics")

    # --- 3. Run Ablation Study ---
    run_step("ablation_study.py", "Running Ablation Study")

    # --- 4. Generate Report ---
    generate_html_report()
