# Assignment 2: Hybrid RAG System with Automated Evaluation

## Group 72 - Hybrid RAG Implementation
**Group Member Names:**
* SATISH KUMAR PATHAK (2024AA05578) 100%
* PRANAV DEWANGAN (2024AA05554) 100%
* TRIPATHY BIJAY KUMAR (2024AA05067) 100%
* SHRUTIKA ARORA(2023AC05440) 100%
* VENKATARAMANAN S (2024AA05555) 100%

### 1. Project Overview
This project implements a **Hybrid Retrieval-Augmented Generation (RAG)** system designed to answer questions from a corpus of 500 Wikipedia articles. It combines **Dense Retrieval** (Vector Search via FAISS) and **Sparse Retrieval** (Keyword Search via BM25) using **Reciprocal Rank Fusion (RRF)** to retrieve the most relevant context for a Generative LLM (`Flan-T5-base`).

The system includes a fully **Automated Pipeline** that orchestrates data generation, evaluation metrics (MRR, BERTScore), ablation studies, and generates a final HTML report.

---

### 2. File Structure
* `run_pipeline.py`: **[Main Script]** Runs the entire evaluation suite in one command and generates the final HTML report.
* `data_prep.py`: Fetches 200 fixed + 300 random Wikipedia articles and creates the chunked dataset.
* `hybrid_rag.py`: Core engine class containing FAISS, BM25, RRF logic, and LLM generation.
* `app.py`: Streamlit-based User Interface for interactive testing.
* `generate_qa.py`: Uses an LLM to generate 100 synthetic Question-Answer pairs.
* `evaluate_rag.py`: Runs the RAG system against the test dataset and calculates MRR, Precision@K, and BERTScore.
* `ablation_study.py`: Compares Dense vs. Sparse vs. Hybrid performance and generates a visualization.
* `requirements.txt`: List of Python dependencies.

---

### 3. Installation

**Prerequisites:** Python 3.10+ (Tested on Python 3.14).

1.  **Clone/Unzip the project.**
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If on macOS, see the Troubleshooting section below regarding FAISS.*

---

### 4. How to Run the System

#### A. Data Preparation (First Run Only)
Before running anything, you must download the data.
```bash
python data_prep.py
```

#### B. The Automated Pipeline (Requirement 2.4)
This single command runs the entire evaluation suite:

Generates 100 Test Questions (if missing).

Calculates Metrics (MRR, Precision, BERTScore).

Runs Ablation Study (Dense vs. Hybrid).

**Generates** `final_rag_report.html`.

**Command:**
```bash
# MacOS Users (prevents crashing):
OMP_NUM_THREADS=1 python run_pipeline.py

# Windows/Linux Users:
python run_pipeline.py
```

**Output:** Open final_rag_report.html in your browser to see the results.

#### C. Interactive Web App (UI)
Launch the dashboard to test queries manually.

```bash
# MacOS Users:
OMP_NUM_THREADS=1 streamlit run app.py

# Windows/Linux Users:
streamlit run app.py
```

* Open your browser at `http://localhost:8501`.

### 5. Troubleshooting
**Issue: `Segmentation Fault` or Python Crash on macOS**
* **Cause:** Conflict between FAISS and macOS threading libraries (OpenMP).
* **Fix:** Always run scripts with OMP_NUM_THREADS=1 before the python command.

**Issue:`AttributeError: pipeline` or `KeyError: text2text-generation`**
* **Cause:** Newer `transformers` versions in Python 3.14 have changed pipeline registration.
* **Fix:** The provided `hybrid_rag.py` uses `model.generate()` directly to avoid this.

**Issue: "Model weights not tied" Warning**
* **Status:** Harmless.
* **Fix:** The code includes config.tie_word_embeddings = False to suppress this warning.

### 6. System Architecture

1. **Ingestion:** Wikipedia API -> Clean -> Chunk (Sliding Window).
2. **Indexing:** 
  * **Dense:** `all-MiniLM-L6-v2` embeddings stored in **FAISS**.
  * **Sparse:** Tokenized text stored in **BM25Okapi**.
3. **Retrieval:** Top-K results from both indices are fused using **Reciprocal Rank Fusion (RRF)** ($k=60$).
4. **Generation:** Top-N fused chunks + Query -> Flan-T5-Base -> Answer.

### 7. Credits
* **Libraries:** HuggingFace Transformers, Sentence-Transformers, FAISS, Rank-BM25.
* **Models:** `all-MiniLM-L6-v2`, `google/flan-t5-base`.
