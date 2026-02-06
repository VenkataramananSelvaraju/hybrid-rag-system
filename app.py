import streamlit as st
import time
from hybrid_rag import HybridRAG  # Importing your backend class

# --- Page Configuration ---
st.set_page_config(
    page_title="Hybrid RAG System",
    page_icon="🧠",
    layout="wide"
)

# --- Initialize System (Cached) ---
# We use @st.cache_resource so the model/index loads only ONCE, not every time you click a button.
@st.cache_resource
def load_rag_system():
    return HybridRAG(corpus_file="rag_corpus.json")

# Display a loading spinner while the system starts up
with st.spinner("Loading RAG System (Models & Indices)... This may take a minute."):
    rag = load_rag_system()

# --- UI Layout ---
st.title("🧠 Hybrid RAG: Dense + Sparse + RRF")
st.markdown("""
This system combines **Vector Search** (Semantic) and **BM25** (Keyword) using **Reciprocal Rank Fusion**.
""")

# Sidebar for controls
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Retrieval Count (Top-K)", min_value=10, max_value=100, value=50)
    top_n = st.slider("Final Context (Top-N)", min_value=1, max_value=10, value=5)
    rrf_k = st.number_input("RRF Constant (k)", value=60)

# Main Query Input
query = st.text_input("Enter your question about the Wikipedia corpus:", placeholder="e.g., What is the theory of relativity?")

if st.button("Generate Answer", type="primary"):
    if not query:
        st.warning("Please enter a question.")
    else:
        # --- Execution & Timing ---
        start_time = time.time()
        
        # 1. Run the specific pipeline steps manually to get granular data for UI
        dense_results = rag.search_dense(query, top_k=top_k)
        sparse_results = rag.search_sparse(query, top_k=top_k)
        
        rrf_results = rag.reciprocal_rank_fusion(dense_results, sparse_results, k=rrf_k)
        final_context = rrf_results[:top_n]
        
        # 2. Generate Answer
        answer = rag.generate_answer(query, final_context)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # --- Display Results ---
        st.success("Analysis Complete!")
        
        # Metric Cards
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Response Time", value=f"{response_time:.2f} sec")
        with col2:
            st.metric(label="Chunks Retrieved", value=len(final_context))
            
        st.divider()
        
        # Generated Answer Section
        st.subheader("🤖 Generated Answer")
        st.info(answer)
        
        # Evidence Section
        st.subheader("📚 Source Evidence (Top-N Chunks)")
        
        for i, doc in enumerate(final_context):
            with st.expander(f"Rank {i+1}: {doc['source']} (Score: {doc['rrf_score']:.4f})"):
                st.markdown(f"**Text:** {doc['text']}")
                st.markdown(f"**Chunk ID:** `{doc['chunk_id']}`")
                
        # Debug / Explainer Section (Good for marks!)
        st.divider()
        st.subheader("🔍 Retrieval Internals")
        tab1, tab2 = st.tabs(["Dense Results (Vector)", "Sparse Results (BM25)"])
        
        with tab1:
            st.dataframe(dense_results[:10]) # Show top 10 raw dense
        with tab2:
            st.dataframe(sparse_results[:10]) # Show top 10 raw sparse