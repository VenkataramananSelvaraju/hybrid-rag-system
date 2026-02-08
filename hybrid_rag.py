import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
import nltk
import torch
import os

# Fix for MacOS Segmentation Fault
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# Ensure nltk data is ready
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

class HybridRAG:
    def __init__(self, corpus_file="rag_corpus.json", model_name="all-MiniLM-L6-v2", llm_name="google/flan-t5-base"):
        """
        Initializes the Hybrid RAG system.
        """
        self.corpus_file = corpus_file
        self.embedding_model = SentenceTransformer(model_name)
        
        # Load Corpus
        print(f"Loading corpus from {corpus_file}...")
        try:
            with open(corpus_file, 'r') as f:
                self.corpus = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find {corpus_file}. Did you run data_prep.py?")
        
        # Lists to store text data for indexing
        # We assume the list order is preserved, so index i in texts corresponds to index i in sources
        self.corpus_texts = [item['text_content'] for item in self.corpus]
        self.corpus_ids = [item['chunk_id'] for item in self.corpus]
        self.corpus_sources = [item['source_url'] for item in self.corpus]  # <--- NEW: Store sources for O(1) lookup
        
        # --- 1. Initialize Dense Index (FAISS) ---
        print("Building Dense Index (FAISS)...")
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Create Embeddings
        embeddings = self.embedding_model.encode(self.corpus_texts, convert_to_tensor=True, show_progress_bar=True)
        # Convert to numpy for FAISS
        embeddings_np = embeddings.cpu().detach().numpy()
        self.index.add(embeddings_np)
        
        # --- 2. Initialize Sparse Index (BM25) ---
        print("Building Sparse Index (BM25)...")
        tokenized_corpus = [nltk.word_tokenize(doc.lower()) for doc in self.corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # --- 3. Initialize Generator (LLM) ---
        print(f"Loading LLM: {llm_name}...")
        
        # STANDARD LOAD (Removes the 'tie_word_embeddings' fix which broke the model)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(llm_name)
        
        # Move to GPU if available (MPS for Mac, CUDA for Nvidia, CPU otherwise)
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps" # Optimized for Mac M1/M2/M3
        else:
            self.device = "cpu"
            
        self.model.to(self.device)
        print(f"Hybrid RAG System Ready on {self.device}!")

    def search_dense(self, query, top_k=50):
        """
        Retrieves top-K chunks using Vector Search.
        """
        query_embedding = self.embedding_model.encode([query])
        D, I = self.index.search(query_embedding, top_k)
        
        results = []
        for rank, idx in enumerate(I[0]):
            if idx != -1: # Valid index
                results.append({
                    "chunk_id": self.corpus_ids[idx],
                    "text": self.corpus_texts[idx],
                    "source": self.corpus_sources[idx],
                    "rank": rank + 1,  # 1-based rank
                    "score": float(D[0][rank]) # Distance score
                })
        return results

    def search_sparse(self, query, top_k=50):
        """
        Retrieves top-K chunks using BM25.
        """
        tokenized_query = nltk.word_tokenize(query.lower())
        # Get scores for all docs
        scores = self.bm25.get_scores(tokenized_query)
        # Sort and get top_k indices
        top_n_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_n_indices):
            results.append({
                "chunk_id": self.corpus_ids[idx],
                "text": self.corpus_texts[idx],
                "source": self.corpus_sources[idx],
                "rank": rank + 1,
                "score": float(scores[idx])
            })
        return results

    def reciprocal_rank_fusion(self, dense_results, sparse_results, k=60):
        """
        Combines results using RRF formula: Score = 1 / (k + rank)
        """
        fusion_scores = {}
        
        # Process Dense
        for doc in dense_results:
            cid = doc['chunk_id']
            if cid not in fusion_scores:
                fusion_scores[cid] = 0
            fusion_scores[cid] += 1 / (k + doc['rank'])
            
        # Process Sparse
        for doc in sparse_results:
            cid = doc['chunk_id']
            if cid not in fusion_scores:
                fusion_scores[cid] = 0
            fusion_scores[cid] += 1 / (k + doc['rank'])
            
        # Sort by RRF score descending
        sorted_ids = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Reconstruct result objects
        final_results = []
        for cid, score in sorted_ids:
            # Efficiently find the doc by checking our loaded lists (index lookup is tricky here due to sorting)
            # So we use the robust lookup
            original_doc = next(item for item in self.corpus if item['chunk_id'] == cid)
            
            final_results.append({
                "chunk_id": cid,
                "text": original_doc['text_content'],
                "source": original_doc['source_url'],
                "rrf_score": score
            })
            
        return final_results

    def generate_answer(self, query, context_chunks):
        """
        Generates an answer using the LLM and retrieved context.
        """
        # 1. Safety Check
        if not context_chunks:
            return "I could not find relevant information in the provided documents."

        # 2. Combine Context
        context_text = " ".join([c['text'] for c in context_chunks])
        
        # 3. Strict Prompting
        input_text = f"question: {query} context: {context_text}"
        
        # 4. Tokenize
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True).to(self.device)
        
        # 5. Generate (Deterministic Mode)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=200,      # Generate up to 200 new words
            num_beams=5,             # Look 5 steps ahead (Smart search)
            do_sample=False,         # DISABLE Sampling (Fixes the gibberish)
            temperature=1.0,         # Reset temperature (ignored when do_sample=False)
            repetition_penalty=1.2,  # Penalize repeating the same sentence
            length_penalty=1.0,      # Encourage standard length answers
            early_stopping=True      # Stop when the sentence is complete
        )
        
        # 6. Decode
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

    def run_pipeline(self, query, top_n_final=5):
        """
        Full RAG pipeline: Query -> Dense+Sparse -> RRF -> Generate
        """
        # 1. Retrieve
        dense_res = self.search_dense(query)
        sparse_res = self.search_sparse(query)
        
        # 2. Fuse
        rrf_results = self.reciprocal_rank_fusion(dense_res, sparse_res)
        top_context = rrf_results[:top_n_final]
        
        # 3. Generate
        answer = self.generate_answer(query, top_context)
        
        return {
            "query": query,
            "answer": answer,
            "context": top_context,
            "raw_dense": dense_res[:3], # Return simplified debug info
            "raw_sparse": sparse_res[:3]
        }

# --- Testing Block ---
if __name__ == "__main__":
    # Ensure you have 'rag_corpus.json' from Phase 1
    try:
        rag = HybridRAG()
        test_query = "What is the theory of relativity?" # Change to match your fixed URLs topic
        result = rag.run_pipeline(test_query)
        
        print("\n=== FINAL ANSWER ===")
        print(result['answer'])
        print("\n=== TOP CONTEXT SOURCE ===")
        print(result['context'][0]['source'])
    except Exception as e:
        print(f"Error during test run: {e}")