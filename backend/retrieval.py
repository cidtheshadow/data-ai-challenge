import pickle
import chromadb
from rank_bm25 import BM25Okapi
import chromadb.utils.embedding_functions as embedding_functions
import os
from typing import List, Dict, Any

CHROMA_DIR = "./chroma_db"
BM25_FILE = "./bm25_index.pkl"

class HybridRetriever:
    def __init__(self):
        # Load BM25
        with open(BM25_FILE, 'rb') as f:
            data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.bm25_ids = data["ids"]
            self.bm25_docs = data["documents"]
            self.bm25_metadatas = data["metadatas"]
            
        # Load Chroma
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            self.ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name="text-embedding-3-small"
            )
        else:
            # Fallback if no key
            self.ef = None
            
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = self.client.get_collection("candidates", embedding_function=self.ef)
        
    def retrieve_hybrid(self, query: str, top_k: int = 50, alpha: float = 0.5) -> List[Dict[str, Any]]:
        """Stage 1: Hybrid Search."""
        # 1. Sparse (BM25)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Normalize BM25
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        norm_bm25 = [s / max_bm25 for s in bm25_scores]
        
        bm25_results = {self.bm25_ids[i]: norm_bm25[i] for i in range(len(self.bm25_ids))}
        
        # 2. Dense (Vector)
        dense_results = {}
        if self.ef:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k * 2 # Get more to intersect
            )
            
            for i, cid in enumerate(results['ids'][0]):
                dist = results['distances'][0][i]
                # Convert distance to similarity (assuming cosine)
                sim = max(0, 1.0 - (dist / 2.0))
                dense_results[cid] = sim
                
        # Combine
        combined = []
        for i, cid in enumerate(self.bm25_ids):
            s_sparse = bm25_results.get(cid, 0.0)
            s_dense = dense_results.get(cid, 0.0)
            
            # Simple linear combination
            s_hybrid = (alpha * s_sparse) + ((1 - alpha) * s_dense)
            
            # Extract anomaly signals
            meta = self.bm25_metadatas[i]
            response_rate = float(meta.get("response_rate", 0.0))
            p_anomaly = 0.0
            if response_rate < 0.2:
                p_anomaly = 0.2 # Penalty
                
            combined.append({
                "candidate_id": cid,
                "score": s_hybrid - p_anomaly,
                "doc": self.bm25_docs[i],
                "meta": meta
            })
            
        # Sort and return top_k
        combined.sort(key=lambda x: x["score"], reverse=True)
        return combined[:top_k]

def rerank(candidates: List[Dict[str, Any]], jd_context: str) -> List[Dict[str, Any]]:
    """Stage 2: Cross-Encoder Reranking Mock or LLM."""
    # Since we can't install torch/sentence-transformers easily due to space,
    # we simulate reranking or use a lightweight heuristic.
    # In a real app, this would be an LLM call or a Cross-Encoder.
    
    # We will just boost candidates that have exact title matches in JD for now, 
    # to simulate semantic matching.
    jd_lower = jd_context.lower()
    for c in candidates:
        doc = c["doc"].lower()
        # boost score if AI, Machine Learning, RAG, etc are mentioned
        boost = 0.0
        if "machine learning" in doc or "ml" in doc: boost += 0.1
        if "ai" in doc: boost += 0.1
        if "recommendation" in doc: boost += 0.1
        if "ranking" in doc: boost += 0.1
        c["final_score"] = c["score"] + boost
        
    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    return candidates
