import json
import os
from pathlib import Path
from typing import List, Dict, Any
import pickle
import chromadb
from rank_bm25 import BM25Okapi

DATA_DIR = Path("../datasets")
CANDIDATES_FILE = DATA_DIR / "candidates.jsonl"
CHROMA_DIR = "./chroma_db"
BM25_FILE = "./bm25_index.pkl"

def strip_pii(candidate: Dict[str, Any]) -> str:
    """Creates a comprehensive text representation of the candidate without PII."""
    profile = candidate.get("profile", {})
    
    # Anonymized summary and core profile info
    headline = profile.get("headline", "")
    summary = profile.get("summary", "")
    yoe = profile.get("years_of_experience", 0)
    
    doc_parts = []
    doc_parts.append(f"Professional Summary ({yoe} Years Experience):\nHeadline: {headline}\nSummary: {summary}")
    
    skills = candidate.get("skills", [])
    if skills:
        skills_str = ", ".join([f"{s.get('name', '')} ({s.get('proficiency', '')})" for s in skills])
        doc_parts.append(f"\nSkills: {skills_str}")
        
    career_history = candidate.get("career_history", [])
    if career_history:
        doc_parts.append("\nCareer History:")
        for role in career_history:
            title = role.get("title", "")
            duration = role.get("duration_months", 0)
            desc = role.get("description", "")
            doc_parts.append(f"- Role: {title} ({duration} months)\n  Description: {desc}")
            
    education = candidate.get("education", [])
    if education:
        doc_parts.append("\nEducation:")
        for edu in education:
            degree = edu.get("degree", "")
            field = edu.get("field_of_study", "")
            doc_parts.append(f"- {degree} in {field}")
            
    return "\n".join(doc_parts)

def build_indices():
    if not CANDIDATES_FILE.exists():
        print(f"File not found: {CANDIDATES_FILE}")
        return

    print("Loading and parsing candidates...")
    documents = []
    metadatas = []
    ids = []
    
    # We will limit to 10,000 for this script if the file is massive, or process all.
    # The file is 487MB, so processing all at once for embedding might be slow and expensive.
    # Let's process the first 1000 for local demonstration/hackathon purposes as per the spec.
    max_candidates = 1000
    
    count = 0
    with open(CANDIDATES_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            cand = json.loads(line)
            cid = cand.get("candidate_id")
            
            # Extract Signals for metadata scoring later
            signals = cand.get("redrob_signals", {})
            response_rate = signals.get("recruiter_response_rate", 0.0)
            last_active = signals.get("last_active_date", "")
            
            text_doc = strip_pii(cand)
            
            documents.append(text_doc)
            metadatas.append({
                "candidate_id": cid,
                "response_rate": float(response_rate),
                "last_active": last_active
            })
            ids.append(cid)
            
            count += 1
            if count >= max_candidates:
                break

    print(f"Processed {len(documents)} candidates.")
    
    print("Building BM25 Index...")
    tokenized_corpus = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    
    with open(BM25_FILE, 'wb') as f:
        pickle.dump({
            "bm25": bm25,
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas
        }, f)
        
    print("Building ChromaDB Index...")
    import chromadb.utils.embedding_functions as embedding_functions
    # Use OpenAI Embeddings if key exists, else a local mini model
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        print("Using OpenAI text-embedding-3-small...")
        ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-3-small"
        )
    else:
        print("Using local sentence-transformers model...")
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    # Delete if exists to recreate
    try:
        client.delete_collection("candidates")
    except Exception:
        pass
        
    collection = client.create_collection(name="candidates", embedding_function=ef)
    
    # Batch add to chroma
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        collection.add(
            documents=documents[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end]
        )
        print(f"Added {end}/{len(ids)} to Chroma.")
        
    print("Ingestion complete!")

if __name__ == "__main__":
    build_indices()
