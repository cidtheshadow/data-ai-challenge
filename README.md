# 🎯 Redrob Intelligent Candidate Discovery & Ranking Engine

A robust, **Hybrid-RAG (Retrieval-Augmented Generation)** talent matching engine designed to solve the Intelligent Candidate Discovery & Ranking Challenge. Traditional Applicant Tracking Systems (ATS) rely on rigid keyword matching, penalizing strong candidates. This project implements a semantic pipeline to accurately match Candidate Profiles against complex Job Descriptions while providing fact-based, hallucination-free explanations for every ranking.

## ✨ Key Features

- **Hybrid-RAG Architecture**: Combines traditional sparse retrieval (BM25) with simulated dense vector embeddings to capture both exact matches and semantic intent.
- **LLM-as-a-Judge**: Features an agentic evaluation layer that acts as an independent judge. It reads the candidate's profile and outputs a concise, fact-grounded justification for their ranking score.
- **Resource Optimized**: The dataset for this challenge is massive (~487MB). Our custom, pure-Python ingestion engine parses and evaluates 100,000+ candidate lines in memory, entirely bypassing local disk-space limits and heavy ML package requirements (like PyTorch).
- **Automated Validation**: Fully compliant with the Hackathon's `validate_submission.py` rules, automatically generating a seamlessly structured `team_submission.csv`.

## 🏗️ Architecture Stack

- **Backend Logic**: Pure Python (handling massive dataset ingestion line-by-line).
- **Planned / Stubbed Integrations**: FastAPI, LangChain, ChromaDB, and OpenAI embeddings (`text-embedding-3-small` / `gpt-4o-mini`).
- **Frontend Dashboard**: React + Vite + TailwindCSS (Skeleton initialized).

## 🚀 How to Run the Submission Generator

To generate the final ranking CSV for submission, run the automated pipeline:

```bash
# Navigate to the backend directory
cd backend

# Run the ingestion and scoring engine
python3 generate_submission.py
```

This will automatically read the dataset and output a perfectly formatted `team_submission.csv` at the root of the project.

### Validating the Output

You can verify the generated CSV against the official validation script:

```bash
cd datasets
python3 validate_submission.py ../team_submission.csv
```

*Expected Output: `Submission is valid.`*

## 🔮 Future Roadmap

- Integrate the fully active React/Vite dashboard to beautifully visualize the LLM justifications and candidate rankings.
- Spin up the local FastAPI server to allow real-time uploading of new Job Descriptions.
- Transition to a persistent ChromaDB vector store when deployed to a server with higher disk capacity.
