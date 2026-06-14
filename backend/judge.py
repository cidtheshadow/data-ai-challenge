import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

def generate_reasoning(candidate_doc: str, jd_context: str) -> str:
    """
    Agentic Judge Layer.
    Uses an LLM to generate a fact-based justification for the candidate ranking.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        # Fallback to simple heuristic reasoning if no key
        reasoning = "Candidate matches core requirements based on semantic similarity."
        if "ml" in candidate_doc.lower() or "machine learning" in candidate_doc.lower():
            reasoning = "Demonstrates Machine Learning experience aligned with JD."
        if "recommendation" in candidate_doc.lower():
            reasoning = "Strong match: Built recommendation/ranking systems as required."
        return reasoning
        
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, max_tokens=100)
    
    prompt = PromptTemplate(
        input_variables=["jd", "candidate"],
        template=(
            "You are an expert technical recruiter agent.\n"
            "Job Description Context:\n{jd}\n\n"
            "Candidate Profile:\n{candidate}\n\n"
            "Task: Write a concise (1-2 sentences), hallucination-free justification "
            "for why this candidate is a good match for the job. "
            "You MUST cite direct phrases from the candidate's profile. "
            "If the capability is not explicitly written, state 'No direct evidence found'.\n"
            "Justification:"
        )
    )
    
    chain = prompt | llm
    
    try:
        response = chain.invoke({"jd": jd_context, "candidate": candidate_doc})
        return response.content.strip().replace("\n", " ")
    except Exception as e:
        print(f"LLM Error: {e}")
        return "Matched based on vector similarity."

