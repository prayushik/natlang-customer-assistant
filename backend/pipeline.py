# backend/pipeline.py
from typing import Dict

def run_pipeline(user_message: str) -> Dict[str, str]:
    """
    Runs the full NatLang pipeline.
    This will be wired to natlang.py next.
    """
    sentiment = "neutral"
    response = "Pipeline not yet connected."

    return {
        "sentiment": sentiment,
        "response": response
    }
