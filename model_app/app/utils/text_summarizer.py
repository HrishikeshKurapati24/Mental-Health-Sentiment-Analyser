import cohere
import os
from typing import Dict, List, Tuple

class TextSummarizer:
    def __init__(self):
        self.cohere_model = cohere.Client(os.getenv("COHERE_API_KEY"))
    
    def generate_summary(
        self,
        text: str,
        predictions: Dict[str, float],
        explanations: Dict[str, List[Tuple[str, float]]]
    ) -> str:
        # Format the input for the summarization
        context = f"Mental health sentiment analysis of text: '{text}'\n\n"
        
        # Add prediction scores
        context += "Model's predictions:\n"
        for label, score in predictions.items():
            context += f"- {label.capitalize()}: {score:.2%}\n"
        
        # Add key contributing phrases with scores
        context += "\nContributing phrases as per SHAP analysis:\n"
        for label, phrases in explanations.items():
            if phrases:  # Only show labels that have explanations
                top_phrases = [f"'{phrase}' ({score:.3f})" for phrase, score in phrases[:3]]
                context += f"- {label.capitalize()}: {', '.join(top_phrases)}\n"
        
        # Generate summary using Cohere
        prompt = (
            f"{context}\n\n"
            "Summarize the mental health analysis by explaining what the model detected "
            "and which specific phrases from the text led to each prediction. "
            "Be concise and evidence-based."
        )
        
        response = self.cohere_model.generate(
            prompt=prompt,
            max_tokens=1000,
            temperature=0.7
        )
        
        return response.generations[0].text.strip()

# Create a singleton instance
summarizer = TextSummarizer()