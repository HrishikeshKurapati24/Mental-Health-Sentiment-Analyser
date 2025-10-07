import shap
import numpy as np
import torch
from typing import List, Dict, Tuple
from app.models.sentiment_model import model

# -------------- SHAP EXPLAINER --------------
class SHAPWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, texts):
        # SHAP might send numpy arrays of strings, so cast to list[str]
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        if isinstance(texts, str):
            texts = [texts]

        # Now tokenize
        tokens = self.tokenizer(
            texts,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=1024
        )

        with torch.no_grad():
            outputs = self.model(**tokens)
            return torch.sigmoid(outputs.logits).detach().numpy()

# --------- Dynamic phrase aggregation function ---------
def aggregate_phrases_dynamic(words: List[str], values: List[float], min_k: int = 2, max_k: int = 4):
    """
    Aggregate SHAP scores into phrase-level scores with variable phrase lengths.
    Generates all n-grams in range [min_k, max_k].
    """
    phrases = []

    for k in range(min_k, max_k+1):
        for i in range(len(words) - k + 1):
            phrase = " ".join(words[i:i+k])
            phrase_score = sum(values[i:i+k]) / k
            phrases.append((phrase, phrase_score, i, i+k))

    # Sort by absolute importance
    phrases.sort(key=lambda x: abs(x[1]), reverse=True)

    # Cluster & filter overlapping phrases
    selected = []
    covered_spans = []

    for phrase, score, start, end in phrases:
        overlap = False
        for s, e in covered_spans:
            # Check if this phrase overlaps with an already selected one
            if not (end <= s or start >= e):  # intervals overlap
                overlap = True
                break

        if not overlap:
            selected.append((phrase, score))
            covered_spans.append((start, end))

    return selected

class ShapExplainer:
    def __init__(self):
        self.explainer = shap.Explainer(
            SHAPWrapper(model.model, model.tokenizer), 
            masker=shap.maskers.Text(model.tokenizer) # pyright: ignore[reportAttributeAccessIssue]
        )
    
    def explain_text(self, text: str, binary_preds: Dict[str, int]) -> Dict[str, List[Tuple[str, float]]]:
        # Get SHAP values for the input text
        shap_val = self.explainer([text])

        # Process SHAP values for each label
        explanations = {}
        for idx, label in enumerate(model.labels):
            if binary_preds[label] == 1:  # Only explain positive predictions
                # Get word importance scores
                words, values = [], []
                for word, value in zip(shap_val.data[0], shap_val.values[0, :, idx]):
                    if word != "":  # Skip padding tokens
                        words.append(word)
                        values.append(float(value))

                # Sort words by absolute importance
                word_importance = list(zip(words, values))
                word_importance.sort(key=lambda x: abs(x[1]), reverse=True)

                # Dynamic phrase aggregation (2–5 n-grams)
                explanations[label] = aggregate_phrases_dynamic(words, values, min_k=2, max_k=5)[:15]
        
        return explanations

# Create a singleton instance
explainer = ShapExplainer()