from transformers import LongformerTokenizerFast, LongformerForSequenceClassification
import torch
from typing import List, Dict, Tuple
import os
import json
from pathlib import Path

class SentimentModel:
    def __init__(self):
        # Initialize model and tokenizer from Hugging Face Hub
        model_id = os.getenv("MODEL_ID", "your-model-id/mental-health-sentiment")
        local_model_path = "./local_model"  # folder inside your Space
        
        self.labels = ["neutral", "low mood/depressed", "anxious/worried", "stressed/overwhelmed"]

        # Load thresholds from JSON file
        self.thresholds = self._load_thresholds()

        # Check if local model exists, else download and save
        if not os.path.exists(local_model_path):
            print("Downloading model and tokenizer from Hugging Face Hub...")
            tokenizer = LongformerTokenizerFast.from_pretrained(model_id, use_fast=True)
            model = LongformerForSequenceClassification.from_pretrained(model_id)

            tokenizer.save_pretrained(local_model_path)
            model.save_pretrained(local_model_path)
        else:
            print("Loading model and tokenizer from local folder...")

        # Load model and tokenizer from local folder
        self.tokenizer = LongformerTokenizerFast.from_pretrained(local_model_path, use_fast=True, local_files_only=True)
        self.model = LongformerForSequenceClassification.from_pretrained(local_model_path, local_files_only=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device) # type: ignore
        self.model.eval() # type: ignore
        
    def predict(self, text: str) -> Tuple[Dict[str, float], Dict[str, int]]:
        # Tokenize and prepare input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=1024
        ).to(self.device)
        
        # Model forward pass
        with torch.no_grad():
            outputs = self.model(**inputs) # type: ignore
            probabilities = torch.sigmoid(outputs.logits).squeeze().tolist()

        # Create prediction dictionary with probabilities
        predictions = {
            label: float(prob)
            for label, prob in zip(self.labels, probabilities)
        }
        
        #  Create binary classifications based on thresholds
        self.binary_predictions = {
            label: 1 if prob >= self.thresholds[i] else 0
            for i, (label, prob) in enumerate(predictions.items())
        }
        
        return (predictions, self.binary_predictions)

    def _load_thresholds(self) -> List[float]:
        """Load classification thresholds from JSON file (expects a list/array)"""
        threshold_path = Path(__file__).parent / "best_thresholds.json"
        default_thresholds = [0.5, 0.45, 0.45, 0.45]
        try:
            if threshold_path.exists():
                with open(threshold_path, 'r') as f:
                    loaded_thresholds = json.load(f)
                if isinstance(loaded_thresholds, list) and len(loaded_thresholds) == len(self.labels):
                    return loaded_thresholds
                else:
                    print("Warning: thresholds.json format/length incorrect, using default thresholds")
                    return default_thresholds
            else:
                print("Warning: thresholds.json not found, using default thresholds")
                with open(threshold_path, 'w') as f:
                    json.dump(default_thresholds, f, indent=4)
                return default_thresholds
        except Exception as e:
            print(f"Error loading thresholds: {str(e)}, using default thresholds")
            return default_thresholds

# Create a singleton instance
model = SentimentModel()