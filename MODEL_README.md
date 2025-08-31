# Mental Health Sentiment Analyzer

You can download the model from this Hugging Face Hub link: https://huggingface.co/Hrishikesh4/mental-health-classifier-longformer.

## Model Information

- **Model Name:** `Hrishikesh4/mental-health-classifier-longformer`
- **Base Model:** `AllenAI/longformer-base-4096`
- **Model Type:** Longformer-4096-based, fine-tuned for mental health classification
- **Language:** English
- **License:** MIT

## Overview

The Mental Health Sentiment Analyzer is a specialized text classification model designed to detect early signs of depression, anxiety, stress, and neutral mental states from user-provided journals, reflections, or messages. Leveraging Longformer’s extended context window, it handles long, complex narratives for accurate detection of subtle and overlapping emotional states.

> The model addresses the challenge of recognizing mental health struggles in digital interactions. It analyzes voluntary text inputs—such as journals, self-reflections, or chatbot conversations—to identify patterns of stress, anxiety, or depression. Depending on user consent, it can provide self-help recommendations, notify a therapist, or alert an emergency contact. The goal is timely intervention with privacy and ethical use.

## Model Details

- **Fine-tuning Stages:** Two-stage process (multi-class → multi-label)
- **Purpose:** Early detection of mental health risk indicators

### Training Process

#### Stage 1: Multi-class Fine-tuning

- **Datasets:**
    - Kaggle’s Emotions dataset (15k sampled)
    - Synthetic mental health journals (20k, generated via ChatGPT)
- **Task:** Single-label emotion classification
- **Configuration:**
    - Learning rate: 2e-5
    - Weight decay: 0.01
    - Epochs: 3
- **Results:** ~98% accuracy (train/validation); ~50% accuracy on longer Reddit examples

#### Stage 2: Multi-label Fine-tuning

- **Datasets:**
    - Reddit emotional posts (17k, relabeled via Cohere + Gemini, manually verified)
    - PsychForums dataset (2k)
    - Synthetic neutral journals (500, via ChatGPT)
- **Task:** Multi-label classification (neutral, depressed, anxious, stressed)
- **Configuration:**
    - Learning rate: 2e-5
    - Weight decay: 0.02
    - Epochs: 2
- **Results:**
    - Macro F1: 0.8104
    - Weighted F1: 0.8132
    - Adapted test set F1: ~0.89

## Model Architecture

Based on Longformer, which uses sparse attention for efficient processing of long texts. The model employs a multi-label classification head to capture overlapping emotional states in long-form entries.

## Capabilities

- Classifies text into one or more states: neutral, depression, anxiety, stress
- Handles long journal entries and message histories

## Usage

Load and use the model with Hugging Face Transformers:

```python
from transformers import LongformerTokenizerFast, LongformerForSequenceClassification
import torch

model_name = "Hrishikesh4/mental-health-classifier-longformer"
model = LongformerForSequenceClassification.from_pretrained(model_name)
tokenizer = LongformerTokenizerFast.from_pretrained(model_name)

text = "I’ve been feeling overwhelmed with work and can’t sleep properly."
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.sigmoid(outputs.logits)

print(predictions)
```

### Example Queries

- “I feel like I’m drowning in stress lately.”
- “I’m anxious about meeting new people.”
- “Sometimes I just feel empty and hopeless.”
- “I’m doing okay today.”

## Limitations

- Non-clinical tool; not a substitute for professional therapy or diagnosis
- May reflect biases in training data
- Complex emotions may not map cleanly to defined labels

## Ethical Considerations

Use responsibly and communicate that this is not a medical tool. Users in distress should be directed to qualified professionals. Integrations should include user consent, privacy protection, and crisis resources.

## Citation

If you use this model, please cite:

```bibtex
@misc{kurapati2025MentalHealthSentimentAnalyzer,
    author = {Hrishikesh Kurapati},
    title = {Mental Health Sentiment Analyzer (Longformer-based)},
    year = {2025},
    url = {https://huggingface.co/Hrishikesh4/mental-health-classifier-longformer}
}
```