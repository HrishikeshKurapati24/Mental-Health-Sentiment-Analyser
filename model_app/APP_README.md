This app has been deployed in Hugging Face Spaces at this link - https://huggingface.co/spaces/Hrishikesh4/mental-health-emotion-classifier.

NOTE: Running the app in the space is taking a lot of time for model inference. I suggest you to download the model and app files and run the app locally.

# Mental Health Sentiment Analyzer App

The Mental Health Sentiment Analyzer is an AI-powered application that analyzes user text to detect potential signs of mental health states such as depression, anxiety, and stress. The app leverages a fine-tuned Longformer model for multi-label text classification, built with Hugging Face Transformers, and provides explainable predictions with SHAP, along with concise summaries using Cohere API.

This project is deployed on Hugging Face Spaces with a Gradio-based interface.

---

## 🚀 Features

- Multi-label classification of user text into categories: depression, anxiety, stress, neutral.
- Explainable AI: SHAP visualizations for understanding model predictions.
- Summary generation for analyzed text using Cohere API.
- User-friendly web interface built with Gradio.
- Deployable on Hugging Face Spaces.

---

## 🛠️ Tech Stack

- **Language Model:** Fine-tuned Longformer
- **Summarization:** Cohere API
- **Frameworks:** Hugging Face Transformers, PyTorch
- **Interface:** Gradio
- **Explainability:** SHAP
- **Deployment:** Hugging Face Spaces

---

## 📂 Project Structure

```
Mental-Health-Sentiment-Analyzer/
│
├── app/
│   ├── gradio_interface.py      # Gradio UI definition
│   ├── models/                  # Trained model files
│   │   ├── __init__.py
│   │   ├── best_thresholds.json
│   │   └── sentiment_model.py
│   ├── utils/                   # Explanation and summarizer files
│   │   ├── __init__.py
│   │   ├── shap_explainer.py
│   │   └── text_summarizer.py
│
├── requirements.txt             # Dependencies
├── app.py                       # Entry point for Hugging Face Spaces
├── README.md                    # Project documentation
```

---


## 🤝 Contributors

- Hrishikesh Kurapati (Lead Developer)

---

## 📜 License

mit

---

## 🔮 Future Work

- Add support for more mental health categories.
- Improve explainability features.
- Expand dataset for better generalization.

---

## 🙏 Acknowledgements

- Hugging Face for Transformers & Spaces.
- Gradio for interactive UI.

---

> “The model is designed for early detection of mental health risks and can be integrated into wellness or therapy support applications. It works by analyzing a user’s voluntary text inputs — such as journals, self-reflections, or chatbot interactions — and identifying patterns of stress, anxiety, or depression. Depending on the user’s consent settings, it can either provide self-help recommendations, notify a therapist, or in high-risk cases, alert a pre-approved emergency contact. The goal is to empower timely intervention while ensuring privacy and ethical use.”