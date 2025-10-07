from __future__ import annotations
import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Any
from app.models.sentiment_model import model
from app.utils.shap_explainer import explainer
from app.utils.text_summarizer import summarizer

# -----------------------------
# Visualization helpers
# -----------------------------

def _sorted_predictions(predictions: Dict[str, float]) -> Dict[str, float]:
    """Return predictions sorted by confidence (desc)."""
    return dict(sorted(predictions.items(), key=lambda kv: kv[1], reverse=True))

def create_prediction_table(predictions: Dict[str, float]) -> pd.DataFrame:
    preds_sorted = _sorted_predictions(predictions)
    df = pd.DataFrame({"Label": list(preds_sorted.keys()), "Confidence": list(preds_sorted.values())})
    return df

def create_prediction_chart(predictions: Dict[str, float]) -> go.Figure:
    labels = list(predictions.keys())
    values = list(predictions.values())

    colors = ["#6366F1", "#06B6D4", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#14B8A6"]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color=colors[: len(labels)],
                text=[f"{v:.1%}" for v in values],
                textposition="auto",
                hovertemplate="<b>%{x}</b><br>Confidence: %{y:.1%}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title="Mental Health Sentiment Predictions",
        xaxis_title="Mental Health Categories",
        yaxis_title="Confidence",
        yaxis_tickformat=".0%",
        yaxis_range=[0, 1],
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        height=420,
        margin=dict(l=30, r=20, t=60, b=40),
    )

    return fig

def _sorted_explanations(explanations: Dict[str, List[Tuple[str, float]]]) -> Dict[str, List[Tuple[str, float]]]:
    """Sort phrases in each label by absolute SHAP magnitude (desc)."""
    sorted_expls = {}
    for label, phrases in (explanations or {}).items():
        sorted_expls[label] = sorted(phrases, key=lambda p: abs(p[1]), reverse=True)

    return sorted_expls

def create_explanations_table(explanations: Dict[str, List[Tuple[str, float]]], top_k: int = 8) -> pd.DataFrame:
    if not explanations:
        return pd.DataFrame(columns=["Label", "Phrase", "SHAP Score"])
    rows = []
    for label, phrases in _sorted_explanations(explanations).items():
        for phrase, score in phrases[:top_k]:
            rows.append({"Label": label, "Phrase": phrase, "SHAP Score": float(score)})
    df = pd.DataFrame(rows)

    return df

def create_explanation_heatmap(explanations: Dict[str, List[Tuple[str, float]]]) -> go.Figure:
    if not explanations:
        return go.Figure()

    sorted_expls = _sorted_explanations(explanations)
    labels = list(sorted_expls.keys())
    max_phrases = max(len(v) for v in sorted_expls.values()) if labels else 0

    # Build matrix and x-axis labels from the unioned top-N positions (position-based view)
    matrix: List[List[float]] = []
    x_labels: List[str] = []

    # Construct x tick labels from first label's phrases (positional), fallback to index
    ref_phrases = sorted_expls[labels[0]] if labels else []
    for i in range(max_phrases):
        if i < len(ref_phrases):
            phrase = ref_phrases[i][0]
            x_labels.append((phrase[:24] + "...") if len(phrase) > 24 else phrase)
        else:
            x_labels.append(f"Top {i+1}")

    for label in labels:
        phrases = sorted_expls[label]
        row = []
        for i in range(max_phrases):
            if i < len(phrases):
                row.append(float(phrases[i][1]))
            else:
                row.append(0.0)
        matrix.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=x_labels,
            y=labels,
            colorscale="RdBu",
            reversescale=True,  # positive vs. negative pops better in many palettes
            zmid=0,
            hovertemplate="<b>%{y}</b><br>Pos #%{x}<br>SHAP: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="SHAP Explanation Heatmap (positional top terms)",
        xaxis_title="Top terms by |SHAP| (per label)",
        yaxis_title="Labels",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=420,
        margin=dict(l=30, r=20, t=60, b=40),
    )
    
    return fig


# -----------------------------
# Core analysis function
# -----------------------------

def analyze_text(text: str) -> tuple[Any, Any, str, Any, Any]:
    """Run the full pipeline and return UI-friendly objects.

    Returns:
        - predictions_table (pd.DataFrame)
        - explanations_table (pd.DataFrame)
        - summary_markdown (str)
        - prediction_chart (plotly.graph_objects.Figure)
        - explanation_heatmap (plotly.graph_objects.Figure)
    """
    if not text or not text.strip():
        empty_preds = pd.DataFrame(columns=["Label", "Confidence"])
        empty_expls = pd.DataFrame(columns=["Label", "Phrase", "SHAP Score"])
        return empty_preds, empty_expls, "", go.Figure(), go.Figure()

    try:
        # --- Backend calls ---
        model_preds = model.predict(text)
        predictions: Dict[str, float] = model_preds[0]
        binary_predictions = model_preds[1]

        explanations: Dict[str, List[Tuple[str, float]]] = explainer.explain_text(text, binary_predictions)
        summary: str = summarizer.generate_summary(text, predictions, explanations)
        # ---------------------------------

        # Visuals & Tables
        preds_sorted = _sorted_predictions(predictions)
        prediction_chart = create_prediction_chart(preds_sorted)
        predictions_table = create_prediction_table(preds_sorted)

        explanation_heatmap = create_explanation_heatmap(explanations)
        explanations_table = create_explanations_table(explanations, top_k=10)

        return (
            predictions_table,
            explanations_table,
            summary,
            prediction_chart,
            explanation_heatmap,
        )

    except Exception as e:
        # Surface clean errors in the UI
        raise gr.Error(f"Analysis failed: {e}")

def clear_outputs():
    return (
        pd.DataFrame(columns=["Label", "Confidence"]),
        pd.DataFrame(columns=["Label", "Phrase", "SHAP Score"]),
        "",
        None,
        None,
    )

# -----------------------------
# Gradio UI
# -----------------------------

def create_interface() -> gr.Blocks:
    custom_css = """
    .gradio-container {max-width: 1200px !important; margin: 0 auto !important;}
    .main-header {text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                  color: white; border-radius: 12px; margin-bottom: 16px;}
    """

    theme = gr.themes.Soft()

    with gr.Blocks(css=custom_css, theme=theme, title="Mental Health Sentiment Analyzer") as demo:
        # Header
        gr.HTML(
            """
            <div class="main-header">
              <h1>🧠 Mental Health Sentiment Analyzer</h1>
              <p>Analyze text for mental health indicators using fine-tuned Longformer</p>
            </div>
            """
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                gr.Markdown("### 📝 Enter Your Text")
                text_input = gr.Textbox(
                    label="Text to Analyze",
                    placeholder="Paste or type text here…",
                    lines=7,
                )
                with gr.Row():
                    analyze_btn = gr.Button("🔍 Analyze Text", variant="primary")
                    clear_btn = gr.Button("🧼 Clear")

                gr.Markdown("_💡 Tip: Longer texts can produce more reliable patterns._")

                # Click-to-run examples
                gr.Examples(
                    examples=[
                        "Lately I feel overwhelmed and panicky about small things. Sleep has been rough.",
                        "Feeling alright today. Work was manageable and I enjoyed lunch with friends.",
                        "Deadlines are stressing me out. Headaches and racing thoughts every night.",
                    ],
                    inputs=[text_input],
                    outputs=None,
                )

            with gr.Column(scale=1):
                with gr.Accordion("ℹ️ How it works", open=False):
                    gr.Markdown(
                        """
                        * The model predicts mental-health related categories with confidence.
                        * SHAP shows which phrases push predictions up or down.
                        * A concise, supportive summary is generated using Cohere API for quick understanding.
                        * **Note:** This tool has been developed as part of a project for early mental health risk detection, not medical diagnosis.
                        """
                    )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📊 Predictions")
                predictions_output = gr.Dataframe(
                    headers=["Label", "Confidence"],
                    datatype=["str", "number"],
                    interactive=False,
                    wrap=True,
                    elem_id="predictions_table",
                    label="Model Predictions (sorted)",
                )
                prediction_chart = gr.Plot(label="Prediction Chart")

            with gr.Column():
                gr.Markdown("### 🔍 SHAP Explanations")
                explanations_output = gr.Dataframe(
                    headers=["Label", "Phrase", "SHAP Score"],
                    datatype=["str", "str", "number"],
                    interactive=False,
                    wrap=True,
                    elem_id="explanations_table",
                    label="Top contributing phrases",
                )
                explanation_heatmap = gr.Plot(label="Explanation Heatmap")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📝 Analysis Summary")
                summary_output = gr.Markdown()

        gr.HTML(
            """
            <div style="text-align:center; padding: 16px; color:#666; border-top: 1px solid #eee; margin-top: 20px;">
              <p>Built with ❤️ for mental health awareness</p>
              <p><small>For educational and research use only. Not a substitute for professional advice.</small></p>
            </div>
            """
        )

        # Wiring
        analyze_btn.click(
            fn=analyze_text,
            inputs=[text_input],
            outputs=[
                predictions_output,
                explanations_output,
                summary_output,
                prediction_chart,
                explanation_heatmap,
            ],
        )

        text_input.submit(
            fn=analyze_text,
            inputs=[text_input],
            outputs=[
                predictions_output,
                explanations_output,
                summary_output,
                prediction_chart,
                explanation_heatmap,
            ],
        )

        clear_btn.click(
            fn=clear_outputs,
            inputs=None,
            outputs=[
                predictions_output,
                explanations_output,
                summary_output,
                prediction_chart,
                explanation_heatmap,
            ],
        )

    return demo