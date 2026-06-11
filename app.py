"""Streamlit web application for AI Plant Disease Detection.

Professional agriculture technology platform interface.
Runs MobileNetV2 predictions and stores history in SQLite.
"""

from __future__ import annotations

import tempfile
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

# Try to import Plotly for advanced visualization; fall back gracefully if unavailable
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ------------------------------------------------------------------
# Project imports
# ------------------------------------------------------------------
from src.predictor import PlantDiseasePredictor
from src.disease_info import DiseaseInfoManager
from database import get_prediction_history
from config import DATABASE_PATH

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
LOGGER = logging.getLogger("streamlit_app")
logging.basicConfig(level=logging.INFO)


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
_VALID_EXTENSIONS: set[str] = {"png", "jpg", "jpeg", "webp", "bmp"}


# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Plant Health Intelligence",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------------
# Cached resource
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing AI Model...")
def _get_predictor() -> PlantDiseasePredictor:
    return PlantDiseasePredictor()


# ------------------------------------------------------------------
# Custom CSS
# ------------------------------------------------------------------
def _inject_custom_css() -> None:
    # Streamlit-compatible styling
    # Avoid unsupported widget arguments by using CSS injection instead
    st.markdown(
        """
        <style>
        /* ---- Color Variables ---- */
        :root {
            --bg-color: #F7F3EC;
            --card-bg: #FFFDF8;
            --primary: #6B8E23;
            --secondary: #7A9E7E;
            --accent: #D6C7A1;
            --border: #E5DCCB;
            --text-main: #2F3E2E;
            --text-muted: #5E6D5D;
            --success: #4E944F;
            --warning: #D4A017;
            --error: #C94C4C;
        }

        /* ---- Global Overrides ---- */
        .stApp, .stApp > header {
            background-color: var(--bg-color);
            color: var(--text-main);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen-Sans, Ubuntu, Cantarell, "Helvetica Neue", sans-serif;
        }
        
        h1, h2, h3, h4, h5, h6, p, span, div {
            color: var(--text-main);
        }

        /* Make sidebar match theme */
        [data-testid="stSidebar"] {
            background-color: var(--card-bg) !important;
            border-right: 1px solid var(--border) !important;
        }

        /* Reduce vertical spacing by approx 30% */
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 2rem !important;
            max-width: 1000px;
        }

        /* ---- Cards & Bordered Containers ---- */
        .agri-card, div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: var(--card-bg) !important;
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
            padding: 1.25rem !important;
            box-shadow: 0 4px 6px rgba(47, 62, 46, 0.08) !important;
            margin-bottom: 1rem !important;
        }

        /* ---- Header ---- */
        .hero-section {
            text-align: center;
            padding: 1rem 0;
            margin-bottom: 1rem;
            border-bottom: 1px solid var(--border);
        }

        .header-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: var(--primary) !important;
            margin-bottom: 0.25rem;
        }
        
        .header-subtitle {
            color: var(--text-muted) !important;
            font-size: 1.05rem;
            font-weight: 400;
            max-width: 700px;
            margin: 0 auto;
        }

        /* ---- Custom Progress Bar ---- */
        .stProgress > div > div > div > div {
            background-color: var(--primary) !important;
        }

        /* ---- DataFrames / Tables ---- */
        .stDataFrame {
            border-radius: 8px;
            border: 1px solid var(--border);
            background-color: var(--card-bg);
        }

        /* ---- Info Sections ---- */
        .info-header {
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--primary) !important;
            margin-bottom: 0.75rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .info-list {
            color: var(--text-main) !important;
            font-size: 0.95rem;
            line-height: 1.5;
            margin-bottom: 0.4rem;
        }

        /* ---- File Uploader Light Card Style ---- */
        [data-testid="stFileUploader"] {
            background-color: var(--card-bg);
            border: 2px dashed var(--secondary);
            border-radius: 12px;
            padding: 1.5rem;
        }
        [data-testid="stFileUploader"] * {
            color: var(--text-main) !important;
        }
        [data-testid="stFileUploader"] button {
            background-color: var(--primary) !important;
            color: #FFFFFF !important;
            border: none;
        }

        /* ---- Tabs ---- */
        [data-baseweb="tab-list"] {
            border-bottom: 2px solid var(--border);
            margin-bottom: 1rem;
        }
        [data-baseweb="tab"] {
            color: var(--text-muted) !important;
            font-weight: 600 !important;
        }
        [data-baseweb="tab"][aria-selected="true"] {
            color: var(--primary) !important;
            border-bottom-color: var(--primary) !important;
        }

        /* ---- Image Styling ---- */
        [data-testid="stImage"] img {
            border-radius: 8px;
            border: 1px solid var(--border);
        }

        /* ---- Footer ---- */
        .footer {
            text-align: center;
            margin-top: 3rem;
            padding-top: 1.5rem;
            border-top: 1px solid var(--border);
            color: var(--text-muted) !important;
            font-size: 0.85rem;
        }
        
        /* Metric overriding for dashboard look */
        [data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
            font-weight: 700 !important;
            color: var(--primary) !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.95rem !important;
            color: var(--text-muted) !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("<h3 style='color: var(--primary);'>Plant Health</h3>", unsafe_allow_html=True)
        st.markdown("<div style='color: var(--text-muted); font-size: 0.9rem; margin-bottom: 1rem;'>Project Information</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style='font-size: 0.9rem; color: var(--text-main); padding-bottom: 1rem; border-bottom: 1px solid var(--border);'>
            <div style='display: flex; justify-content: space-between; margin-bottom: 0.5rem;'>
                <span style='color: var(--text-muted);'>Model</span>
                <span style='font-weight: 500;'>MobileNetV2</span>
            </div>
            <div style='display: flex; justify-content: space-between; margin-bottom: 0.5rem;'>
                <span style='color: var(--text-muted);'>Accuracy</span>
                <span style='font-weight: 500; color: var(--success);'>93.3%</span>
            </div>
            <div style='display: flex; justify-content: space-between; margin-bottom: 0.5rem;'>
                <span style='color: var(--text-muted);'>Classes</span>
                <span style='font-weight: 500;'>6</span>
            </div>
            <div style='display: flex; justify-content: space-between;'>
                <span style='color: var(--text-muted);'>Dataset</span>
                <span style='font-weight: 500;'>PlantVillage</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("Supported Diseases", expanded=False):
            st.markdown("""
            - Potato Early Blight
            - Potato Late Blight
            - Potato Healthy
            - Tomato Early Blight
            - Tomato Late Blight
            - Tomato Healthy
            """)
            
        with st.expander("Usage Guide", expanded=False):
            st.markdown("""
            1. Upload a clear photo of a single leaf.
            2. The AI will analyze the image.
            3. Review the diagnosis and confidence score.
            4. Follow treatment/prevention guidelines.
            """)


# ------------------------------------------------------------------
# Image validation
# ------------------------------------------------------------------
def _validate_image(uploaded_file: Any) -> Image.Image | None:
    if uploaded_file is None:
        return None

    filename: str = uploaded_file.name
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in _VALID_EXTENSIONS:
        st.error(f"Unsupported file type: `.{ext}`.")
        return None

    try:
        image = Image.open(uploaded_file)
        image.verify()
        uploaded_file.seek(0)
        image = Image.open(uploaded_file)
        return image
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        st.error("Unable to read image file.")
        LOGGER.warning("Image validation failed for %s: %s", filename, exc)
        return None


# ------------------------------------------------------------------
# Interpretation Helper
# ------------------------------------------------------------------
def _get_confidence_interpretation(confidence: float) -> tuple[str, str]:
    if confidence >= 95.0:
        return "High Confidence", "The model is highly certain of this diagnosis."
    elif confidence >= 80.0:
        return "Medium Confidence", "The model is fairly confident, but consider verifying."
    else:
        return "Low Confidence", "The model is uncertain. Results may be unreliable."


# ------------------------------------------------------------------
# Tab 1 — Main Detection Page
# ------------------------------------------------------------------
def _render_detection_tab() -> None:
    # Initialize session state for prediction results to avoid recalculating on rerun
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None
    if "last_file" not in st.session_state:
        st.session_state.last_file = None

    # 1. Clean uploader container
    uploaded_file = st.file_uploader(
        "Upload Leaf Image",
        type=list(_VALID_EXTENSIONS),
        help="Upload a plant leaf image (PNG, JPG, WEBP) to begin diagnosis.",
        label_visibility="hidden"
    )

    image = None
    if uploaded_file is not None:
        image = _validate_image(uploaded_file)
        
    # Reset prediction result if the user uploads a different file
    if uploaded_file != st.session_state.last_file:
        st.session_state.prediction_result = None
        st.session_state.last_file = uploaded_file
        
    if uploaded_file and image:
        st.markdown("<hr style='margin: 1rem 0; border-color: var(--border);'>", unsafe_allow_html=True)
        # 2. Two-column layout
        col_left, col_right = st.columns([1, 1], gap="large")

        with col_left:
            st.markdown("<div class='info-header' style='border:none; margin:0;'>Uploaded Image</div>", unsafe_allow_html=True)
            # Streamlit-compatible styling: No 'style' argument
            st.image(image, use_container_width=True)
            
            # Metadata
            m1, m2, m3 = st.columns(3)
            m1.caption(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
            m2.caption(f"**Dims:** {image.size[0]}x{image.size[1]}")
            m3.caption(f"**Fmt:** {image.format}")
            
            if st.button("Run AI Diagnosis", type="primary", use_container_width=True):
                st.session_state.prediction_result = _run_prediction(uploaded_file)

        with col_right:
            if st.session_state.prediction_result is not None:
                _render_prediction_summary(st.session_state.prediction_result)
            else:
                st.info("Click 'Run AI Diagnosis' to analyze the leaf image.")

        # Full-width detailed sections below the main columns to provide maximum space for information
        if st.session_state.prediction_result is not None:
            st.markdown("<hr style='margin: 1.5rem 0; border-color: var(--border);'>", unsafe_allow_html=True)
            _render_detailed_diagnosis(st.session_state.prediction_result)


def _run_prediction(uploaded_file: Any) -> dict[str, Any] | None:
    with st.spinner("Analyzing leaf patterns..."):
        try:
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                uploaded_file.seek(0)
                tmp.write(uploaded_file.read())
                tmp_path = Path(tmp.name)

            predictor = _get_predictor()
            result: dict[str, Any] = predictor.predict(tmp_path)

            try:
                tmp_path.unlink()
            except OSError:
                pass

            return result
            
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            LOGGER.exception("Prediction failed")
            return None


def _render_prediction_summary(result: dict[str, Any]) -> None:
    predicted_class: str = result["predicted_class"]
    confidence: float = result["confidence"]
    display_name: str = DiseaseInfoManager.format_display_name(predicted_class)
    
    interp_label, interp_desc = _get_confidence_interpretation(confidence)
    
    # Diagnosis Card using native Streamlit container
    with st.container(border=True):
        st.markdown("<div class='info-header' style='border-bottom: none; margin-bottom: 0;'>Prediction Summary</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 1.4rem; font-weight: 600; margin-bottom: 1rem;'>{display_name}</div>", unsafe_allow_html=True)
        st.metric("Confidence Score", f"{confidence:.1f}%", delta=interp_label, delta_color="normal" if confidence >= 80 else "inverse")
        st.progress(confidence / 100.0)
        st.caption(interp_desc)


def _render_detailed_diagnosis(result: dict[str, Any]) -> None:
    predicted_class: str = result["predicted_class"]
    is_healthy = "healthy" in predicted_class.lower()
    
    # Information section (Symptoms, Treatment, Prevention)
    st.markdown("#### Disease Information")
    c1, c2, c3 = st.columns(3, gap="medium")
    
    with c1:
        symptoms_html = "<div class='agri-card' style='height: 100%;'><div class='info-header'>Symptoms</div>"
        if is_healthy:
            symptoms_html += "<div class='info-list'>None. Plant is healthy.</div>"
        else:
            for item in result.get("symptoms", ["N/A"]):
                symptoms_html += f"<div class='info-list'>• {item}</div>"
        symptoms_html += "</div>"
        st.markdown(symptoms_html, unsafe_allow_html=True)

    with c2:
        treatment_html = "<div class='agri-card' style='height: 100%;'><div class='info-header'>Treatment</div>"
        if is_healthy:
            treatment_html += "<div class='info-list'>No treatment required.</div>"
        else:
            for item in result.get("treatment", ["N/A"]):
                treatment_html += f"<div class='info-list'>• {item}</div>"
        treatment_html += "</div>"
        st.markdown(treatment_html, unsafe_allow_html=True)

    with c3:
        prevention_html = "<div class='agri-card' style='height: 100%;'><div class='info-header'>Prevention</div>"
        for item in result.get("prevention", ["N/A"]):
            prevention_html += f"<div class='info-list'>• {item}</div>"
        prevention_html += "</div>"
        st.markdown(prevention_html, unsafe_allow_html=True)

    # Probability breakdown
    st.markdown("#### Prediction Breakdown")
    probs: dict[str, float] = result.get("all_probabilities", {})
    if probs:
        with st.container(border=True):
            # Attempt to render interactive Plotly chart if library is available
            use_plotly = False
            if PLOTLY_AVAILABLE:
                try:
                    # Prepare data sorted for Plotly (highest probability at the top of horizontal chart)
                    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=False)
                    labels = [DiseaseInfoManager.format_display_name(cls) for cls, _ in sorted_probs]
                    values = [prob for _, prob in sorted_probs]
                    
                    df_chart = pd.DataFrame({
                        'Disease': labels,
                        'Probability (%)': values
                    })
                    
                    # Plotly chart using plotly_white theme
                    fig = px.bar(df_chart, x='Probability (%)', y='Disease', orientation='h')
                    fig.update_layout(
                        template="plotly_white",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=0, r=0, t=10, b=0),
                        height=250,
                        font=dict(color="#2F3E2E", family="sans-serif")
                    )
                    fig.update_traces(marker_color="#6B8E23")
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                    use_plotly = True
                except Exception as exc:
                    LOGGER.warning("Plotly rendering failed, falling back to native Streamlit chart: %s", exc)
                    use_plotly = False
            
            # Fallback to native Streamlit bar chart or DataFrame if Plotly is unavailable or fails
            if not use_plotly:
                try:
                    # Prepare data sorted descending (highest probability first) for Streamlit native bar chart
                    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                    labels = [DiseaseInfoManager.format_display_name(cls) for cls, _ in sorted_probs]
                    values = [prob for _, prob in sorted_probs]
                    
                    df_chart = pd.DataFrame({
                        'Disease': labels,
                        'Probability (%)': values
                    })
                    
                    # Use native Streamlit bar chart with the primary theme color
                    st.bar_chart(
                        df_chart,
                        x='Disease',
                        y='Probability (%)',
                        color='#6B8E23',
                        use_container_width=True
                    )
                except Exception as exc:
                    LOGGER.error("Streamlit bar_chart rendering failed, falling back to DataFrame table: %s", exc)
                    # Last resort fallback: Simple DataFrame table
                    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                    df_table = pd.DataFrame([
                        {
                            "Disease": DiseaseInfoManager.format_display_name(cls),
                            "Probability": f"{prob:.2f}%"
                        }
                        for cls, prob in sorted_probs
                    ])
                    st.dataframe(df_table, use_container_width=True, hide_index=True)


# ------------------------------------------------------------------
# Tab 2 — Prediction History
# ------------------------------------------------------------------
def _render_history_tab() -> None:
    try:
        history = get_prediction_history(limit=100, database_path=DATABASE_PATH)
    except Exception as exc:
        st.error("Unable to load prediction history.")
        LOGGER.exception("Failed to read history")
        return

    if not history:
        st.info("No predictions recorded yet.")
        return

    # Process data
    rows = []
    for record in history:
        rows.append(
            {
                "ID": record.get("id", "—"),
                "Date": record.get("timestamp", "—")[:16], # trim seconds for cleaner look
                "Predicted Class": DiseaseInfoManager.format_display_name(record.get("predicted_class", "Unknown")),
                "Confidence": record.get("confidence", 0.0),
            }
        )
    df = pd.DataFrame(rows)

    # Metrics
    st.markdown("#### Overview Metrics")
    m1, m2, m3 = st.columns(3)
    with m1:
        with st.container(border=True):
            st.metric("Total Predictions", len(df))
    with m2:
        with st.container(border=True):
            avg_conf = df["Confidence"].mean()
            st.metric("Average Confidence", f"{avg_conf:.1f}%")
    with m3:
        with st.container(border=True):
            latest_pred = df.iloc[0]["Predicted Class"] if not df.empty else "—"
            st.metric("Latest Prediction", latest_pred)

    # Data Table
    st.markdown("#### Prediction History Log")
    with st.container(border=True):
        st.dataframe(
            df, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence",
                    format="%f%%",
                    min_value=0,
                    max_value=100,
                )
            }
        )


# ------------------------------------------------------------------
# Main application
# ------------------------------------------------------------------
def main() -> None:
    _inject_custom_css()
    _render_sidebar()

    # Hero Section
    st.markdown("""
    <div class='hero-section'>
        <div class='header-title'>Plant Health Intelligence</div>
        <div class='header-subtitle'>
            Upload a leaf image and receive AI-powered disease diagnosis, 
            treatment recommendations, and prevention guidance.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tabs
    tab_detect, tab_history = st.tabs(["Diagnosis", "Analytics & History"])

    with tab_detect:
        _render_detection_tab()

    with tab_history:
        _render_history_tab()

    # Footer
    st.markdown("""
    <div class='footer'>
        <strong>Plant Health Intelligence</strong><br>
        Powered by TensorFlow • MobileNetV2 • Streamlit<br>
        Built by Srikesh
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
