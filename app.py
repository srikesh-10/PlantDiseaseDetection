"""Streamlit web application for AI Plant Disease Detection.

Phase 6 — Provides a professional web interface for uploading leaf images,
running MobileNetV2-based disease predictions, and browsing prediction
history stored in SQLite.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import tempfile
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

# ------------------------------------------------------------------
# Project imports — reuse existing prediction & knowledge-base logic
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

# Confidence thresholds for colour-coded feedback
_HIGH_CONFIDENCE: float = 80.0
_MEDIUM_CONFIDENCE: float = 50.0


# ------------------------------------------------------------------
# Cached resource — load heavy model assets only once per session
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading AI model …")
def _get_predictor() -> PlantDiseasePredictor:
    """Instantiate the prediction engine (cached across reruns).

    Returns:
        A ready-to-use ``PlantDiseasePredictor`` instance.
    """
    return PlantDiseasePredictor()


# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="AI Plant Disease Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------------
# Custom CSS for a polished look
# ------------------------------------------------------------------
def _inject_custom_css() -> None:
    """Inject minimal custom CSS for visual refinement."""
    st.markdown(
        """
        <style>
        /* ---- Global tweaks ---- */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        /* ---- Header area ---- */
        .app-header {
            text-align: center;
            padding: 1.2rem 0 0.6rem;
        }
        .app-header h1 {
            font-size: 2.4rem;
            margin-bottom: 0.2rem;
        }
        .app-header p {
            color: #888;
            font-size: 1.05rem;
            max-width: 720px;
            margin: 0 auto;
        }

        /* ---- Result card ---- */
        .result-card {
            background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
            border-radius: 12px;
            padding: 1.5rem;
            margin-top: 1rem;
            border-left: 5px solid #22c55e;
        }
        [data-theme="dark"] .result-card,
        .stApp[data-theme="dark"] .result-card {
            background: linear-gradient(135deg, #14532d33 0%, #16653733 100%);
        }

        /* ---- Sidebar section titles ---- */
        .sidebar-section {
            font-weight: 600;
            font-size: 0.95rem;
            margin-bottom: 0.3rem;
        }

        /* ---- Metric override ---- */
        [data-testid="stMetricValue"] {
            font-size: 2.2rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
def _render_sidebar() -> None:
    """Render the sidebar with project information and links."""
    with st.sidebar:
        st.image(
            "https://img.icons8.com/color/96/000000/plant-under-sun.png",
            width=80,
        )
        st.title("🌱 Plant Disease AI")
        st.caption("Powered by Deep Learning")
        st.divider()

        # -- Project Information --
        st.markdown("##### 📋 Project Information")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Model**")
            st.markdown("**Classes**")
            st.markdown("**Accuracy**")
            st.markdown("**Dataset**")
        with col2:
            st.markdown("MobileNetV2")
            st.markdown("6")
            st.markdown("93.3 %")
            st.markdown("PlantVillage")
        st.divider()

        # -- Supported Classes --
        st.markdown("##### 🌿 Detectable Diseases")
        st.markdown(
            """
            - 🥔 Potato — Early Blight
            - 🥔 Potato — Late Blight
            - 🥔 Potato — Healthy
            - 🍅 Tomato — Early Blight
            - 🍅 Tomato — Late Blight
            - 🍅 Tomato — Healthy
            """
        )
        st.divider()

        # -- Instructions --
        st.markdown("##### 🚀 How to Use")
        st.markdown(
            """
            1. Upload a leaf image
            2. Click **🔬 Analyze Leaf**
            3. View diagnosis & recommendations
            """
        )
        st.divider()
        st.caption("© 2026 AI Plant Disease Detection")


# ------------------------------------------------------------------
# Image validation
# ------------------------------------------------------------------
def _validate_image(uploaded_file: Any) -> Image.Image | None:
    """Validate the uploaded file is a readable image.

    Args:
        uploaded_file: Streamlit ``UploadedFile`` object.

    Returns:
        A PIL ``Image`` if valid, otherwise ``None`` (error messages are
        displayed via Streamlit in-place).
    """
    if uploaded_file is None:
        return None

    # Check file extension
    filename: str = uploaded_file.name
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in _VALID_EXTENSIONS:
        st.error(
            f"❌ Unsupported file type: `.{ext}`.  "
            f"Please upload one of: {', '.join(sorted(_VALID_EXTENSIONS))}."
        )
        return None

    # Attempt to open with PIL
    try:
        image = Image.open(uploaded_file)
        image.verify()
        # After verify() the file pointer is consumed — re-open.
        uploaded_file.seek(0)
        image = Image.open(uploaded_file)
        return image
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        st.error(f"❌ Unable to read image: {exc}")
        LOGGER.warning("Image validation failed for %s: %s", filename, exc)
        return None


# ------------------------------------------------------------------
# Confidence helpers
# ------------------------------------------------------------------
def _confidence_emoji(confidence: float) -> str:
    """Return an emoji matching the confidence level."""
    if confidence >= _HIGH_CONFIDENCE:
        return "🟢"
    if confidence >= _MEDIUM_CONFIDENCE:
        return "🟡"
    return "🔴"


def _show_confidence_feedback(confidence: float) -> None:
    """Display contextual feedback based on the confidence score."""
    if confidence >= _HIGH_CONFIDENCE:
        st.success(f"✅ High confidence ({confidence:.1f} %) — the model is very sure about this diagnosis.")
    elif confidence >= _MEDIUM_CONFIDENCE:
        st.warning(f"⚠️ Moderate confidence ({confidence:.1f} %) — consider consulting an expert for confirmation.")
    else:
        st.error(f"🚨 Low confidence ({confidence:.1f} %) — the result may be unreliable. Please try a clearer image.")


# ------------------------------------------------------------------
# Tab 1 — Disease Detection
# ------------------------------------------------------------------
def _render_detection_tab() -> None:
    """Render the main disease detection interface."""

    # -- Image upload area --
    st.markdown("### 📸 Upload Leaf Image")
    uploaded_file = st.file_uploader(
        "Choose a leaf image for analysis",
        type=list(_VALID_EXTENSIONS),
        help="Supported formats: PNG, JPG, JPEG, WebP, BMP",
        key="leaf_uploader",
    )

    if uploaded_file is not None:
        image = _validate_image(uploaded_file)
        if image is None:
            return  # validation already showed an error

        # Preview layout
        col_img, col_info = st.columns([1, 1])
        with col_img:
            st.image(image, caption="📷 Uploaded Leaf", use_container_width=True)
        with col_info:
            st.markdown("#### 📄 Image Details")
            st.markdown(f"- **Filename:** `{uploaded_file.name}`")
            st.markdown(f"- **Dimensions:** {image.size[0]} × {image.size[1]} px")
            st.markdown(f"- **Format:** {image.format or 'N/A'}")
            st.markdown(f"- **Size:** {uploaded_file.size / 1024:.1f} KB")

        st.markdown("---")

        # -- Predict button --
        if st.button("🔬 Analyze Leaf", type="primary", use_container_width=True):
            _run_prediction(uploaded_file, image)
    else:
        # Placeholder when no image is uploaded
        st.info("ℹ️ Upload a leaf image above to get started with disease detection.")


def _run_prediction(uploaded_file: Any, image: Image.Image) -> None:
    """Execute the prediction pipeline and display results.

    Args:
        uploaded_file: Streamlit ``UploadedFile`` object.
        image: Validated PIL ``Image``.
    """
    with st.spinner("🧠 Analyzing leaf image with AI model …"):
        try:
            # Save the upload to a temporary file (predictor requires a path)
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                uploaded_file.seek(0)
                tmp.write(uploaded_file.read())
                tmp_path = Path(tmp.name)

            # Run prediction via the existing engine
            predictor = _get_predictor()
            result: dict[str, Any] = predictor.predict(tmp_path)

            # Clean up the temporary file
            try:
                tmp_path.unlink()
            except OSError:
                pass

        except FileNotFoundError as exc:
            st.error(f"❌ Model file not found: {exc}")
            LOGGER.exception("Prediction failed — model missing")
            return
        except ValueError as exc:
            st.error(f"❌ Prediction error: {exc}")
            LOGGER.exception("Prediction failed — value error")
            return
        except Exception as exc:
            st.error(f"❌ An unexpected error occurred: {exc}")
            LOGGER.exception("Prediction failed — unexpected error")
            return

    # -- Display results --
    _render_results(result)


def _render_results(result: dict[str, Any]) -> None:
    """Render the prediction results card.

    Args:
        result: Dictionary returned by ``PlantDiseasePredictor.predict()``.
    """
    predicted_class: str = result["predicted_class"]
    confidence: float = result["confidence"]
    display_name: str = DiseaseInfoManager.format_display_name(predicted_class)

    st.markdown("---")
    st.markdown("## 🩺 Diagnosis Results")

    # -- Result header with metric --
    st.markdown(
        f'<div class="result-card">'
        f"<h3>{_confidence_emoji(confidence)} {display_name}</h3>"
        f"</div>",
        unsafe_allow_html=True,
    )

    col_metric, col_feedback = st.columns([1, 2])
    with col_metric:
        st.metric(
            label="Confidence Score",
            value=f"{confidence:.1f} %",
            delta=None,
        )
    with col_feedback:
        _show_confidence_feedback(confidence)

    # -- Determine if healthy --
    is_healthy = "healthy" in predicted_class.lower()

    if is_healthy:
        st.success("🎉 Great news! This plant appears to be **healthy**. No disease detected.")
    else:
        st.warning("⚠️ Disease detected — review the details below for treatment guidance.")

    # -- Expandable detail sections --
    symptoms: list[str] = result.get("symptoms", ["N/A"])
    treatment: list[str] = result.get("treatment", ["N/A"])
    prevention: list[str] = result.get("prevention", ["N/A"])

    with st.expander("🔍 Symptoms", expanded=not is_healthy):
        for item in symptoms:
            st.markdown(f"- {item}")

    with st.expander("💊 Treatment", expanded=not is_healthy):
        for item in treatment:
            st.markdown(f"- {item}")

    with st.expander("🛡️ Prevention", expanded=True):
        for item in prevention:
            st.markdown(f"- {item}")

    # -- All class probabilities --
    with st.expander("📊 All Class Probabilities"):
        probs: dict[str, float] = result.get("all_probabilities", {})
        if probs:
            prob_df = pd.DataFrame(
                [
                    {
                        "Class": DiseaseInfoManager.format_display_name(cls),
                        "Confidence (%)": prob,
                    }
                    for cls, prob in sorted(
                        probs.items(), key=lambda x: x[1], reverse=True
                    )
                ]
            )
            st.dataframe(prob_df, use_container_width=True, hide_index=True)
            st.bar_chart(
                prob_df.set_index("Class"),
                use_container_width=True,
            )


# ------------------------------------------------------------------
# Tab 2 — Prediction History
# ------------------------------------------------------------------
def _render_history_tab() -> None:
    """Render the prediction history from the SQLite database."""

    st.markdown("### 📜 Prediction History")
    st.caption("Recent predictions stored in the local SQLite database.")

    try:
        history = get_prediction_history(limit=100, database_path=DATABASE_PATH)
    except Exception as exc:
        st.error(f"❌ Unable to load prediction history: {exc}")
        LOGGER.exception("Failed to read prediction history")
        return

    if not history:
        st.info("ℹ️ No predictions recorded yet. Upload an image in the **Disease Detection** tab to get started!")
        return

    # Build a DataFrame for display
    rows = []
    for record in history:
        rows.append(
            {
                "ID": record.get("id", "—"),
                "Timestamp": record.get("timestamp", "—"),
                "Predicted Class": DiseaseInfoManager.format_display_name(
                    record.get("predicted_class", "Unknown"),
                ),
                "Confidence (%)": record.get("confidence", 0.0),
            }
        )

    df = pd.DataFrame(rows)

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Predictions", len(df))
    with col2:
        avg_conf = df["Confidence (%)"].mean()
        st.metric("Average Confidence", f"{avg_conf:.1f} %")
    with col3:
        most_common = df["Predicted Class"].mode()
        st.metric(
            "Most Common",
            most_common.iloc[0] if not most_common.empty else "—",
        )

    st.divider()
    st.dataframe(df, use_container_width=True, hide_index=True)


# ------------------------------------------------------------------
# Main application entry point
# ------------------------------------------------------------------
def main() -> None:
    """Assemble and run the Streamlit application."""
    _inject_custom_css()
    _render_sidebar()

    # -- Header --
    st.markdown(
        '<div class="app-header">'
        "<h1>🌱 AI Plant Disease Detection System</h1>"
        "<p>Upload a leaf image and receive disease diagnosis, confidence score, "
        "symptoms, treatment, and prevention information.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # -- Tabs --
    tab_detect, tab_history = st.tabs(["🔬 Disease Detection", "📜 Prediction History"])

    with tab_detect:
        _render_detection_tab()

    with tab_history:
        _render_history_tab()


if __name__ == "__main__":
    main()
