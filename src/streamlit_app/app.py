"""Streamlit UI to collect inputs and fetch house price predictions from a FastAPI service.

API contract (expected by backend /predict):
  Request (JSON):
    {
      sqft: int,
      bedrooms: int,
      bathrooms: float,
      location: str,
      year_built: int,
      condition: str
    }
  Response (JSON):
    {
      predicted_price: float,
      confidence_interval: [float, float],
      features_importance: { str: float },
      prediction_time: str
    }

Run:
  streamlit run src/streamlit_app/app.py

Environment variables:
  API_URL      Base URL of prediction API (default: http://localhost:8000)
  APP_VERSION  App version string for footer (default: 1.0.0)
"""

import os
import socket  # For hostname and IP address
import time

import requests
import streamlit as st
import yaml


# Set the page configuration (must be the first Streamlit command)
st.set_page_config(
    page_title="House Price Predictor", layout="wide", initial_sidebar_state="collapsed"
)


# Function to load model configuration
@st.cache_data
def load_model_config():
    """Load model configuration from YAML file"""
    try:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "configs", "model_config.yaml"
        )
        with open(config_path) as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        st.warning(f"Could not load model config: {e}")
        return {"model": {"best_model": "Unknown"}}


# Load model configuration
model_config = load_model_config()
best_model = model_config.get("model", {}).get("best_model", "Unknown")

# ---------- Global Styles (modern look & feel) ---------- #
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
      :root {
        --panel: #ffffff;
        --muted: #6b7280;
        --text: #111827;
        --border: #e5e7eb;
        --accent: #3b82f6;    /* blue-500 */
        --accent-2: #06b6d4;  /* cyan-500 */
      }
      html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
      .title-wrap { display:flex; align-items:center; gap:14px; }
      .brand-pill { background: linear-gradient(90deg,var(--accent),var(--accent-2)); color:#ffffff; padding:6px 10px; border-radius: 999px; font-size:12px; letter-spacing:.4px; font-weight:700; }
      .subtle { color:var(--muted); }
      .card { background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 18px 18px; box-shadow: 0 2px 12px rgba(0,0,0,0.06); }
      .primary-btn button { background: linear-gradient(90deg,var(--accent),var(--accent-2)) !important; color: #ffffff !important; border: none !important; font-weight:700; }
      .prediction-value { font-size: 44px; font-weight: 800; letter-spacing: .5px; background: linear-gradient(90deg,#111827,var(--accent)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
      .info-card { background: #f9fafb; border: 1px solid var(--border); border-radius: 12px; padding: 14px; }
      .info-label { color:var(--muted); margin:0; font-size: 12px; letter-spacing:.3px; }
      .info-value { color:#111827; margin:6px 0 0; font-weight:700; font-size: 18px; }
      .top-factors { background:#ffffff; border:1px dashed var(--border); border-radius: 12px; padding: 12px 16px; color:#111827; }
      .api-link { font-size:12px; color: var(--accent); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Title ---------- #
st.markdown(
    """
    <div class="title-wrap">
      <div class="brand-pill">MLOps Demo</div>
      <h2 style="margin:0">House Price Prediction</h2>
    </div>
    <p class="subtle" style="margin-top:8px">Real‑time predictions powered by FastAPI + Streamlit</p>
    """,
    unsafe_allow_html=True,
)

# Create a two-column layout
col1, col2 = st.columns(2, gap="large")

# Input form
with col1:

    # Square Footage slider
    sqft = st.slider("Square Footage", 500, 5000, 1500, 50)
    st.caption(f"{sqft} sq ft")
    st.markdown(
        f"<script>document.getElementById('sqft-value').innerText = '{sqft} sq ft';</script>",
        unsafe_allow_html=True,
    )

    # Bedrooms and Bathrooms in two columns
    bed_col, bath_col = st.columns(2)
    with bed_col:
        bedrooms = st.selectbox("Bedrooms", options=[1, 2, 3, 4, 5, 6], index=2)

    with bath_col:
        bathrooms = st.selectbox("Bathrooms", options=[1, 1.5, 2, 2.5, 3, 3.5, 4], index=2)

    # Location dropdown (aligned with API categories)
    location = st.selectbox(
        "Location",
        options=["Downtown", "Mountain", "Rural", "Suburb", "Urban", "Waterfront"],
        index=4,
    )

    # Year Built slider (UI allows 1800–2025; will be clamped for API)
    year_built = st.slider("Year Built", 1800, 2025, 2000, 1, key="year")

    # Condition selector
    condition = st.selectbox(
        "Condition",
        options=["Excellent", "Good", "Fair", "Poor"],
        index=1,
    )

    # Predict button
    predict_button = st.button("Predict Price", use_container_width=True, type="primary")

    # end left column

# Results section
with col2:
    st.markdown("<h2>Prediction Results</h2>", unsafe_allow_html=True)

    # If button is clicked, show prediction
    if predict_button:
        # Show loading spinner
        with st.spinner("Calculating prediction..."):
            # Prepare data for API call
            # Clamp year to API schema max (2023) to avoid validation errors
            safe_year_built = min(year_built, 2023)

            api_data = {
                "sqft": sqft,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "location": location,  # keep capitalization as API expects
                "year_built": safe_year_built,
                "condition": condition,
            }

            try:
                # Get API endpoint from environment variable or use default localhost
                api_endpoint = os.getenv("API_URL", "http://localhost:8005")
                predict_url = f"{api_endpoint.rstrip('/')}/predict"

                st.caption(
                    f"Connecting to API at: <a class='api-link' href='{predict_url}' target='_blank'>{predict_url}</a>",
                    unsafe_allow_html=True,
                )

                # Make API call to FastAPI backend
                response = requests.post(predict_url, json=api_data, timeout=10)
                response.raise_for_status()  # Raise exception for bad status codes
                prediction = response.json()

                # Store prediction in session state
                st.session_state.prediction = prediction
                st.session_state.prediction_time = time.time()
            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to API: {e}")
                st.warning(
                    "Using mock data for demonstration purposes. Please check your API connection."
                )
                # For demo purposes, use mock data if API fails
                st.session_state.prediction = {
                    "predicted_price": 467145,
                    "confidence_interval": [420430.5, 513859.5],
                    "features_importance": {
                        "sqft": 0.43,
                        "location": 0.27,
                        "bathrooms": 0.15,
                    },
                    "prediction_time": "0.12 seconds",
                }
                st.session_state.prediction_time = time.time()

    # Display prediction if available
    if "prediction" in st.session_state:
        pred = st.session_state.prediction

        # Format the predicted price
        formatted_price = "${:,.0f}".format(pred["predicted_price"])
        st.markdown(
            f'<div class="prediction-value">{formatted_price}</div>',
            unsafe_allow_html=True,
        )

        # Display confidence score and model used
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown('<p class="info-label">Confidence Score</p>', unsafe_allow_html=True)
            st.markdown('<p class="info-value">92%</p>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown('<p class="info-label">Model Used</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="info-value">{best_model}</p>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Display price range and prediction time
        col_c, col_d = st.columns(2)
        with col_c:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown('<p class="info-label">Price Range</p>', unsafe_allow_html=True)
            lower = "${:,.1f}".format(pred["confidence_interval"][0])
            upper = "${:,.1f}".format(pred["confidence_interval"][1])
            st.markdown(f'<p class="info-value">{lower} - {upper}</p>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_d:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown('<p class="info-label">Prediction Time</p>', unsafe_allow_html=True)
            st.markdown('<p class="info-value">0.12 seconds</p>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Top factors (from feature importance if available)
        if pred.get("features_importance"):
            st.markdown('<div class="top-factors">', unsafe_allow_html=True)
            st.markdown(
                "<p><strong>Top Factors Affecting Price:</strong></p>",
                unsafe_allow_html=True,
            )
            importances = sorted(
                pred["features_importance"].items(), key=lambda x: x[1], reverse=True
            )[:5]
            items = "".join(f"<li>{k}: {v:.3f}</li>" for k, v in importances)
            st.markdown(f"<ul>{items}</ul>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        # Display placeholder message
        st.markdown(
            """
        <div style="display: flex; height: 300px; align-items: center; justify-content: center; color: #6b7280; text-align: center;">
            Fill out the form and click "Predict Price" to see the estimated house price.
        </div>
        """,
            unsafe_allow_html=True,
        )

    # end right column

# Fetch version, hostname, and IP address
version = os.getenv("APP_VERSION", "1.0.0")  # Default version if not set in environment
hostname = socket.gethostname()
ip_address = socket.gethostbyname(hostname)

# Add footer
st.markdown("<hr>", unsafe_allow_html=True)  # Add a horizontal line for separation
st.markdown(
    f"""
    <div style="text-align: center; color: gray; margin-top: 20px;">
        <p><strong>Built for MLOps Bootcamp</strong></p>
        <p>by <a href="https://www.schoolofdevops.com" target="_blank">School of Devops</a></p>
        <p><strong>Version:</strong> {version}</p>
        <p><strong>Hostname:</strong> {hostname}</p>
        <p><strong>IP Address:</strong> {ip_address}</p>
    </div>
    """,
    unsafe_allow_html=True,
)
