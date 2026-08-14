import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras

st.set_page_config(
    page_title="UrbanNest | Housing Price Predictor",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 85% 8%, rgba(80, 120, 255, 0.10), transparent 28%),
            radial-gradient(circle at 10% 30%, rgba(0, 190, 160, 0.07), transparent 25%),
            #f6f8fb;
    }

    .hero {
        padding: 34px 38px 28px 38px;
        border-radius: 24px;
        background: linear-gradient(135deg, #111827 0%, #1f2937 55%, #243b53 100%);
        color: white;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.16);
        margin-bottom: 24px;
    }

    .hero h1 {
        font-family: 'Manrope', sans-serif;
        font-size: 42px;
        line-height: 1.05;
        margin: 0 0 10px 0;
        letter-spacing: -1.5px;
    }

    .hero p {
        color: #cbd5e1;
        font-size: 16px;
        margin: 0;
        max-width: 780px;
    }

    .section-title {
        font-family: 'Manrope', sans-serif;
        font-size: 20px;
        font-weight: 800;
        color: #111827;
        margin: 8px 0 14px 0;
    }

    .result-card {
        background: white;
        border-radius: 22px;
        padding: 28px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.07);
        text-align: center;
    }

    .result-label {
        color: #64748b;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .result-price {
        font-family: 'Manrope', sans-serif;
        font-size: 44px;
        font-weight: 800;
        color: #111827;
        margin: 8px 0 4px 0;
    }

    .result-note {
        color: #64748b;
        font-size: 13px;
    }

    .metric-card {
        background: white;
        border-radius: 18px;
        padding: 18px 20px;
        border: 1px solid #e5e7eb;
        height: 100%;
    }

    .metric-label {
        color: #64748b;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .7px;
    }

    .metric-value {
        font-family: 'Manrope', sans-serif;
        font-size: 22px;
        font-weight: 800;
        color: #111827;
        margin-top: 5px;
    }

    [data-testid="stSidebar"] {
        background: #111827;
    }

    [data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 13px;
        height: 50px;
        font-weight: 700;
        border: 0;
    }

    .disclaimer {
        margin-top: 20px;
        padding: 14px 16px;
        border-radius: 13px;
        background: #eef2ff;
        color: #475569;
        font-size: 12px;
        border: 1px solid #e0e7ff;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load artifacts
# -----------------------------
MODEL_PATH = os.path.join("artifacts", "housing_price_model.keras")
PREPROCESSOR_PATH = os.path.join("artifacts", "preprocessor.joblib")

@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        return None, None
    model = keras.models.load_model(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor

model, preprocessor = load_artifacts()

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("## 🏙️ UrbanNest")
    st.caption("AI-powered residential price estimation")
    st.divider()
    st.markdown("### About the model")
    st.write(
        "A deep neural network trained using the preprocessing and "
        "architecture from the supplied UrbanNest Realty notebook."
    )
    st.divider()
    st.caption("Prediction is an estimate, not a formal appraisal.")

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="hero">
    <h1>Know the value of your home.</h1>
    <p>
        Enter a property's characteristics and get an estimated market price
        from the UrbanNest deep-learning housing model.
    </p>
</div>
""", unsafe_allow_html=True)

if model is None or preprocessor is None:
    st.error(
        "Model artifacts are not installed yet. Run `python train_model.py` "
        "with the original cleaned_real_estate_final_price_v2.csv, then start Streamlit again."
    )
    st.stop()

# Get learned category values when available
cat_encoder = preprocessor.named_transformers_.get("cat")
categories = {}
if cat_encoder is not None and hasattr(cat_encoder, "categories_"):
    for col, vals in zip(["status", "city", "state", "zip3"], cat_encoder.categories_):
        categories[col] = [str(v) for v in vals if pd.notna(v)]

# -----------------------------
# Main input form
# -----------------------------
st.markdown('<div class="section-title">Property details</div>', unsafe_allow_html=True)

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        bed = st.number_input("Bedrooms", min_value=0, max_value=20, value=3, step=1)
        bath = st.number_input("Bathrooms", min_value=0.0, max_value=20.0, value=2.0, step=0.5)
        house_size = st.number_input(
            "House size (sq ft)",
            min_value=100.0,
            max_value=50000.0,
            value=1800.0,
            step=50.0,
        )

    with col2:
        acre_lot = st.number_input(
            "Lot size (acres)",
            min_value=0.0,
            max_value=1000.0,
            value=0.20,
            step=0.01,
            format="%.2f",
        )

        if "status" in categories and categories["status"]:
            status = st.selectbox("Property status", categories["status"])
        else:
            status = st.text_input("Property status", value="for_sale")

        has_prior_sale = st.selectbox(
            "Has prior sale?",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )

    with col3:
        if "city" in categories and categories["city"]:
            city = st.selectbox("City", categories["city"])
        else:
            city = st.text_input("City", value="")

        if "state" in categories and categories["state"]:
            state = st.selectbox("State", categories["state"])
        else:
            state = st.text_input("State", value="")

        if "zip3" in categories and categories["zip3"]:
            zip3 = st.selectbox("ZIP3", categories["zip3"])
        else:
            zip3 = st.text_input("ZIP3", value="")

    st.markdown("")
    submitted = st.form_submit_button("✨ Estimate property value")

if submitted:
    input_df = pd.DataFrame([{
        "status": status,
        "city": city,
        "state": state,
        "zip3": zip3,
        "bed": bed,
        "bath": bath,
        "acre_lot": acre_lot,
        "house_size": house_size,
        "has_prior_sale": has_prior_sale,
    }])

    try:
        processed = preprocessor.transform(input_df)
        prediction = float(model.predict(processed, verbose=0).ravel()[0])
        prediction = max(0.0, prediction)

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Estimated property value</div>
                <div class="result-price">${prediction:,.0f}</div>
                <div class="result-note">USD • Deep neural network estimate</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("")
        m1, m2, m3, m4 = st.columns(4)

        metrics = [
            ("Bedrooms", f"{bed:g}"),
            ("Bathrooms", f"{bath:g}"),
            ("House size", f"{house_size:,.0f} sq ft"),
            ("Lot size", f"{acre_lot:,.2f} acres"),
        ]

        for col, (label, value) in zip([m1, m2, m3, m4], metrics):
            with col:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown(
            """
            <div class="disclaimer">
                <b>Note:</b> This estimate is generated by the trained model and
                should be treated as an indicative prediction rather than a
                professional valuation or appraisal.
            </div>
            """,
            unsafe_allow_html=True,
        )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
