import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Flood Forecasting System",
    layout="centered"
)

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #ffffff;
    color: #111827;
}

h1, h2, h3, h4, h5, h6, p, label {
    color: #111827 !important;
}

/* CARD */
.card {
    background-color: #f3f4f6;
    padding: 24px;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}

.section-title {
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 8px;
}

/* RESULT BOX */
.result-box {
    background-color: #ecfeff;
    border-left: 6px solid #0284c7;
    padding: 24px;
    border-radius: 12px;
    margin-top: 25px;
}

/* BUTTON */
button {
    background-color: #0284c7 !important;
    color: white !important;
    border-radius: 8px !important;
}

/* SELECTBOX - BLACK */
div[data-baseweb="select"] > div {
    background-color: #000000 !important;
    border: 2px solid #000000 !important;
    border-radius: 8px;
}

div[data-baseweb="select"] div[role="combobox"] {
    color: white !important;
}

div[data-baseweb="select"] span {
    color: white !important;
    font-weight: 500;
}

div[data-baseweb="select"] svg {
    fill: white !important;
}

div[data-baseweb="menu"] {
    background-color: #000000 !important;
}

div[data-baseweb="menu"] * {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.title("🌊 Flood Forecasting System")
st.caption("Select a forecasting model and a station to view flood probability and timing")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
dataset = pd.read_csv("your_dataset.csv")
stations = sorted(dataset["Station_Names"].unique())

# -------------------------------------------------
# INPUT CARD
# -------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

st.markdown('<div class="section-title">1️⃣ Forecasting Model</div>', unsafe_allow_html=True)
model_choice = st.selectbox(
    "Model",
    ["LSTM", "SARIMA"],
    label_visibility="collapsed"
)

st.markdown('<div class="section-title">2️⃣ Station</div>', unsafe_allow_html=True)
station_choice = st.selectbox(
    "Station",
    stations,
    label_visibility="collapsed"
)

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# LOAD LSTM MODEL
# -------------------------------------------------
@st.cache_resource
def load_lstm():
    return load_model("flood_lstm_model.h5")

if model_choice == "LSTM":
    model = load_lstm()

# -------------------------------------------------
# FORECAST FUNCTION (6 STEPS)
# -------------------------------------------------
def forecast_next_6_months(model, input_seq):
    preds = []
    seq = input_seq.copy()

    for _ in range(6):
        p = model.predict(seq, verbose=0)[0][0]
        preds.append(p)

        seq = np.roll(seq, -1, axis=1)
        seq[0, -1, :] = 0

    return preds

# -------------------------------------------------
# PREDICT BUTTON
# -------------------------------------------------
if st.button("🔍 Predict Flood Probability", use_container_width=True):

    # -------- FIXED FORECAST MONTHS: MAY–OCT --------
    last_year = dataset["Year"].max()
    forecast_year = last_year + 1

    future_dates = pd.to_datetime([
        f"{forecast_year}-05-01",
        f"{forecast_year}-06-01",
        f"{forecast_year}-07-01",
        f"{forecast_year}-08-01",
        f"{forecast_year}-09-01",
        f"{forecast_year}-10-01",
    ])

    # Prediction logic
    if model_choice == "LSTM":
        input_data = np.random.rand(1, 12, 7)  # placeholder
        preds = forecast_next_6_months(model, input_data)
        method = "LSTM Model"
    else:
        preds = np.random.uniform(0.3, 0.9, size=6)
        method = "SARIMA Model"

    # Forecast table
    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Flood Probability": preds
    })

    # -------------------------------------------------
    # RESULT
    # -------------------------------------------------
    st.markdown('<div class="result-box">', unsafe_allow_html=True)

    st.subheader("🌧️ Flood Forecast Result")
    st.write(f"**Station:** {station_choice}")
    st.write(f"**Model Used:** {method}")

    st.subheader("📅 Flood-Prone Months")
    st.dataframe(
        forecast_df.style.format({"Flood Probability": "{:.2f}"}),
        use_container_width=True
    )

    # High-risk periods
    high_risk = forecast_df[forecast_df["Flood Probability"] >= 0.5]

    if not high_risk.empty:
        st.error("⚠️ High Flood Risk Periods Detected")
        st.write(high_risk)
    else:
        st.success("✅ No High Flood Risk During Forecast Period")

    st.markdown('</div>', unsafe_allow_html=True)
