import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# =====================================
# KONFIGURASI HALAMAN
# =====================================

st.set_page_config(
    page_title="Prediksi Penjualan Supermarket",
    page_icon="🛒",
    layout="wide"
)

# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>

.stApp{
    background-color:#F4F8FB;
}

h1{
    color:#1565C0;
    text-align:center;
}

[data-testid="stSidebar"]{
    background-color:#1565C0;
}

[data-testid="stSidebar"] *{
    color:white;
}

div.stButton > button{
    background-color:#1565C0;
    color:white;
    border-radius:10px;
    width:100%;
    height:50px;
    font-size:18px;
    font-weight:bold;
}

.metric-card{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 0px 10px rgba(0,0,0,0.1);
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# LOAD MODEL
# =====================================

model_path = Path("model_supermarket.pkl")

with open(model_path, "rb") as file:
    model = pickle.load(file)

# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("🛒 MLOps Supermarket")

st.sidebar.markdown("""
### Tentang Aplikasi

Aplikasi ini digunakan untuk:

- Prediksi Total Penjualan
- Machine Learning
- Random Forest / XGBoost
- Deployment Streamlit

**Dibuat oleh:**
Tya Sepriani
""")

# =====================================
# HEADER
# =====================================

st.title("🛒 Prediksi Total Penjualan Supermarket")

st.markdown("---")

# =====================================
# METRIC
# =====================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Model",
        value="Random Forest"
    )

with col2:
    st.metric(
        label="Dataset",
        value="1000 Data"
    )

with col3:
    st.metric(
        label="Status",
        value="Aktif"
    )

st.markdown("---")

# =====================================
# INPUT USER
# =====================================

st.subheader("📋 Input Data")

col1, col2 = st.columns(2)

with col1:

    unit_price = st.number_input(
        "Unit Price",
        min_value=0.0,
        value=50.0
    )

    quantity = st.number_input(
        "Quantity",
        min_value=1,
        value=5
    )

    cogs = st.number_input(
        "COGS",
        min_value=0.0,
        value=200.0
    )

with col2:

    gross_income = st.number_input(
        "Gross Income",
        min_value=0.0,
        value=20.0
    )

    rating = st.slider(
        "Rating",
        0.0,
        10.0,
        7.0
    )

# =====================================
# PREDIKSI
# =====================================

if st.button("🔮 Prediksi Penjualan"):

    input_data = pd.DataFrame({
        'Unit price': [unit_price],
        'Quantity': [quantity],
        'cogs': [cogs],
        'gross income': [gross_income],
        'Rating': [rating]
    })

    prediction = model.predict(input_data)

    st.success(
        f"💰 Prediksi Total Penjualan: Rp {prediction[0]:,.2f}"
    )

    st.balloons()

# =====================================
# TABEL INPUT
# =====================================

st.markdown("---")

st.subheader("📊 Data Input")

preview = pd.DataFrame({
    'Unit Price':[unit_price],
    'Quantity':[quantity],
    'COGS':[cogs],
    'Gross Income':[gross_income],
    'Rating':[rating]
})

st.dataframe(
    preview,
    use_container_width=True
)

# =====================================
# FOOTER
# =====================================

st.markdown("---")

st.markdown(
"""
<center>
<h4>🛒 MLOps Supermarket Sales Prediction</h4>
<p>Dibuat menggunakan Streamlit, Machine Learning dan Python</p>
</center>
""",
unsafe_allow_html=True
)