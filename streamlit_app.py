import streamlit as st
import pandas as pd
import pickle

# ==================================
# LOAD MODEL
# ==================================

model = pickle.load(open("model_supermarket.pkl", "rb"))

# ==================================
# JUDUL
# ==================================

st.set_page_config(
    page_title="Prediksi Penjualan Supermarket",
    page_icon="🛒",
    layout="centered"
)

st.title("🛒 Prediksi Total Penjualan Supermarket")
st.write("Masukkan data produk untuk memprediksi total penjualan.")

# ==================================
# INPUT USER
# ==================================

unit_price = st.number_input(
    "Unit Price",
    min_value=0.0,
    value=10.0
)

quantity = st.number_input(
    "Quantity",
    min_value=1,
    value=1
)

cogs = st.number_input(
    "COGS",
    min_value=0.0,
    value=10.0
)

gross_income = st.number_input(
    "Gross Income",
    min_value=0.0,
    value=1.0
)

rating = st.number_input(
    "Rating",
    min_value=0.0,
    max_value=10.0,
    value=5.0
)

# ==================================
# PREDIKSI
# ==================================

if st.button("Prediksi"):

    data = pd.DataFrame({
        'Unit price': [unit_price],
        'Quantity': [quantity],
        'cogs': [cogs],
        'gross income': [gross_income],
        'Rating': [rating]
    })

    hasil = model.predict(data)

    st.success(
        f"Prediksi Total Penjualan: {hasil[0]:,.2f}"
    )