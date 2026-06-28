from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI(
    title="API Prediksi Total Penjualan Supermarket",
    description="API untuk memprediksi total penjualan berdasarkan data transaksi",
    version="1.0"
)

# ==========================
# LOAD MODEL
# ==========================
with open("model_supermarket.pkl", "rb") as file:
    model = pickle.load(file)

# ==========================
# HALAMAN UTAMA
# ==========================
@app.get("/")
def home():
    return {
        "message": "API Prediksi Total Penjualan Supermarket"
    }

# ==========================
# TEST MODEL
# ==========================
@app.get("/test")
def test():

    fitur = np.array([[74.69, 7, 522.83, 26.14, 9.1]])

    hasil = model.predict(fitur)

    return {
        "prediksi_total": float(hasil[0])
    }

# ==========================
# PREDIKSI
# ==========================
@app.get("/predict")
def predict(
    unit_price: float,
    quantity: int,
    cogs: float,
    gross_income: float,
    rating: float
):

    fitur = np.array([[
        unit_price,
        quantity,
        cogs,
        gross_income,
        rating
    ]])

    hasil = model.predict(fitur)

    return {
        "unit_price": unit_price,
        "quantity": quantity,
        "cogs": cogs,
        "gross_income": gross_income,
        "rating": rating,
        "prediksi_total": float(hasil[0])
    }