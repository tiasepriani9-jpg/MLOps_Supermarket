from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open("model_supermarket.pkl", "rb") as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    unit_price = float(request.form['unit_price'])
    quantity = int(request.form['quantity'])
    cogs = float(request.form['cogs'])
    gross_income = float(request.form['gross_income'])
    rating = float(request.form['rating'])

    data = np.array([[
        unit_price,
        quantity,
        cogs,
        gross_income,
        rating
    ]])

    hasil = model.predict(data)[0]

    return render_template(
        'index.html',
        prediction_text=f"Prediksi Total Penjualan: {hasil:.2f}"
    )

if __name__ == '__main__':
    app.run(debug=True)
    