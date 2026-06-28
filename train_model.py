import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# LOAD DATA
df = pd.read_csv("Supermarket Sales Cleaned (1).csv")

# Feature dan Target
X = df[['Unit price',
        'Quantity',
        'cogs',
        'gross income',
        'Rating']]

y = df['Total']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# TRAIN MODEL
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluasi
pred = model.predict(X_test)
mse = mean_squared_error(y_test, pred)

print("MSE :", mse)

# SIMPAN MODEL
with open("model_supermarket.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model berhasil disimpan!")

# Cek ukuran file
print("Ukuran file:", os.path.getsize("model_supermarket.pkl"), "bytes")