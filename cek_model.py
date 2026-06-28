import pickle

with open("model_supermarket.pkl", "rb") as file:
    model = pickle.load(file)

print("Model berhasil dimuat")
print(type(model))