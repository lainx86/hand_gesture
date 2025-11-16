import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

data_file = 'gesture_data.csv'
try:
    data = pd.read_csv(data_file)
except FileNotFoundError:
    print(f"Error: File '{data_file}' tidak ditemukan.")
    print("Pastikan Anda sudah menjalankan 'data_collector.py' terlebih dahulu.")
    exit()

if data.empty:
    print(f"Error: File '{data_file}' kosong.")
    print("Silakan kumpulkan data gestur terlebih dahulu.")
    exit()

print(f"Data berhasil dimuat. Total {len(data)} sampel ditemukan.")

X = data.drop('label', axis=1)
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Data dibagi: {len(X_train)} untuk training, {len(X_test)} untuk testing.")

model = RandomForestClassifier(n_estimators=100, random_state=42)

print("Memulai pelatihan model...")
model.fit(X_train, y_train)
print("Pelatihan model selesai.")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("-" * 30)
print(f"Akurasi Model: {accuracy * 100:.2f}%")
print("-" * 30)

if accuracy < 0.90:
    print("Peringatan: Akurasi di bawah 90%. Coba kumpulkan lebih banyak data gestur.")

model_file = 'gesture_model.pkl'
with open(model_file, 'wb') as f:
    pickle.dump(model, f)

print(f"Model berhasil disimpan ke '{model_file}'")
print("Anda sekarang siap untuk langkah terakhir: integrasi ke 'main.py'")
