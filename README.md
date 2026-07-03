#  Taxi Fare Prediction with ElasticNet Regression

Proyek regresi untuk **memprediksi tarif taksi** (*taxi fare*) menggunakan algoritma **ElasticNet** (kombinasi regularisasi L1 & L2), dilengkapi feature engineering dari data lokasi & waktu, serta hyperparameter tuning dengan `RandomizedSearchCV`. Dibangun dengan **scikit-learn** dan dibantu library [`jcopml`](https://pypi.org/project/jcopml/) untuk mempercepat proses preprocessing, tuning, dan evaluasi.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit--learn](https://img.shields.io/badge/scikit--learn-MachineLearning-orange)
![Status](https://img.shields.io/badge/status-active-success)
![License](https://img.shields.io/badge/license-MIT-green)

---

##  Deskripsi

Proyek ini membangun model **regresi linear ter-regularisasi (ElasticNet)** untuk memprediksi `fare_amount` (tarif taksi) berdasarkan data perjalanan seperti waktu penjemputan, jumlah penumpang, dan jarak tempuh. Notebook ini juga membahas proses **debugging model** — mulai dari analisis residual, identifikasi data dengan error prediksi terbesar, hingga menemukan anomali pada data mentah (jarak nol & tarif negatif).

Proyek ini cocok sebagai referensi belajar:
- **Feature engineering** dari data datetime & koordinat geografis
- Penggunaan **ElasticNet** dengan **polynomial features** untuk menangkap hubungan non-linear
- **Hyperparameter tuning** otomatis dengan `RandomizedSearchCV`
- Analisis **feature importance** dan **residual plot** untuk evaluasi model
- Proses **error analysis** — menelusuri data mana yang menyebabkan prediksi meleset paling jauh

---

##  Struktur Proyek

```
.
├── Elasticnet.ipynb   # Notebook utama
├── data/
│   └── taxi_fare.csv           # Dataset perjalanan taksi
├── model/                      # Output model tersimpan (opsional)
└── README.md
```

---

##  Instalasi

1. Clone repository ini:
```bash
git clone https://github.com/username/nama-repo.git
cd nama-repo
```

2. Buat virtual environment (opsional tapi disarankan):
```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

3. Install dependencies:
```bash
pip install numpy pandas scikit-learn matplotlib jupyter
pip install jcopml
```

---

##  Alur Data (Data Pipeline)

### 1. Feature Engineering
| Fitur Baru | Sumber | Keterangan |
|---|---|---|
| `year`, `month`, `day`, `hour` | `pickup_datetime` | Diekstrak dari timestamp penjemputan |
| `distance` | koordinat pickup & dropoff | Dihitung dari selisih absolut longitude + latitude (Manhattan distance sederhana) |

Kolom `pickup_datetime` serta koordinat mentah (`pickup_longitude`, `dropoff_longitude`, `pickup_latitude`, `dropoff_latitude`) dihapus setelah fitur baru terbentuk, karena informasinya sudah terwakili oleh fitur turunan di atas.

### 2. Data Cleaning
- Baris dengan nilai `NaN` dihapus (`dropna`)
- Ditemukan anomali: beberapa baris memiliki `fare_amount` **negatif** — indikasi data kotor yang perlu ditangani lebih lanjut

### 3. Train-Test Split
Data dibagi 80:20 (`test_size=0.2`, `random_state=42`) menjadi `X_train`, `X_test`, `y_train`, `y_test`.

---

##  Pipeline & Preprocessing

```python
preprocessor = ColumnTransformer([
    ("numeric", num_pipe(poly=2, transform="yeo-johnson"), ["passenger_count", "year", "distance"]),
    ("categoric", cat_pipe(encoder="onehot"), ["month", "day", "hour"])
])

pipeline = Pipeline([
    ("prep", preprocessor),
    ("algo", ElasticNet(random_state=42))
])
```

| Komponen | Detail |
|---|---|
| **Fitur numerik** | `passenger_count`, `year`, `distance` → ditransformasi dengan **Yeo-Johnson** (menstabilkan skewness) + **polynomial degree 2** (menangkap hubungan non-linear & interaksi antar fitur) |
| **Fitur kategorik** | `month`, `day`, `hour` → di-encode dengan **One-Hot Encoding** |
| **Model** | `ElasticNet` — regresi linear dengan kombinasi regularisasi L1 (Lasso) & L2 (Ridge) |

---

##  Training & Tuning

Model dituning menggunakan `RandomizedSearchCV` dengan parameter grid `enet_poly_params` bawaan `jcopml`:

```python
model = RandomizedSearchCV(
    pipeline,
    rsp.enet_poly_params,
    cv=3,
    n_iter=100,
    n_jobs=-1,
    verbose=1,
    random_state=42
)
model.fit(X_train, y_train)
```

| Parameter | Nilai |
|---|---|
| **Cross-validation** | 3-fold |
| **Jumlah iterasi random search** | 100 |
| **Parallel jobs** | -1 (menggunakan semua core CPU) |

Setelah training, tiga skor dievaluasi: skor pada data train, skor CV terbaik (`best_score_`), dan skor pada data test — untuk memastikan model tidak overfit/underfit.

---

##  Evaluasi Model

### Feature Importance
```python
mean_score_decrease(X_train, y_train, model, plot=True, topk=10)
```
Menampilkan 10 fitur paling berpengaruh terhadap prediksi tarif, berdasarkan penurunan skor model saat fitur tersebut diacak (*permutation importance*).

### Residual Plot
```python
plot_residual(X_train, y_train, X_test, y_test, model)
```
Digunakan untuk melihat sebaran kesalahan prediksi (residual) — pola yang tidak acak pada plot ini mengindikasikan model masih belum menangkap seluruh struktur data (*underfitting* pada area tertentu).

###  Error Analysis
Notebook ini melangkah lebih jauh dengan **menelusuri data spesifik** yang menyebabkan error terbesar:

```python
df_analysis['fare'] = y_train
df_analysis['error'] = np.abs(pred - y_train)
df_analysis.sort_values("error", ascending=False).head(10)
```

**Temuan menarik:**
- Sebagian besar error terbesar berasal dari data dengan **`distance = 0`** — kemungkinan kesalahan input data (pickup dan dropoff di titik yang sama, namun tarif tetap tercatat)
- Ditemukan pula **`fare_amount` bernilai negatif** pada beberapa baris — jelas merupakan anomali karena tarif taksi tidak mungkin negatif

>  **Insight kunci:** Akurasi model regresi sangat bergantung pada kualitas data. Sebelum menyalahkan model karena residual yang besar, penting untuk melakukan *error analysis* guna memeriksa apakah penyebabnya adalah data kotor/anomali — bukan semata-mata kelemahan algoritma.

---

##  Tech Stack

- **scikit-learn** — Model ElasticNet, Pipeline, ColumnTransformer, RandomizedSearchCV
- **jcopml** — Preprocessing pipeline (`num_pipe`, `cat_pipe`), tuning params, plotting (residual, feature importance)
- **pandas** & **NumPy** — Manipulasi data & feature engineering
- **Matplotlib** — Visualisasi hasil evaluasi

---

##  Lisensi

Proyek ini menggunakan lisensi **MIT** — bebas digunakan, dimodifikasi, dan didistribusikan ulang untuk keperluan pembelajaran maupun pengembangan lebih lanjut.

---

##  Kontribusi

Kontribusi, saran, dan diskusi sangat terbuka! Silakan buat *issue* atau *pull request* — terutama untuk menambahkan tahap pembersihan data anomali (`distance = 0`, `fare_amount < 0`) sebagai bagian dari pipeline preprocessing agar performa model semakin baik.

---


