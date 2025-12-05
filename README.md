# Machine Learning & Deep Learning Pipelines for Credit‑Card and Transaction Data

## Purpose of the repository

Repositori ini dikembangkan untuk memenuhi tugas individu mata kuliah **Machine Learning & Deep Learning**. Isi repositori meliputi beberapa notebook Jupyter yang mengimplementasikan *end‑to‑end* pipeline untuk berbagai jenis masalah (klasifikasi, regresi dan clustering). Tujuan utama repositori ini adalah:

1. Menunjukkan tahapan lengkap dalam mengolah data mentah (data loading, EDA, pra‑proses, handling missing values) hingga menghasilkan model yang siap digunakan.  
2. Membandingkan performa berbagai algoritma *conventional machine learning* dengan model deep learning pada beberapa kasus.  
3. Menyajikan contoh pengukuran kinerja model menggunakan metrik evaluasi yang tepat, misalnya akurasi, precision, recall, F1 score:contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1} untuk klasifikasi dan MSE/RMSE/MAE/R² untuk regresi.

## Overview of the project

Dataset yang digunakan di repositori ini mencakup:

* **Fraud Detection (transaction dataset)** – Dataset transaksi daring dengan 590.540 baris dan 393 fitur serta label `isFraud`. Dalam versi *machine learning* digunakan model LightGBM sebagai baseline untuk mendeteksi penipuan. Selain itu disediakan versi *deep learning* yang membangun jaringan saraf feedforward.  
* **Release Year Prediction (regression dataset)** – Dataset audio di mana nilai pertama pada tiap baris merupakan tahun rilis, sedangkan kolom lainnya adalah fitur numerik. Notebook ini membandingkan model konvensional (Linear Regression, Ridge Regression, RandomForest) dan model deep learning untuk memprediksi tahun rilis.  
* **Credit Card Behaviour (clustering dataset)** – Dataset penggunaan kartu kredit yang awalnya ditujukan untuk clustering. Untuk tujuan pembelajaran, kami membuat label biner `HighBalance` (saldo ≥ median) sehingga dataset dapat digunakan untuk tugas klasifikasi. Tersedia versi conventional ML (Logistic Regression, RandomForest, XGBoost) dan versi deep learning.

Masing‑masing notebook memuat alur pengerjaan yang serupa: memuat data, menyiapkan fitur/target, menangani nilai hilang, membagi data train‑test, melatih model, mengevaluasi hasil, dan memberikan interpretasi singkat. Proses evaluasi menggunakan metrik yang relevan: akurasi dan F1 score untuk klasifikasi:contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}; MAE, MSE, RMSE, dan R² untuk regresi.

## Models and metric results

### Fraud detection (LightGBM & Deep Learning)

* **Notebook:** `midterm_solution.ipynb` / `transaction_deep_learning.ipynb`  
* **Model ML:** menggunakan **LightGBM** sebagai algoritma utama untuk prediksi `isFraud`. LightGBM dipilih karena mampu menangani data dengan banyak fitur dan missing values. Pada notebook disertakan pipeline pra‑proses (imputasi, encoding, scaling), oversampling dengan SMOTE, pelatihan model, hyperparameter tuning, serta pembuatan file submission.  
* **Model DL:** menggunakan jaringan saraf feedforward tiga lapis dengan fungsi aktivasi ReLU dan sigmoid. Model dilatih menggunakan `binary_crossentropy` dengan class weighting untuk menangani ketidakseimbangan kelas.  
* **Metode evaluasi:** metrik akurasi, precision, recall, F1 score, dan ROC‑AUC:contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5}. Hasil evaluasi bersifat contoh; nilai aktual akan muncul setelah Anda menjalankan notebook dengan dataset asli.

### Regression (Year prediction)

* **Notebook:** `regression_conventional.ipynb` / `regression_deep_learning.ipynb`  
* **Model ML:** Linear Regression, Ridge Regression, RandomForestRegressor, LightGBMRegressor (opsional).  
* **Model DL:** MLP (dense neural network) dengan beberapa hidden layer.  
* **Metode evaluasi:** Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), dan koefisien determinasi (R²). MSE dan RMSE mengukur rata‑rata kuadrat selisih antara prediksi dan target, sedangkan MAE menghitung rata‑rata selisih absolut:contentReference[oaicite:6]{index=6}:contentReference[oaicite:7]{index=7}.

### Credit card customer classification

* **Notebook:** `credit_card_classification.ipynb` / `credit_card_classification_dl.ipynb`  
* **Model ML:** Logistic Regression, RandomForestClassifier, XGBoostClassifier. Label `HighBalance` ditentukan berdasarkan median saldo.  
* **Model DL:** jaringan saraf feedforward dengan dua hidden layer.  
* **Metode evaluasi:** akurasi, precision, recall, F1 score, dan ROC‑AUC. F1 score adalah rata‑rata harmonis dari precision dan recall sehingga cocok untuk dataset tidak seimbang:contentReference[oaicite:8]{index=8}.

## Navigating the repository / notebooks

Semua notebook berada di direktori utama repositori dan dinamai sesuai tugasnya:

| Notebook | Tujuan |
|---|---|
| `midterm_solution.ipynb` | Pipeline deteksi fraud menggunakan LightGBM. Memuat data, EDA, pra‑proses, model LightGBM, tuning, evaluasi & submission. |
| `transaction_deep_learning.ipynb` | Versi deep learning dari deteksi fraud dengan MLP dan class weights. |
| `regression_conventional.ipynb` | Pipeline prediksi tahun rilis menggunakan model regresi konvensional (Linear, Ridge, RandomForest). |
| `regression_deep_learning.ipynb` | Versi deep learning untuk regresi menggunakan MLP. |
| `credit_card_classification.ipynb` | Klasifikasi pelanggan kartu kredit berdasarkan saldo menggunakan model konvensional (LogReg, RF, XGBoost). |
| `credit_card_classification_dl.ipynb` | Versi deep learning untuk klasifikasi pelanggan kartu kredit. |

Untuk menjalankan notebook:

1. Pastikan paket yang dibutuhkan (`gdown`, `pandas`, `scikit‑learn`, `xgboost`, `lightgbm`, `tensorflow`) telah terinstal.  
2. Jalankan sel pertama untuk mengunduh dataset dari Google Drive (ganti placeholder `<FILE_ID>`/`<FOLDER_ID>` dengan ID dataset Anda).  
3. Eksekusi sel selanjutnya secara berurutan. Notebook akan menghasilkan evaluasi model dan file submission (untuk tugas klasifikasi penipuan).  
4. Untuk notebook deep learning, beberapa epoch pelatihan mungkin memerlukan waktu; gunakan `EarlyStopping` untuk menghindari overfitting.

## Identification

**Nama:** Muhamad Mario Rizki  
**Kelas:** TK-46-GAB – Teknik Komputer - Telkom University  
**NIM:** *1103223063*

---

Apabila Anda memiliki pertanyaan atau memerlukan penyesuaian lebih lanjut (misalnya menggunakan metrik evaluasi lain, menambahkan analisis visual, atau menerapkan algoritma lainnya), silakan hubungi penulis.
