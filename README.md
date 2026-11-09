# 📊 Dashboard Forecasting KB — Kecamatan Anggana

Aplikasi web interaktif untuk menganalisis tren, memprediksi jumlah peserta Keluarga Berencana (KB), dan menganalisis pergeseran preferensi jenis kontrasepsi di Kecamatan Anggana, Kutai Kartanegara, menggunakan algoritma **Prophet**.

🔗 **Live App:** https://ta-kb-anggana.streamlit.app
> Jika muncul tombol *"Yes, get this app back up!"*, klik dan tunggu sekitar satu menit. Aplikasi gratis di Streamlit Cloud akan "tidur" jika lama tidak dibuka.

🧪 **Mau mencoba?** Unduh [`data_contoh.xlsx`](data_contoh.xlsx) di repository ini, lalu unggah ke aplikasi.

<!-- Ganti dengan screenshot asli: simpan di folder images/ -->
<!-- ![Forecasting](images/forecasting.png) -->
<!-- ![Preferensi Kontrasepsi](images/preferensi.png) -->

---

## 📌 Tentang Proyek

Proyek ini adalah bagian dari Tugas Akhir berjudul:

**"Analisis Tren dan Prediksi Jumlah Peserta KB serta Pergeseran Preferensi Jenis Kontrasepsi di Kecamatan Anggana Menggunakan Metode Prophet"**

| | |
|---|---|
| **Penulis** | Muhammad Rizqi Ramadani (NIM 236151015) |
| **Program Studi** | D3 Teknik Informatika, Jurusan Teknologi Informasi |
| **Institusi** | Politeknik Negeri Samarinda, 2026 |
| **Pembimbing** | Asrina Astagani, ST., MT. dan Fransisca Angelia Sebayang, S.Kom., M.Kom. |

**Masalah yang diselesaikan:** Balai Penyuluhan KB membutuhkan gambaran jumlah peserta dan metode kontrasepsi yang akan diminati agar perencanaan program dan alokasi alat kontrasepsi lebih tepat.

---

## 📈 Hasil Utama

**Akurasi model (validasi LOOCV, 36 data bulanan)**

| Metrik | Nilai | Kategori |
|---|---|---|
| MAE | 101,61 peserta | n/a |
| RMSE | 143,37 peserta | n/a |
| MAPE | 3,56% | Sangat Akurat (< 10%) |

**Peramalan:** Jumlah peserta KB Modern diprediksi **meningkat sepanjang 2025**, dengan puncak **3.638 peserta pada Desember 2025**.

**Pergeseran preferensi kontrasepsi (rata-rata peserta/bulan, 2022 vs 2024)**

| Metode | 2022 | 2024 | Perubahan |
|---|---|---|---|
| Suntik | 1.513,67 | 1.863,17 | +23,10% |
| Pil | 892,50 | 703,50 | -21,18% |
| Kondom | 71,50 | 83,17 | +16,32% |
| Implant | 18,75 | 33,17 | **+76,89%** |
| IUD | 101,67 | 140,50 | +38,18% |
| MOP | 86,00 | 123,92 | +44,09% |
| MOW | 5,00 | 0,00 | -100% |

Suntik tetap menjadi metode paling banyak digunakan. Kenaikan persentase Implant tergolong besar karena berangkat dari jumlah awal yang kecil.

---

## 🎯 Fitur Utama

| Fitur | Deskripsi |
|---|---|
| 🎯 Business Understanding | Latar belakang, rumusan masalah, tujuan, dan alur penelitian |
| 📂 Data Acquisition | Tampilan data mentah SIGA BKKBN dan statistik deskriptif |
| 🛠️ Data Preparation | Preprocessing, deteksi outlier (IQR), pembagian data |
| 🤖 Modeling Prophet | Konfigurasi parameter dan dekomposisi tren dan musiman |
| 📈 Forecasting | Prediksi jumlah peserta KB periode Januari–Desember 2025 |
| ✅ Evaluasi Model | MAE, RMSE, MAPE dengan validasi LOOCV |
| 📉 Analisis Tren | Tren jangka panjang dan pola musiman tahunan |
| 💊 Preferensi Kontrasepsi | Pergeseran preferensi 7 jenis kontrasepsi (Suntik, Pil, Kondom, Implant, IUD, MOW, MOP) |

---

## 🗂️ Sumber Data

Data sekunder berupa laporan bulanan peserta KB Modern dari **Sistem Informasi Keluarga (SIGA) BKKBN**, diperoleh melalui Balai Penyuluhan KB Kecamatan Anggana, periode **Januari 2022 – Desember 2024 (36 data poin)**.

---

## 🧮 Metodologi

1. Pengumpulan data
2. Data cleaning dan preprocessing
3. Deteksi outlier (Interquartile Range / IQR)
4. Validasi dengan Leave-One-Out Cross Validation (LOOCV)
5. Pemodelan dengan Prophet
6. Forecasting 12 bulan ke depan
7. Evaluasi model (MAE, RMSE, MAPE)
8. Visualisasi dan analisis pergeseran preferensi kontrasepsi

---

## 🛠️ Teknologi

Python 3.11 · Streamlit · Prophet · Pandas · NumPy · Scikit-learn · Plotly

---

## 🚀 Menjalankan Secara Lokal

```bash
git clone https://github.com/muhammadrizqiramadani010-star/ta-kb-anggana.git
cd ta-kb-anggana

# (opsional) virtual environment
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows

pip install -r requirements.txt
streamlit run app.py
```

Buka http://localhost:8501

## 📖 Cara Menggunakan Aplikasi

1. Buka live app (atau jalankan lokal).
2. Unggah file Excel (`.xlsx`) dengan kolom minimal `ds` (tanggal) dan `total_peserta`. Anda bisa memakai `data_contoh.xlsx`.
3. Pilih tahapan analisis di sidebar kiri.
4. Jelajahi mulai dari pemahaman data hingga prediksi dan analisis preferensi.

## 📁 Struktur Repository

```
ta-kb-anggana/
├── app.py                # Aplikasi Streamlit
├── requirements.txt      # Dependency Python
├── packages.txt          # Dependency sistem (build-essential untuk Prophet)
├── data_contoh.xlsx      # Contoh data untuk mencoba aplikasi
├── images/               # Screenshot aplikasi
└── README.md
```

---

## ⚠️ Keterbatasan

- Data hanya **36 bulan** dari **satu kecamatan**, sehingga hasil peramalan bersifat perkiraan dan bukan kepastian.
- Model hanya memakai pola historis; faktor eksternal (kebijakan, ketersediaan alat, kampanye) tidak dimodelkan.
- Validasi LOOCV dipilih karena jumlah data terbatas; untuk data yang lebih panjang, validasi berurutan (rolling/time series cross validation) lebih disarankan.

## 🙏 Ucapan Terima Kasih

Kepada kedua dosen pembimbing dan Balai Penyuluhan KB Kecamatan Anggana atas data dan informasi yang diberikan.

## 📄 Lisensi

Dibuat untuk keperluan akademik Tugas Akhir Program Studi D3 Teknik Informatika, Politeknik Negeri Samarinda.
