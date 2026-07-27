# 📊 Dashboard Forecasting KB — Kecamatan Anggana

Aplikasi web interaktif untuk analisis tren, prediksi jumlah peserta Keluarga Berencana (KB), dan analisis pergeseran preferensi jenis kontrasepsi di Kecamatan Anggana menggunakan metode **Time Series** dengan algoritma **Prophet**.

🔗 **Live App:** [https://ta-kb-anggana.streamlit.app](https://ta-kb-anggana.streamlit.app)

---

## 📌 Tentang Proyek

Aplikasi ini merupakan bagian dari Tugas Akhir dengan judul:

> **"Analisis Tren dan Prediksi Jumlah Peserta KB serta Pergeseran Preferensi Jenis Kontrasepsi di Kecamatan Anggana dengan Metode Time Series"**

**Penulis:** Muhammad Rizqi Ramadani
**NIM:** 236151015
**Program Studi:** D3 Teknik Informatika
**Jurusan:** Teknologi Informasi
**Institusi:** Politeknik Negeri Samarinda
**Tahun:** 2026

---

## 🎯 Fitur Utama

| Fitur | Deskripsi |
|---|---|
| 🎯 **Business Understanding** | Latar belakang, rumusan masalah, tujuan, dan alur penelitian |
| 📂 **Data Acquisition** | Tampilan data mentah SIGA BKKBN dan statistik deskriptif |
| 🛠️ **Data Preparation** | Preprocessing data, deteksi outlier (IQR), pembagian data latih/uji |
| 🤖 **Modeling Prophet** | Konfigurasi parameter dan dekomposisi komponen tren & musiman |
| 📈 **Forecasting** | Prediksi jumlah peserta KB periode Januari–Desember 2025 |
| ✅ **Evaluasi Model** | Perhitungan MAE, RMSE, MAPE dengan validasi LOOCV |
| 📉 **Analisis Tren** | Visualisasi tren jangka panjang dan pola musiman tahunan |
| 💊 **Preferensi Kontrasepsi** | Analisis pergeseran preferensi 7 jenis kontrasepsi Modern |

---

## 🗂️ Sumber Data

Data yang digunakan merupakan data sekunder berupa laporan bulanan peserta KB Modern yang bersumber dari **Sistem Informasi Keluarga (SIGA) BKKBN**, diperoleh melalui **Balai Penyuluhan KB Kecamatan Anggana**, periode **Januari 2022 – Desember 2024** (36 data poin).

---

## 🧮 Metodologi

1. Pengumpulan Data
2. Data Cleaning & Preprocessing
3. Deteksi Outlier (Interquartile Range / IQR)
4. Pembagian Data (Leave-One-Out Cross Validation)
5. Pemodelan menggunakan algoritma **Prophet**
6. Forecasting 12 bulan ke depan
7. Evaluasi Model (MAE, RMSE, MAPE)
8. Visualisasi & Analisis Pergeseran Preferensi Kontrasepsi

---

## 🛠️ Teknologi yang Digunakan

- **Python 3.11**
- **Streamlit** — Framework aplikasi web interaktif
- **Prophet** — Algoritma peramalan deret waktu
- **Pandas & NumPy** — Pengolahan data
- **Scikit-learn** — Perhitungan metrik evaluasi
- **Plotly** — Visualisasi data interaktif

---

## 🚀 Cara Menjalankan Secara Lokal

1. Clone repository ini
```bash
git clone https://github.com/username-kamu/ta-kb-anggana.git
cd ta-kb-anggana
```

2. Buat virtual environment (opsional tapi disarankan)
```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Jalankan aplikasi
```bash
streamlit run app.py
```

5. Buka browser di `http://localhost:8501`

---

## 📁 Struktur Repository

```
ta-kb-anggana/
├── app.py                # Kode utama aplikasi Streamlit
├── requirements.txt       # Daftar dependency Python
├── packages.txt           # Dependency sistem (build-essential untuk Prophet)
└── README.md              # Dokumentasi proyek
```

---

## 📖 Cara Menggunakan Aplikasi

1. Buka aplikasi melalui link di atas
2. Upload file Excel (`.xlsx`) dengan kolom minimal `ds` (tanggal) dan `total_peserta` (jumlah peserta KB)
3. Pilih menu tahapan analisis di sidebar kiri
4. Jelajahi setiap tahap: mulai dari pemahaman data hingga hasil prediksi dan analisis pergeseran preferensi kontrasepsi

---

## 📊 Hasil Evaluasi Model

| Metrik | Nilai | Kategori |
|---|---|---|
| MAE | 102 peserta | — |
| RMSE | 143 peserta | — |
| MAPE | 3.56% | **Sangat Akurat** (< 10%) |

---

## 🙏 Ucapan Terima Kasih

- **Dosen Pembimbing I:** Asrina Astagani, ST., MT.
- **Dosen Pembimbing II:** Fransisca Angelia Sebayang, S.Kom., M.Kom.
- **Balai Penyuluhan KB Kecamatan Anggana** atas data dan informasi yang diberikan

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan akademik Tugas Akhir Program Studi D3 Teknik Informatika, Politeknik Negeri Samarinda.
