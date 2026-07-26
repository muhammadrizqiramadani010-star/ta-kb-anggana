import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Dashboard Forecasting KB — Kec. Anggana",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

PRIMARY      = "#1565C0"
ACCENT       = "#0D47A1"
WARN         = "#E65100"
DANGER       = "#C62828"
SURFACE      = "#FFFFFF"
BG           = "#EEF2FF"
BORDER       = "#BBDEFB"
MUTED        = "#546E7A"
TEXT_PRIMARY   = "#0D1B4B"
TEXT_SECONDARY = "#1A3A6B"
TEXT_MUTED     = "#78909C"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
[data-testid="stAppViewContainer"] { background-color: #EEF2FF; }
.main .block-container { padding-top:1.5rem; padding-bottom:2rem; max-width:1400px; background-color:#EEF2FF; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg,#0D47A1 0%,#1565C0 60%,#1976D2 100%); border-right:2px solid #0D47A1; }
section[data-testid="stSidebar"] * { color:#FFFFFF !important; }
section[data-testid="stSidebar"] p { color:#BBDEFB !important; }
[data-testid="stHeader"] { background:linear-gradient(90deg,#0D47A1,#1565C0); border-bottom:2px solid #0D47A1; }
h1,h2,h3,h4 { color:#0D1B4B !important; }
p,span,div,label { color:#1A3A6B; }
[data-testid="metric-container"] { background:#FFFFFF; border:1px solid #BBDEFB; border-top:3px solid #1565C0; border-radius:12px; padding:18px 20px; box-shadow:0 2px 8px rgba(21,101,192,0.1); transition:all 0.2s; }
[data-testid="metric-container"]:hover { border-top-color:#0D47A1; box-shadow:0 4px 16px rgba(13,71,161,0.18); }
[data-testid="metric-container"] label { color:#546E7A !important; font-size:0.78rem; font-weight:500; letter-spacing:0.05em; text-transform:uppercase; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color:#0D1B4B !important; font-size:1.6rem; font-weight:700; }
[data-testid="stDataFrame"] { border-radius:10px; overflow:hidden; border:1px solid #BBDEFB; }
.streamlit-expanderHeader { background:#E3F2FD !important; color:#0D47A1 !important; border-radius:8px; font-weight:600; border:1px solid #BBDEFB !important; }
.stExpander { border:1px solid #BBDEFB; border-radius:10px; overflow:hidden; }
.kpi-card { background:#FFFFFF; border:1px solid #BBDEFB; border-radius:14px; padding:20px 22px; margin-bottom:14px; box-shadow:0 2px 8px rgba(21,101,192,0.08); }
.kpi-card h4 { margin:0 0 6px 0; font-size:0.75rem; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:#546E7A !important; }
.kpi-card .val { font-size:2rem; font-weight:700; color:#0D1B4B !important; line-height:1.1; }
.kpi-card .sub { font-size:0.78rem; color:#78909C; margin-top:4px; }
.info-box { background:#E3F2FD; border-left:4px solid #1565C0; border-radius:0 10px 10px 0; padding:14px 18px; margin-bottom:14px; }
.info-box p { margin:0; color:#1A3A6B; font-size:0.875rem; line-height:1.6; }
.section-title { font-size:1.1rem; font-weight:700; color:#0D47A1; margin:20px 0 12px 0; padding-bottom:8px; border-bottom:2px solid #BBDEFB; }
.badge { display:inline-block; padding:3px 10px; border-radius:99px; font-size:0.72rem; font-weight:600; letter-spacing:0.04em; text-transform:uppercase; }
.badge-blue  { background:rgba(21,101,192,0.12); color:#0D47A1; }
.badge-green { background:rgba(22,163,74,0.12);  color:#15803D; }
.badge-amber { background:rgba(230,81,0,0.12);   color:#BF360C; }
.badge-red   { background:rgba(198,40,40,0.12);  color:#B71C1C; }
hr { border-color:#BBDEFB !important; margin:18px 0; }
.js-plotly-plot { border-radius:12px; overflow:hidden; border:1px solid #BBDEFB; }
</style>
""", unsafe_allow_html=True)

CHART_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="rgba(255,255,255,1)",
    plot_bgcolor="rgba(238,242,255,1)",
    font=dict(family="Inter, sans-serif", color="#1A3A6B", size=12),
    legend=dict(bgcolor="rgba(255,255,255,0.95)", bordercolor="#BBDEFB", borderwidth=1, font=dict(size=11)),
    margin=dict(l=50, r=20, t=50, b=50),
    xaxis=dict(gridcolor="#BBDEFB", linecolor="#90CAF9", zerolinecolor="#90CAF9"),
    yaxis=dict(gridcolor="#BBDEFB", linecolor="#90CAF9", zerolinecolor="#90CAF9"),
)

# ======================================================
# SIDEBAR
# ======================================================
with st.sidebar:
    st.markdown("""
    <div style="padding: 16px 0 8px 0; text-align: center;">
        <div style="font-size: 2rem;">📊</div>
        <div style="font-weight: 700; font-size: 0.95rem; color: #0D1B4B; margin-top: 6px;">
            Dashboard Forecasting KB
        </div>
        <div style="font-size: 0.72rem; color: #64748B; margin-top: 4px;">
            Kecamatan Anggana · Prophet Time Series
        </div>
    </div>
    <hr style="border-color: rgba(255,255,255,0.2); margin: 12px 0;">
    """, unsafe_allow_html=True)

    st.markdown('<p style="font-size:0.7rem; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:#475569; margin-bottom:8px;">📂 Dataset</p>', unsafe_allow_html=True)
    file = st.file_uploader("Upload File Excel (.xlsx)", type=["xlsx"], label_visibility="collapsed")

    st.markdown('<hr style="border-color:#BBDEFB; margin:16px 0;">', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.7rem; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:#475569; margin-bottom:10px;">📋 Tahapan Analisis</p>', unsafe_allow_html=True)

    # Menu sesuai laporan
    menu_items = [
        ("🎯", "Business Understanding"),
        ("📂", "Data Understanding"),
        ("🛠️", "Data Preparation"),
        ("🤖", "Modeling Prophet"),
        ("📈", "Forecasting"),
        ("✅", "Evaluasi Model"),
        ("📉", "Analisis Tren"),
        ("💊", "Preferensi Kontrasepsi"),
    ]

    tahap = st.radio(
        "menu",
        [f"{e} {t}" for e, t in menu_items],
        label_visibility="collapsed"
    )

    st.markdown('<hr style="border-color:#BBDEFB; margin:16px 0;">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.7rem; color:#475569; line-height:1.6;">
        <strong style="color:#90CAF9;">Penulis:</strong> Muhammad Rizqi Ramadani<br>
        <strong style="color:#90CAF9;">NIM:</strong> 236151015<br>
        <strong style="color:#90CAF9;">Institusi:</strong> Politeknik Negeri Samarinda<br>
        <strong style="color:#90CAF9;">Prodi:</strong> D3 Teknik Informatika<br>
        <strong style="color:#90CAF9;">Tahun:</strong> 2025/2026
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# HELPERS
# ======================================================
def page_header(icon, title, subtitle=""):
    st.markdown(f"""
    <div style="margin-bottom: 24px;">
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:4px;">
            <span style="font-size:1.8rem;">{icon}</span>
            <h2 style="margin:0; font-size:1.5rem; font-weight:700; color:#0D1B4B !important;">{title}</h2>
        </div>
        {"" if not subtitle else f'<p style="margin:0 0 0 52px; color:#64748B; font-size:0.875rem;">{subtitle}</p>'}
        <div style="height:3px; background:linear-gradient(90deg,#2563EB,transparent); border-radius:99px; margin-top:12px;"></div>
    </div>
    """, unsafe_allow_html=True)

def info_card(text, color=PRIMARY):
    st.markdown(f"""
    <div style="background:rgba(37,99,235,0.06); border-left:3px solid {color};
                border-radius:0 10px 10px 0; padding:14px 18px; margin:10px 0;">
        <p style="margin:0; color:#1A3A6B; font-size:0.875rem; line-height:1.7;">{text}</p>
    </div>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data(uploaded_file):
    return pd.read_excel(uploaded_file)

# ======================================================
# WELCOME SCREEN — no file
# ======================================================
if not file:
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown("""
        <div style="text-align:center; padding: 60px 0 40px 0;">
            <div style="font-size:4rem; margin-bottom:16px;">📊</div>
            <h1 style="font-size:1.8rem; font-weight:700; color:#0D1B4B !important; margin-bottom:8px;">
                Dashboard Forecasting Peserta KB
            </h1>
            <p style="color:#64748B; font-size:0.95rem; margin-bottom:8px;">
                Analisis Tren & Prediksi · Time Series Prophet
            </p>
            <p style="color:#475569; font-size:0.8rem; margin-bottom:32px;">
                Kecamatan Anggana, Kabupaten Kutai Kartanegara
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#FFFFFF; border:1px dashed #334155; border-radius:16px;
                    padding:32px; text-align:center; margin-bottom:24px;">
            <div style="font-size:2rem; margin-bottom:12px;">⬆️</div>
            <p style="color:#1A3A6B; font-weight:600; margin-bottom:4px;">Upload file Excel dataset KB</p>
            <p style="color:#475569; font-size:0.8rem; margin-bottom:0;">
                Gunakan sidebar kiri → bagian <strong style="color:#546E7A;">Dataset</strong><br>
                Format: <code style="background:#E3F2FD; padding:2px 6px; border-radius:4px; color:#60A5FA;">.xlsx</code>
                dengan kolom <code style="background:#E3F2FD; padding:2px 6px; border-radius:4px; color:#60A5FA;">ds</code> &
                <code style="background:#E3F2FD; padding:2px 6px; border-radius:4px; color:#60A5FA;">total_peserta</code>
            </p>
        </div>
        """, unsafe_allow_html=True)

        features = [
            ("🎯", "Business Understanding", "Latar belakang, rumusan masalah, tujuan, dan alur penelitian"),
            ("📂", "Data Acquisition", "Tampilkan data mentah SIGA BKKBN dan informasi dataset"),
            ("🛠️", "Data Preparation", "Preprocessing: missing value, format Prophet (ds,y), deteksi outlier IQR, pembagian data"),
            ("🤖", "Modeling Prophet", "Konfigurasi parameter, komponen tren & musiman, pelatihan model"),
            ("📈", "Forecasting", "Prediksi 12 bulan ke depan (Jan–Des 2025) dengan interval kepercayaan 95%"),
            ("✅", "Evaluasi Model", "Perhitungan manual MAE, RMSE, MAPE dan perbandingan aktual vs prediksi"),
            ("📉", "Analisis Tren", "Visualisasi tren jangka panjang dan pola musiman tahunan"),
            ("💊", "Preferensi Kontrasepsi", "Analisis pergeseran proporsi penggunaan 7 jenis kontrasepsi"),
        ]
        for icon, title, desc in features:
            st.markdown(f"""
            <div style="background:#FFFFFF; border:1px solid #BBDEFB; border-radius:10px;
                        padding:14px 18px; margin-bottom:10px; display:flex; gap:14px; align-items:center;">
                <span style="font-size:1.4rem;">{icon}</span>
                <div>
                    <div style="font-weight:600; color:#0D1B4B; font-size:0.875rem;">{title}</div>
                    <div style="color:#78909C; font-size:0.78rem;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    st.stop()

# ======================================================
# LOAD DATA
# ======================================================
df = load_data(file)

if not all(col in df.columns for col in ['ds', 'total_peserta']):
    st.error("❌ Kolom wajib tidak ditemukan! Pastikan file Excel memiliki kolom **'ds'** dan **'total_peserta'**.")
    st.stop()

df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
df['total_peserta'] = pd.to_numeric(df['total_peserta'], errors='coerce')
df_model = df[['ds', 'total_peserta']].rename(columns={'total_peserta': 'y'})
df_model = df_model.dropna().sort_values('ds').reset_index(drop=True)

if df_model.empty:
    st.error("Dataset kosong setelah pembersihan data.")
    st.stop()

# Outlier IQR
q1 = df_model['y'].quantile(0.25)
q3 = df_model['y'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = df_model[(df_model['y'] < lower_bound) | (df_model['y'] > upper_bound)]

outliers = df_model[(df_model['y'] < lower_bound) | (df_model['y'] > upper_bound)]

# ✅ UPDATE: Validasi memakai LOOCV (Leave-One-Out Cross Validation)
# Dipilih karena arahan dosen penguji — dataset kecil (36 data, di bawah rasio ideal 1:50)
# tidak boleh divalidasi dengan split rasio tetap (mis. 80:20 / 28:8), karena hasil evaluasi
# hanya berasal dari SATU kali pembagian data sehingga berpotensi bias.
# LOOCV melatih ulang model N kali (N = jumlah data), setiap kali menyisihkan HANYA 1 titik
# sebagai data uji, sisanya (N-1) dipakai untuk latihan — lalu error dari seluruh iterasi dirata-rata.
N_TOTAL = len(df_model)

PROPHET_PARAMS = dict(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10,
    interval_width=0.95
)

@st.cache_resource
def build_model_full(full_df):
    m = Prophet(**PROPHET_PARAMS)
    m.fit(full_df)
    return m

@st.cache_resource
def run_loocv(full_df):
    """Leave-One-Out Cross Validation: latih ulang Prophet N kali,
    setiap iterasi menyisihkan 1 titik data sebagai data uji."""
    rows = []
    for i in range(len(full_df)):
        train_i = full_df.drop(full_df.index[i]).reset_index(drop=True)
        held_out = full_df.iloc[[i]]
        m_i = Prophet(**PROPHET_PARAMS)
        m_i.fit(train_i)
        future_i = held_out[['ds']].reset_index(drop=True)
        forecast_i = m_i.predict(future_i)
        rows.append({
            'Periode':    held_out['ds'].values[0],
            'Aktual':     held_out['y'].values[0],
            'Prediksi':   forecast_i['yhat'].values[0],
            'Batas Bawah': forecast_i['yhat_lower'].values[0],
            'Batas Atas':  forecast_i['yhat_upper'].values[0],
        })
    return pd.DataFrame(rows)

with st.spinner(f"⚙️ Menjalankan LOOCV — melatih ulang model Prophet {N_TOTAL} kali (menyisihkan 1 data tiap iterasi)..."):
    model_full = build_model_full(df_model)
    loocv_df   = run_loocv(df_model)

future_full   = model_full.make_future_dataframe(periods=12, freq='ME')
forecast_full = model_full.predict(future_full)
pred_future   = forecast_full.tail(12)

# Residual table (36 baris — hasil LOOCV, tiap baris = 1 iterasi "sisakan-satu")
hasil_eval = loocv_df.copy()
hasil_eval['Residual']   = hasil_eval['Aktual'] - hasil_eval['Prediksi']
hasil_eval['APE (%)']    = (abs(hasil_eval['Residual']) / hasil_eval['Aktual'] * 100).round(2)

mape = mean_absolute_percentage_error(hasil_eval['Aktual'], hasil_eval['Prediksi']) * 100
mae  = mean_absolute_error(hasil_eval['Aktual'], hasil_eval['Prediksi'])
rmse = np.sqrt(mean_squared_error(hasil_eval['Aktual'], hasil_eval['Prediksi']))



# ===========================================================
# PAGE: BUSINESS UNDERSTANDING
# ===========================================================
if "Business Understanding" in tahap:
    page_header("🎯", "Business Understanding",
                "Tugas Akhir — Politeknik Negeri Samarinda · 2025/2026")

    tab_bg, tab_rm, tab_tujuan, tab_alur = st.tabs(
        ["📌 Latar Belakang", "❓ Rumusan Masalah", "🎯 Tujuan Penelitian", "🔄 Alur Penelitian"]
    )

    with tab_bg:
        st.markdown("""
        <div class="kpi-card">
            <h4>📌 Latar Belakang</h4>
            <div style="height:4px;width:40px;background:#2563EB;border-radius:99px;margin-bottom:14px;"></div>
            <p style="font-size:0.9rem; line-height:1.8; color:#1A3A6B;">
            Program Keluarga Berencana (KB) merupakan pilar penting dalam mengendalikan pertumbuhan
            penduduk dan meningkatkan kesejahteraan keluarga. Di Kecamatan Anggana, jumlah peserta KB
            Modern mengalami fluktuasi yang cukup signifikan — salah satunya penurunan dari
            <strong style="color:#60A5FA;">3.092 peserta (Desember 2023)</strong> menjadi
            <strong style="color:#F87171;">2.654 peserta (Januari 2024)</strong>.
            Penurunan ini dipengaruhi faktor administratif seperti pembaruan data peserta yang melewati
            usia PUS (≥50 tahun) dan perpindahan domisili, bukan semata berkurangnya minat masyarakat.
            </p>
            <p style="font-size:0.9rem; line-height:1.8; color:#1A3A6B; margin-top:10px;">
            Selain itu, terjadi pergeseran preferensi kontrasepsi dari metode jangka pendek menuju
            Metode Kontrasepsi Jangka Panjang (MKJP). Tanpa analisis berbasis data historis, perencanaan
            program dan pengalokasian logistik alat kontrasepsi cenderung kurang optimal.
            </p>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""
            <div class="kpi-card" style="text-align:center; border-top:3px solid #2563EB;">
                <h4>Periode Data</h4><div class="val" style="font-size:1.2rem;">Jan 2022 – Des 2024</div>
                <div class="sub">36 bulan data bulanan</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="kpi-card" style="text-align:center; border-top:3px solid #16A34A;">
                <h4>Sumber Data</h4><div class="val" style="font-size:1.2rem;">SIGA BKKBN</div>
                <div class="sub">Balai KB Kec. Anggana</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown("""
            <div class="kpi-card" style="text-align:center; border-top:3px solid #D97706;">
                <h4>Metode</h4><div class="val" style="font-size:1.2rem;">Time Series Prophet</div>
                <div class="sub">Meta (Facebook) · 2018</div>
            </div>""", unsafe_allow_html=True)

    with tab_rm:
        rumusan = [
            ("1", "#2563EB", "Bagaimana mengimplementasikan metode Time Series menggunakan algoritma Prophet untuk menganalisis tren perkembangan jumlah peserta KB Modern di Kecamatan Anggana berdasarkan data historis periode Januari 2022 hingga Desember 2024?"),
            ("2", "#16A34A", "Bagaimana membangun model peramalan (forecasting) berbasis Time Series Prophet untuk memprediksi jumlah peserta KB Modern di Kecamatan Anggana pada periode satu tahun ke depan beserta evaluasi tingkat akurasi model menggunakan metrik MAE, RMSE, dan MAPE?"),
            ("3", "#D97706", "Bagaimana menganalisis pergeseran preferensi penggunaan jenis kontrasepsi Modern di Kecamatan Anggana berdasarkan pola historis data kepesertaan KB periode Januari 2022 hingga Desember 2024?"),
        ]
        for num, color, teks in rumusan:
            st.markdown(f"""
            <div style="background:{SURFACE}; border:1px solid {BORDER}; border-left:4px solid {color};
                        border-radius:0 12px 12px 0; padding:20px 22px; margin-bottom:14px;">
                <div style="display:flex; gap:14px; align-items:flex-start;">
                    <div style="min-width:32px; height:32px; border-radius:50%; background:{color};
                                display:flex; align-items:center; justify-content:center;
                                font-weight:700; color:white; font-size:0.9rem;">{num}</div>
                    <p style="margin:0; font-size:0.9rem; color:#1A3A6B; line-height:1.7;">{teks}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab_tujuan:
        tujuan = [
            ("📊", "#2563EB", "Target Mayor — Analisis Tren", "Membangun model peramalan jumlah peserta KB Modern menggunakan Time Series Prophet dan menganalisis tren serta pergeseran preferensi kontrasepsi di Kecamatan Anggana."),
            ("🎯", "#16A34A", "Target Mayor — Prediksi & Evaluasi", "Menghasilkan prediksi jumlah peserta KB untuk periode Januari–Desember 2025 dan mengevaluasi performa model menggunakan MAE, RMSE, dan MAPE."),
            ("🛠️", "#7C3AED", "Target Minor — Preprocessing", "Melakukan pemeriksaan missing value, deteksi outlier (IQR), dan transformasi format baku Prophet (kolom ds dan y)."),
            ("📱", "#D97706", "Target Minor — Aplikasi", "Membangun dashboard interaktif berbasis Streamlit sebagai media visualisasi hasil analisis tren, prediksi, dan pergeseran preferensi kontrasepsi."),
            ("🎓", "#475569", "TPP2", "Memenuhi capaian Tujuan Pendidikan Program Studi — Pengembangan Perangkat Lunak dan Pemrograman (TPP2)."),
        ]
        for icon, color, title, desc in tujuan:
            st.markdown(f"""
            <div style="background:{SURFACE}; border:1px solid {BORDER}; border-radius:12px;
                        padding:18px 20px; margin-bottom:12px; display:flex; gap:16px;">
                <div style="min-width:40px; height:40px; border-radius:10px; background:{color}22;
                            display:flex; align-items:center; justify-content:center; font-size:1.2rem;">{icon}</div>
                <div>
                    <div style="font-weight:700; font-size:0.9rem; color:{color}; margin-bottom:4px;">{title}</div>
                    <p style="margin:0; font-size:0.85rem; color:#546E7A; line-height:1.6;">{desc}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab_alur:
        steps = [
            ("1","#2563EB","Pengumpulan Data","Wawancara dengan Penyuluh KB Kec. Anggana dan pengumpulan data historis dari SIGA BKKBN periode Jan 2022–Des 2024 (36 data poin)"),
            ("2","#7C3AED","Data Cleaning & Preprocessing","Pemeriksaan missing value, konversi format tanggal ke ds (YYYY-MM-DD) dan kolom target y, serta transformasi struktur data sesuai standar Prophet"),
            ("3","#16A34A","Deteksi Outlier (IQR)","Identifikasi data ekstrem menggunakan Interquartile Range — batas bawah Q1−1.5×IQR, batas atas Q3+1.5×IQR. Outlier riil dipertahankan"),
            ("4","#D97706","Validasi Data (LOOCV)","Leave-One-Out Cross Validation: model dilatih ulang 36 kali, tiap iterasi menyisihkan 1 titik data sebagai data uji"),
            ("5","#0891B2","Pemodelan Prophet","Pelatihan model Prophet dengan yearly_seasonality=True, interval_width=0.95. Dekomposisi tren dan musiman"),
            ("6","#DC2626","Forecasting","Prediksi 12 bulan ke depan (Jan–Des 2025) menggunakan model penuh yang dilatih dengan seluruh 36 data"),
            ("7","#059669","Evaluasi Model","Pengukuran akurasi: MAE, RMSE, MAPE. Perhitungan manual per periode data uji"),
            ("8","#7C3AED","Visualisasi & Analisis","Penyajian hasil tren, prediksi, evaluasi, dan pergeseran preferensi kontrasepsi dalam dashboard interaktif"),
        ]
        for i in range(0, len(steps), 2):
            c1, c2 = st.columns(2)
            for col, step in zip([c1, c2], steps[i:i+2]):
                num, color, title, desc = step
                with col:
                    st.markdown(f"""
                    <div style="background:{SURFACE}; border:1px solid {BORDER}; border-radius:12px;
                                padding:16px 18px; margin-bottom:12px;">
                        <div style="display:flex; gap:12px; align-items:center; margin-bottom:8px;">
                            <div style="min-width:32px; height:32px; border-radius:50%; background:{color};
                                        display:flex; align-items:center; justify-content:center;
                                        font-weight:700; color:white; font-size:0.85rem;">{num}</div>
                            <div style="font-weight:700; font-size:0.9rem; color:#0D1B4B;">{title}</div>
                        </div>
                        <p style="margin:0 0 0 44px; font-size:0.8rem; color:#546E7A; line-height:1.6;">{desc}</p>
                    </div>
                    """, unsafe_allow_html=True)

# ===========================================================
# PAGE: DATA Data Understanding
# ===========================================================
elif "Data Understanding" in tahap:
    page_header("📂", "Data Understanding",
                "Data mentah peserta KB Modern dari SIGA BKKBN — Kecamatan Anggana 2022–2024")

    info_card("""
    <strong>Sumber data:</strong> Laporan bulanan peserta KB Modern dari aplikasi Sistem Informasi Keluarga (SIGA) BKKBN,
    diperoleh melalui Balai Penyuluhan KB Kecamatan Anggana. Data mencakup periode Januari 2022 hingga Desember 2024
    dengan frekuensi bulanan (36 data poin).
    """)

    # KPI row
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.metric("Total Baris Data", f"{len(df_model):,}")
    with c2: st.metric("Jumlah Variabel", f"{len(df.columns)}")
    with c3: st.metric("Periode Awal", df_model['ds'].min().strftime('%b %Y'))
    with c4: st.metric("Periode Akhir", df_model['ds'].max().strftime('%b %Y'))
    with c5: st.metric("Frekuensi", "Bulanan")

    st.markdown("")

    col_a, col_b = st.columns([2,1])
    with col_a:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_model['ds'], y=df_model['y'],
            mode='lines+markers',
            line=dict(color='#60A5FA', width=2),
            marker=dict(size=5, color='#60A5FA'),
            fill='tozeroy', fillcolor='rgba(96,165,250,0.08)',
            name='Total Peserta KB',
            hovertemplate='<b>%{x|%b %Y}</b><br>Peserta: %{y:,}<extra></extra>'
        ))
        if not outliers.empty:
            fig.add_trace(go.Scatter(
                x=outliers['ds'], y=outliers['y'],
                mode='markers',
                marker=dict(color='#F87171', size=10, symbol='circle', line=dict(color='white',width=1)),
                name='Mendekati Batas Atas IQR',
                hovertemplate='<b>%{x|%b %Y}</b><br>Peserta: %{y:,}<extra></extra>'
            ))
        fig.update_layout(**CHART_LAYOUT,
            title=dict(text='Grafik Historis Jumlah Peserta KB Modern (2022–2024)', font=dict(size=14, color='#0D1B4B')),
            xaxis_title='Periode', yaxis_title='Jumlah Peserta', height=380, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        stats = df_model['y'].describe()
        st.markdown("""<div class="kpi-card"><h4>📊 Statistik Deskriptif</h4>
        <div style="height:3px;width:30px;background:#2563EB;border-radius:99px;margin-bottom:14px;"></div>""",
        unsafe_allow_html=True)
        for label, val in [
            ("Jumlah Data (N)", f"{int(stats['count'])} Data Poin"),
            ("Nilai Minimum", f"{int(stats['min']):,} Peserta"),
            ("Nilai Maksimum", f"{int(stats['max']):,} Peserta"),
            ("Rata-rata (Mean)", f"{stats['mean']:,.2f} Peserta"),
            ("Median", f"{df_model['y'].median():,.0f} Peserta"),
            ("Std Deviasi", f"{stats['std']:,.2f}"),
            ("Q1 (P25)", f"{stats['25%']:,.0f}"),
            ("Q3 (P75)", f"{stats['75%']:,.0f}"),
        ]:
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; padding:7px 0; border-bottom:1px solid #E3F2FD;">
                <span style="color:#546E7A; font-size:0.82rem;">{label}</span>
                <span style="color:#0D1B4B; font-weight:600; font-size:0.85rem;">{val}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Data mentah interaktif
    metode_cols = [c for c in ['Suntik','Pil','Kondom','Implant','IUD','MOW','MOP'] if c in df.columns]
    col_show = ['ds','total_peserta'] + metode_cols + ['total_pus'] if 'total_pus' in df.columns else ['ds','total_peserta'] + metode_cols

    with st.expander("📋 Tampilkan Data Mentah Lengkap (Klik untuk buka)", expanded=False):
        tahun_filter = st.multiselect("Filter Tahun:", [2022,2023,2024], default=[2022,2023,2024])
        df_show = df[df['ds'].dt.year.isin(tahun_filter)][col_show].copy()
        df_show['ds'] = df_show['ds'].dt.strftime('%B %Y')
        df_show.columns = ['Periode','Total Peserta'] + metode_cols + (['Total PUS'] if 'total_pus' in df.columns else [])
        st.dataframe(df_show, use_container_width=True, height=320)
        st.caption(f"Menampilkan {len(df_show)} dari {len(df)} baris data")

# ===========================================================
# PAGE: DATA PREPARATION
# ===========================================================
elif "Data Preparation" in tahap:
    page_header("🛠️", "Data Preparation",
                "Preprocessing data: missing value, format Prophet, deteksi outlier IQR, dan pembagian data")

    tab_clean, tab_iqr, tab_split = st.tabs(
        ["🧹 Data Cleaning & Format", "📡 Deteksi Outlier (IQR)", "🔁 Validasi LOOCV"]
    )

    with tab_clean:
        info_card("Data mentah dari SIGA diproses melalui 3 tahap sebelum masuk ke pemodelan Prophet.")

        col1, col2 = st.columns([1,1.2])
        with col1:
            steps_clean = [
                ("🗓️","Konversi Datetime","Kolom tanggal dikonversi ke format datetime (YYYY-MM-DD) menggunakan pd.to_datetime(). Baris dengan tanggal tidak valid dihapus."),
                ("🔢","Konversi Numerik","Kolom total_peserta dikonversi ke tipe numerik. Nilai yang tidak valid diubah ke NaN lalu dihapus."),
                ("🔃","Rename & Sort","Kolom diubah ke format baku Prophet: tanggal → ds, jumlah peserta → y. Data diurutkan kronologis."),
                ("✅","Cek Missing Value","Hasil pemeriksaan menunjukkan tidak ada missing value pada 36 data poin. Proses imputasi tidak diperlukan."),
            ]
            for icon, title, desc in steps_clean:
                st.markdown(f"""
                <div style="display:flex; gap:12px; margin-bottom:14px; align-items:flex-start;">
                    <span style="font-size:1.2rem; min-width:26px;">{icon}</span>
                    <div>
                        <div style="font-weight:600; font-size:0.88rem; color:#0D1B4B; margin-bottom:3px;">{title}</div>
                        <div style="font-size:0.8rem; color:#546E7A; line-height:1.55;">{desc}</div>
                    </div>
                </div>""", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-title" style="margin-top:0;">Hasil Format Baku Prophet</div>', unsafe_allow_html=True)
            preview = df_model.head(6).copy()
            preview['ds'] = preview['ds'].dt.strftime('%Y-%m-%d')
            st.dataframe(preview.rename(columns={'ds':'ds (datestamp)','y':'y (target)'}),
                         use_container_width=True, height=220)
            st.caption("Format baku Prophet: kolom ds (tanggal) dan y (nilai target)")

            miss = df_model.isnull().sum()
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Missing Value (ds)", int(miss['ds']))
            col_m2.metric("Missing Value (y)", int(miss['y']))

    with tab_iqr:
        info_card("""
        Deteksi outlier menggunakan metode <strong>Interquartile Range (IQR)</strong> untuk mengidentifikasi data yang
        menyimpang secara statistik. Rumus: Batas Bawah = Q1 − 1,5 × IQR dan Batas Atas = Q3 + 1,5 × IQR.
        """)

        col_calc, col_result = st.columns([1,1])
        with col_calc:
            st.markdown("""<div class="kpi-card">
            <h4>📐 Perhitungan IQR Manual</h4>
            <div style="height:3px;width:30px;background:#16A34A;border-radius:99px;margin-bottom:14px;"></div>
            """, unsafe_allow_html=True)
            for label, val, formula in [
                ("Q1 (Persentil ke-25)", f"{q1:,.0f} peserta", "quantile(0.25)"),
                ("Q3 (Persentil ke-75)", f"{q3:,.0f} peserta", "quantile(0.75)"),
                ("IQR = Q3 − Q1",       f"{iqr:,.0f}",         f"{q3:,.0f} − {q1:,.0f}"),
                ("Batas Bawah",         f"{lower_bound:,.0f} peserta", f"{q1:,.0f} − 1,5 × {iqr:,.0f}"),
                ("Batas Atas",          f"{upper_bound:,.0f} peserta", f"{q3:,.0f} + 1,5 × {iqr:,.0f}"),
            ]:
                st.markdown(f"""
                <div style="padding:9px 0; border-bottom:1px solid #E3F2FD;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
                        <span style="color:#546E7A; font-size:0.82rem;">{label}</span>
                        <span style="font-weight:700; color:#4ADE80; font-size:0.88rem;">{val}</span>
                    </div>
                    <div style="font-size:0.72rem; color:#475569; font-style:italic;">{formula}</div>
                </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_result:
            n_out = len(outliers)
            status_color = "#F87171" if n_out > 0 else "#4ADE80"
            st.markdown(f"""
            <div class="kpi-card" style="border-top:3px solid {status_color};">
                <h4>🔍 Hasil Deteksi</h4>
                <div style="text-align:center; padding:16px 0;">
                    <div style="font-size:3rem; font-weight:700; color:{status_color};">{n_out}</div>
                    <div style="color:#546E7A; font-size:0.85rem;">data terdeteksi outlier</div>
                </div>
            </div>""", unsafe_allow_html=True)

            if n_out > 0:
                st.dataframe(
                    outliers.assign(ds=outliers['ds'].dt.strftime('%B %Y'))
                            .rename(columns={'ds':'Periode','y':'Total Peserta'}),
                    use_container_width=True)
                info_card("""
                Data Desember 2024 (3.301 peserta) mendekati batas atas IQR.
                Berdasarkan hasil wawancara dengan Penyuluh KB Kec. Anggana, lonjakan ini merupakan
                <strong>outlier riil</strong> — terjadi karena intensifikasi pelayanan KB menjelang akhir tahun anggaran.
                Data <strong>dipertahankan</strong> karena Prophet bersifat <em>robust</em> terhadap data ekstrem.
                """, color="#16A34A")

        # Visualisasi IQR
        fig_iqr = go.Figure()
        colors_pt = ['#F87171' if v > upper_bound or v < lower_bound else '#60A5FA' for v in df_model['y']]
        fig_iqr.add_trace(go.Scatter(
            x=df_model['ds'], y=df_model['y'],
            mode='lines+markers',
            line=dict(color='#60A5FA', width=2),
            marker=dict(color=colors_pt, size=7, line=dict(color='white', width=0.5)),
            name='Peserta KB',
            hovertemplate='<b>%{x|%b %Y}</b><br>Peserta: %{y:,}<extra></extra>'
        ))
        fig_iqr.add_hline(y=upper_bound, line_dash="dash", line_color="#F87171", line_width=1.5,
                          annotation_text=f"Batas Atas = {upper_bound:,.0f}", annotation_position="top right")
        fig_iqr.add_hline(y=lower_bound, line_dash="dash", line_color="#F59E0B", line_width=1.5,
                          annotation_text=f"Batas Bawah = {lower_bound:,.0f}", annotation_position="bottom right")
        fig_iqr.update_layout(**CHART_LAYOUT,
            title=dict(text='Visualisasi Batas Outlier IQR pada Data Peserta KB', font=dict(size=13, color='#0D1B4B')),
            height=320, xaxis_title='Periode', yaxis_title='Jumlah Peserta')
        st.plotly_chart(fig_iqr, use_container_width=True)

    with tab_split:
        info_card(f"""
        Validasi model memakai <strong>Leave-One-Out Cross Validation (LOOCV)</strong> — bukan split rasio tetap.
        Dengan data historis kecil (<strong>{N_TOTAL} data</strong>, di bawah rasio ideal 1:50), split rasio tetap
        (mis. 80:20) berisiko bias karena evaluasi hanya berasal dari <em>satu kali</em> pembagian data.
        LOOCV melatih ulang model <strong>{N_TOTAL} kali</strong>; setiap iterasi menyisihkan <strong>1 titik data</strong>
        sebagai data uji, sisanya ({N_TOTAL-1} data) dipakai untuk latihan.
        """)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="kpi-card" style="text-align:center; border-top:3px solid #2563EB;">
                <h4>Jumlah Iterasi</h4>
                <div class="val">{N_TOTAL}</div>
                <div class="sub">1 iterasi per titik data</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="kpi-card" style="text-align:center; border-top:3px solid #16A34A;">
                <h4>Data Latih / Iterasi</h4>
                <div class="val">{N_TOTAL-1}</div>
                <div class="sub">N−1 data</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="kpi-card" style="text-align:center; border-top:3px solid #D97706;">
                <h4>Data Uji / Iterasi</h4>
                <div class="val">1</div>
                <div class="sub">titik yang disisihkan</div>
            </div>""", unsafe_allow_html=True)

        fig_split = go.Figure()
        fig_split.add_trace(go.Scatter(
            x=df_model['ds'], y=df_model['y'], mode='lines+markers',
            name='Seluruh Data (36 titik)',
            line=dict(color='#60A5FA', width=2.2),
            marker=dict(size=6, color='#60A5FA'),
            fill='tozeroy', fillcolor='rgba(96,165,250,0.08)',
            hovertemplate='<b>%{x|%b %Y}</b><br>Peserta: %{y:,}<extra></extra>'
        ))
        fig_split.update_layout(**CHART_LAYOUT,
            title=dict(text='Seluruh Data yang Divalidasi via LOOCV (tiap titik bergantian jadi data uji)', font=dict(size=13, color='#0D1B4B')),
            height=320, xaxis_title='Periode', yaxis_title='Jumlah Peserta', hovermode='x unified')
        st.plotly_chart(fig_split, use_container_width=True)

        info_card("""
        <strong>Contoh cara kerja:</strong> pada iterasi ke-1, data <em>Jan 2022</em> disisihkan sebagai data uji, model
        dilatih dengan 35 data sisanya (Feb 2022 – Des 2024), lalu dipakai memprediksi Jan 2022. Pada iterasi ke-2, gantian
        <em>Feb 2022</em> yang disisihkan, dst. — sampai seluruh 36 titik data pernah menjadi data uji tepat satu kali.
        """, color="#16A34A")

        st.markdown('<div class="section-title">📋 Tabel Skema Iterasi LOOCV</div>', unsafe_allow_html=True)
        skema = pd.DataFrame({
            'Iterasi ke-': range(1, N_TOTAL+1),
            'Data Disisihkan (Uji)': df_model['ds'].dt.strftime('%B %Y'),
            'Jumlah Data Latih': N_TOTAL-1,
        })
        st.dataframe(skema, use_container_width=True, height=260)



# ===========================================================
# PAGE: MODELING PROPHET
# ===========================================================
elif "Modeling" in tahap:
    page_header("🤖", "Modeling Prophet",
                "Konfigurasi parameter, pelatihan model, dan dekomposisi komponen tren & musiman")

    info_card("""
    Model <strong>Prophet</strong> final dibangun menggunakan seluruh data historis (36 data poin). Prophet mendekomposisi data menjadi
    tiga komponen: <strong>g(t)</strong> tren, <strong>s(t)</strong> musiman, dan <strong>ε</strong> residual.
    Formula: <code>y(t) = g(t) + s(t) + h(t) + ε</code>
    """)

    tab_param, tab_komponen, tab_data = st.tabs(["⚙️ Parameter Model", "📈 Komponen Dekomposisi", "📋 Data Training"])

    with tab_param:
        col1, col2 = st.columns([1,1])
        with col1:
            st.markdown('<div class="section-title">Konfigurasi Parameter</div>', unsafe_allow_html=True)
            params = [
                ("yearly_seasonality", "True", "Aktifkan komponen musiman tahunan", "#4ADE80"),
                ("weekly_seasonality", "False", "Dinonaktifkan — data bulanan, bukan harian", "#64748B"),
                ("daily_seasonality",  "False", "Dinonaktifkan — data bulanan", "#64748B"),
                ("changepoint_prior_scale", "0.05", "Fleksibilitas model terhadap perubahan tren (default)", "#FCD34D"),
                ("seasonality_prior_scale", "10",   "Memperkuat pengaruh komponen musiman", "#60A5FA"),
                ("interval_width",     "0.95", "Interval kepercayaan prediksi sebesar 95%", "#C4B5FD"),
            ]
            for param, val, desc, color in params:
                active = val != "False"
                st.markdown(f"""
                <div style="padding:11px 0; border-bottom:1px solid #E3F2FD;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                        <code style="color:{color}; font-size:0.82rem;">{param}</code>
                        <span style="font-weight:700; color:{'#4ADE80' if active else '#475569'}; font-size:0.88rem;">{val}</span>
                    </div>
                    <div style="font-size:0.75rem; color:#64748B;">{desc}</div>
                </div>""", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-title">Formula Model Prophet</div>', unsafe_allow_html=True)
            st.markdown("""
            <div style="background:#E3F2FD; border-radius:12px; padding:20px; text-align:center; margin-bottom:16px;">
                <div style="font-size:1.15rem; color:#1565C0; font-family:monospace; letter-spacing:0.05em;">
                    y(t) = g(t) + s(t) + h(t) + ε
                </div>
            </div>""", unsafe_allow_html=True)
            for sym, color, desc in [
                ("g(t)", "#60A5FA", "Komponen tren — menangkap pergerakan jangka panjang (naik/turun)"),
                ("s(t)", "#4ADE80", "Komponen musiman tahunan — pola berulang setiap tahun"),
                ("h(t)", "#FCD34D", "Efek hari libur — tidak digunakan dalam penelitian ini"),
                ("ε",   "#F87171", "Residual — variasi yang tidak tertangkap model"),
            ]:
                st.markdown(f"""
                <div style="display:flex; gap:14px; margin-bottom:12px; padding:12px; background:#FFFFFF;
                            border-radius:8px; border-left:3px solid {color};">
                    <code style="color:{color}; font-size:1rem; font-weight:700; min-width:40px;">{sym}</code>
                    <p style="margin:0; font-size:0.83rem; color:#546E7A; line-height:1.5;">{desc}</p>
                </div>""", unsafe_allow_html=True)

    with tab_komponen:
        col_t, col_s = st.columns(2)
        with col_t:
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=forecast_full['ds'], y=forecast_full['trend'],
                mode='lines', name='Komponen Tren',
                line=dict(color='#60A5FA', width=2.5),
                hovertemplate='<b>%{x|%b %Y}</b><br>Nilai Tren: %{y:,.0f}<extra></extra>'
            ))
            fig_trend.update_layout(**CHART_LAYOUT,
                title=dict(text='(a) Komponen Tren', font=dict(size=13, color='#0D1B4B')),
                height=300, xaxis_title='Periode', yaxis_title='Nilai Tren')
            st.plotly_chart(fig_trend, use_container_width=True)
            info_card("Tren menunjukkan penurunan 2022–awal 2023, kemudian meningkat signifikan sejak pertengahan 2023 hingga 2025.")

        with col_s:
            fig_seas = go.Figure()
            fig_seas.add_trace(go.Scatter(
                x=forecast_full['ds'], y=forecast_full['yearly'],
                mode='lines', name='Musiman Tahunan',
                line=dict(color='#4ADE80', width=2.5),
                fill='tozeroy', fillcolor='rgba(74,222,128,0.08)',
                hovertemplate='<b>%{x|%b %Y}</b><br>Efek Musiman: %{y:+.0f}<extra></extra>'
            ))
            fig_seas.add_hline(y=0, line_color='#475569', line_width=1)
            fig_seas.update_layout(**CHART_LAYOUT,
                title=dict(text='(b) Komponen Musiman Tahunan', font=dict(size=13, color='#0D1B4B')),
                height=300, xaxis_title='Periode', yaxis_title='Efek Musiman')
            st.plotly_chart(fig_seas, use_container_width=True)
            info_card("Efek musiman positif (naik) pada pertengahan–akhir tahun. Efek negatif (turun) pada Maret–April setiap tahun.")

    with tab_data:
        st.markdown(f'<div class="section-title">Data Pelatihan Model Penuh ({N_TOTAL} data poin — Jan 2022–Des 2024)</div>', unsafe_allow_html=True)
        info_card("""
        Model final (<code>model_full</code>) dilatih menggunakan <strong>seluruh data historis</strong> agar prediksi 12 bulan
        ke depan memanfaatkan informasi selengkap mungkin. Validasi performa model tetap dilakukan terpisah lewat LOOCV
        (lihat tab <strong>🔁 Validasi LOOCV</strong> di halaman Data Preparation).
        """)
        st.dataframe(df_model.assign(ds=df_model['ds'].dt.strftime('%B %Y')).rename(columns={'ds':'Periode','y':'Total Peserta'}), use_container_width=True, height=420)


# ===========================================================
# PAGE: FORECASTING
# ===========================================================
elif "Forecasting" in tahap:
    page_header("📈", "Forecasting",
                "Prediksi jumlah peserta KB Modern periode Januari–Desember 2025 menggunakan model Prophet penuh")

    info_card("""
    Model Prophet dilatih ulang menggunakan <strong>seluruh 36 data poin</strong> (Jan 2022–Des 2024) untuk menghasilkan
    prediksi yang lebih akurat. Prediksi dilakukan untuk <strong>12 bulan ke depan</strong> dengan interval kepercayaan 95%.
    """)

    last_actual = df_model['y'].iloc[-1]
    first_pred  = pred_future['yhat'].iloc[0]
    last_pred   = pred_future['yhat'].iloc[-1]
    delta_pct   = (last_pred - last_actual) / last_actual * 100

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Peserta Terakhir (Des 2024)", f"{int(last_actual):,}")
    with c2: st.metric("Prediksi Jan 2025", f"{int(first_pred):,}")
    with c3: st.metric("Prediksi Des 2025", f"{int(last_pred):,}")
    with c4: st.metric("Perubahan vs Aktual", f"{delta_pct:+.1f}%",
                       delta="Naik" if delta_pct > 0 else "Turun",
                       delta_color="normal" if delta_pct > 0 else "inverse")

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=df_model['ds'], y=df_model['y'], mode='lines',
        name='Data Historis (2022–2024)',
        line=dict(color='#94A3B8', width=1.8),
        hovertemplate='<b>%{x|%b %Y}</b><br>Aktual: %{y:,}<extra></extra>'
    ))
    fig_pred.add_trace(go.Scatter(
        x=pred_future['ds'], y=pred_future['yhat_upper'],
        mode='lines', line=dict(width=0), showlegend=False))
    fig_pred.add_trace(go.Scatter(
        x=pred_future['ds'], y=pred_future['yhat_lower'],
        mode='lines', fill='tonexty', fillcolor='rgba(167,139,250,0.18)',
        line=dict(width=0), name='Interval Kepercayaan 95%'))
    fig_pred.add_trace(go.Scatter(
        x=pred_future['ds'], y=pred_future['yhat'],
        mode='lines+markers', name='Prediksi Prophet 2025',
        line=dict(color='#A78BFA', width=2.5, dash='dash'),
        marker=dict(size=7, color='#A78BFA'),
        hovertemplate='<b>%{x|%b %Y}</b><br>Prediksi: %{y:,}<extra></extra>'
    ))
    fig_pred.add_vline(x=df_model['ds'].iloc[-1], line_dash="dot",
                       line_color="#D97706", line_width=1.5,
                       annotation_text="Des 2024", annotation_position="top left")
    fig_pred.update_layout(**CHART_LAYOUT,
        title=dict(text='Forecast Peserta KB Modern — Januari s.d. Desember 2025', font=dict(size=14, color='#0D1B4B')),
        height=420, xaxis_title='Periode', yaxis_title='Jumlah Peserta', hovermode='x unified')
    st.plotly_chart(fig_pred, use_container_width=True)

    col_tbl, col_int = st.columns([1.5,1])
    with col_tbl:
        st.markdown('<div class="section-title">📋 Tabel Hasil Prediksi Jan–Des 2025</div>', unsafe_allow_html=True)
        tbl = pred_future[['ds','yhat','yhat_lower','yhat_upper']].copy()
        tbl.columns = ['Periode','Prediksi','Batas Bawah (95%)','Batas Atas (95%)']
        tbl['Periode'] = tbl['Periode'].dt.strftime('%B %Y')
        for col_n in ['Prediksi','Batas Bawah (95%)','Batas Atas (95%)']:
            tbl[col_n] = tbl[col_n].round(0).astype(int)
        st.dataframe(tbl, use_container_width=True, height=390)

    with col_int:
        info_card("""
        <strong>Interval Kepercayaan 95%</strong> menunjukkan rentang kemungkinan nilai aktual.
        Semakin jauh periode prediksi, semakin lebar interval karena ketidakpastian bertambah.
        <br><br>
        Nilai <strong>yhat_lower</strong> = batas bawah estimasi<br>
        Nilai <strong>yhat_upper</strong> = batas atas estimasi<br>
        Nilai yang jatuh di dalam rentang ini dianggap <strong>valid</strong> sesuai model.
        """)
        trend_naik = last_pred > first_pred
        st.markdown(f"""
        <div class="kpi-card" style="text-align:center; border-top:3px solid {'#4ADE80' if trend_naik else '#F87171'}; margin-top:14px;">
            <h4>Interpretasi Tren</h4>
            <div style="font-size:3rem; padding:12px 0;">{'📈' if trend_naik else '📉'}</div>
            <div style="font-size:1rem; font-weight:700; color:{'#4ADE80' if trend_naik else '#F87171'};">
                Tren {'Meningkat' if trend_naik else 'Menurun'}
            </div>
            <div style="font-size:0.8rem; color:#546E7A; margin-top:8px;">
                dari {int(first_pred):,} → {int(last_pred):,} peserta
            </div>
        </div>""", unsafe_allow_html=True)

# ===========================================================
# PAGE: EVALUASI MODEL
# ===========================================================
elif "Evaluasi" in tahap:
    page_header("✅", "Evaluasi Model",
                f"Pengukuran akurasi Prophet menggunakan MAE, RMSE, dan MAPE — validasi LOOCV ({N_TOTAL} iterasi)")

    def mape_cat(m):
        if m < 10:  return ("Sangat Akurat", "#4ADE80", "badge-green", "< 10%")
        elif m < 20: return ("Akurat", "#60A5FA", "badge-blue", "10–20%")
        elif m < 50: return ("Cukup Akurat", "#FCD34D", "badge-amber", "20–50%")
        else:        return ("Kurang Akurat", "#F87171", "badge-red", "> 50%")

    cat, cat_color, cat_badge, cat_range = mape_cat(mape)

    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="kpi-card" style="text-align:center; border-top:3px solid {cat_color};">
            <h4>MAPE</h4>
            <div class="val" style="color:{cat_color} !important;">{mape:.2f}%</div>
            <div class="sub">Mean Absolute Percentage Error</div>
            <div style="margin-top:12px;"><span class="badge {cat_badge}">{cat}</span></div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card" style="text-align:center; border-top:3px solid #60A5FA;">
            <h4>MAE</h4>
            <div class="val">{mae:.2f}</div>
            <div class="sub">Mean Absolute Error</div>
            <div style="margin-top:12px; font-size:0.78rem; color:#64748B;">rata-rata selisih absolut</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card" style="text-align:center; border-top:3px solid #C4B5FD;">
            <h4>RMSE</h4>
            <div class="val">{rmse:.2f}</div>
            <div class="sub">Root Mean Squared Error</div>
            <div style="margin-top:12px; font-size:0.78rem; color:#64748B;">penalti lebih besar utk outlier</div>
        </div>""", unsafe_allow_html=True)

    info_card(f"""
    Metrik di atas dihitung dari <strong>{N_TOTAL} iterasi LOOCV</strong>: tiap titik data historis bergantian disisihkan
    sebagai data uji (diprediksi oleh model yang dilatih dari {N_TOTAL-1} data lainnya), lalu error dari seluruh iterasi
    dirata-ratakan. Ini menggantikan evaluasi single-split (80:20) sesuai arahan dosen penguji untuk dataset kecil.
    """, color="#2563EB")

    tab_grafik, tab_hitung, tab_tabel = st.tabs(
        ["📊 Grafik Perbandingan", "🔢 Perhitungan Manual", "📋 Tabel Detail"]
    )

    with tab_grafik:
        fig_eval = go.Figure()
        fig_eval.add_trace(go.Scatter(
            x=hasil_eval['Periode'], y=hasil_eval['Aktual'], mode='lines+markers',
            name='Data Aktual', line=dict(color='#475569', width=1.8),
            marker=dict(size=7, color='#475569'),
            hovertemplate='<b>%{x|%b %Y}</b><br>Aktual: %{y:,}<extra></extra>'))
        fig_eval.add_trace(go.Scatter(
            x=hasil_eval['Periode'], y=hasil_eval['Prediksi'], mode='lines+markers',
            name='Prediksi LOOCV', line=dict(color='#F87171', width=2.5, dash='dash'),
            marker=dict(size=7, color='#F87171', symbol='square'),
            hovertemplate='<b>%{x|%b %Y}</b><br>Prediksi: %{y:,.0f}<extra></extra>'))
        fig_eval.add_trace(go.Scatter(
            x=hasil_eval['Periode'], y=hasil_eval['Batas Atas'], mode='lines', line=dict(width=0), showlegend=False))
        fig_eval.add_trace(go.Scatter(
            x=hasil_eval['Periode'], y=hasil_eval['Batas Bawah'], mode='lines',
            fill='tonexty', fillcolor='rgba(248,113,113,0.1)',
            line=dict(width=0), name='Interval 95%'))
        fig_eval.update_layout(**CHART_LAYOUT,
            title=dict(text=f'Perbandingan Aktual vs Prediksi LOOCV — {N_TOTAL} Titik Data', font=dict(size=14, color='#0D1B4B')),
            height=380, xaxis_title='Periode', yaxis_title='Jumlah Peserta', hovermode='x unified')
        st.plotly_chart(fig_eval, use_container_width=True)



        col_ra, col_rb = st.columns(2)
        with col_ra:
            fig_res = go.Figure()
            fig_res.add_trace(go.Bar(
                x=hasil_eval['Periode'], y=hasil_eval['Residual'],
                marker=dict(color=hasil_eval['Residual'].apply(lambda v: '#4ADE80' if v>=0 else '#F87171')),
                name='Residual', text=hasil_eval['Residual'].round(1),
                textposition='outside', textfont=dict(color='#0D1B4B', size=10),
                hovertemplate='<b>%{x|%b %Y}</b><br>Residual: %{y:,.1f}<extra></extra>'
            ))
            fig_res.add_hline(y=0, line_color='#94A3B8', line_width=1)
            fig_res.update_layout(**CHART_LAYOUT,
                title=dict(text='Analisis Residual (Aktual − Prediksi)', font=dict(size=13, color='#0D1B4B')),
                height=280, xaxis_title='Periode', yaxis_title='Residual')
            st.plotly_chart(fig_res, use_container_width=True)
        with col_rb:
            fig_ape = go.Figure()
            fig_ape.add_trace(go.Bar(
                x=hasil_eval['Periode'], y=hasil_eval['APE (%)'],
                marker=dict(color='#C4B5FD'),
                name='APE (%)', text=hasil_eval['APE (%)'].apply(lambda x: f"{x:.2f}%"),
                textposition='outside', textfont=dict(color='#0D1B4B', size=10),
                hovertemplate='<b>%{x|%b %Y}</b><br>APE: %{y:.2f}%<extra></extra>'
            ))
            fig_ape.add_hline(y=10, line_dash="dash", line_color="#F87171",
                              annotation_text="Batas 10% (Sangat Akurat)")
            fig_ape.update_layout(**CHART_LAYOUT,
                title=dict(text='Absolute Percentage Error (APE) per Periode', font=dict(size=13, color='#0D1B4B')),
                height=280, xaxis_title='Periode', yaxis_title='APE (%)')
            st.plotly_chart(fig_ape, use_container_width=True)

    with tab_hitung:
        info_card(f"Perhitungan manual metrik evaluasi berdasarkan {N_TOTAL} hasil prediksi LOOCV (tiap titik data historis).")
        sum_res = abs(hasil_eval['Residual']).sum()
        sum_sq  = (hasil_eval['Residual']**2).sum()
        sum_ape = hasil_eval['APE (%)'].sum()
        n       = len(hasil_eval)

        col_mae, col_rmse, col_mape = st.columns(3)
        with col_mae:
            st.markdown(f"""
            <div class="kpi-card">
                <h4>MAE — Perhitungan</h4>
                <div style="background:#E3F2FD; border-radius:8px; padding:14px; margin:10px 0; font-family:monospace; font-size:0.85rem; color:#1A3A6B; line-height:2;">
                    MAE = (1/n) × Σ|y − ŷ|<br>
                    MAE = (1/{n}) × {sum_res:.1f}<br>
                    MAE = {sum_res:.1f} / {n}<br>
                    <strong style="color:#4ADE80; font-size:1rem;">MAE = {mae:.2f}</strong>
                </div>
            </div>""", unsafe_allow_html=True)
        with col_rmse:
            st.markdown(f"""
            <div class="kpi-card">
                <h4>RMSE — Perhitungan</h4>
                <div style="background:#E3F2FD; border-radius:8px; padding:14px; margin:10px 0; font-family:monospace; font-size:0.85rem; color:#1A3A6B; line-height:2;">
                    RMSE = √[(1/n) × Σ(y−ŷ)²]<br>
                    RMSE = √[(1/{n}) × {sum_sq:,.2f}]<br>
                    RMSE = √[{sum_sq/n:,.2f}]<br>
                    <strong style="color:#60A5FA; font-size:1rem;">RMSE = {rmse:.2f}</strong>
                </div>
            </div>""", unsafe_allow_html=True)
        with col_mape:
            st.markdown(f"""
            <div class="kpi-card">
                <h4>MAPE — Perhitungan</h4>
                <div style="background:#E3F2FD; border-radius:8px; padding:14px; margin:10px 0; font-family:monospace; font-size:0.85rem; color:#1A3A6B; line-height:2;">
                    MAPE = (1/n) × Σ|y−ŷ|/y × 100%<br>
                    MAPE = (1/{n}) × {sum_ape:.2f}%<br>
                    MAPE = {sum_ape:.2f}% / {n}<br>
                    <strong style="color:{cat_color}; font-size:1rem;">MAPE = {mape:.2f}%</strong>
                </div>
            </div>""", unsafe_allow_html=True)

    with tab_tabel:
        display_eval = hasil_eval.copy()
        display_eval['Periode'] = display_eval['Periode'].dt.strftime('%B %Y')
        display_eval['Aktual']  = display_eval['Aktual'].astype(int)
        display_eval['Prediksi'] = display_eval['Prediksi'].round(0).astype(int)
        display_eval['Residual'] = display_eval['Residual'].round(1)
        display_eval['Residual²'] = (display_eval['Residual']**2).round(2)
        display_eval = display_eval[['Periode','Aktual','Prediksi','Residual','Residual²','APE (%)']]
        st.dataframe(display_eval, use_container_width=True, height=310)

        jumlah_row = pd.DataFrame({
            'Periode':['Jumlah (Σ)'], 'Aktual':[''], 'Prediksi':[''],
            'Residual':[round(abs(display_eval['Residual']).sum(), 1)],
            'Residual²':[round((display_eval['Residual']**2).sum(), 2)],
            'APE (%)':[round(display_eval['APE (%)'].sum(), 2)]
        })

    st.markdown(f"""
    <div class="info-box" style="margin-top:14px;">
        <p>📊 <strong style="color:#93C5FD;">Interpretasi MAPE:</strong> Nilai MAPE sebesar
        <strong style="color:{cat_color};">{mape:.2f}%</strong> mengindikasikan model Prophet
        berada dalam kategori <strong style="color:{cat_color};">{cat}</strong>
        (MAPE &lt; 10% = Sangat Akurat, 10–20% = Akurat, 20–50% = Cukup Akurat, &gt;50% = Kurang Akurat).
        Model dapat diandalkan sebagai dasar pengambilan keputusan perencanaan program KB.</p>
    </div>
    """, unsafe_allow_html=True)

# ===========================================================
# PAGE: ANALISIS TREN
# ===========================================================
elif "Tren" in tahap:
    page_header("📉", "Analisis Tren",
                "Pola tren jangka panjang dan komponen musiman tahunan berdasarkan model Prophet")

    info_card("Analisis tren dilakukan berdasarkan komponen <strong>g(t)</strong> dan <strong>s(t)</strong> dari hasil dekomposisi model Prophet pada data penuh (Jan 2022–Des 2025).")

    trend_start = forecast_full['trend'].iloc[0]
    trend_end   = forecast_full['trend'].iloc[-1]

    c1,c2,c3 = st.columns(3)
    with c1: st.metric("Tren Awal (Jan 2022)", f"{trend_start:,.0f}")
    with c2: st.metric("Tren Akhir (Des 2025)", f"{trend_end:,.0f}")
    with c3: st.metric("Δ Perubahan Tren", f"{trend_end - trend_start:+,.0f}",
                       delta="Meningkat" if trend_end > trend_start else "Menurun",
                       delta_color="normal" if trend_end > trend_start else "inverse")

    fig_all = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.6,0.4], vertical_spacing=0.06,
                            subplot_titles=["Data Historis + Tren + Forecast", "Komponen Musiman Tahunan"])

    fig_all.add_trace(go.Scatter(
        x=df_model['ds'], y=df_model['y'], mode='lines',
        name='Historis', line=dict(color='#60A5FA', width=1.8),
        hovertemplate='<b>%{x|%b %Y}</b><br>Aktual: %{y:,}<extra></extra>'), row=1, col=1)
    fig_all.add_trace(go.Scatter(
        x=pred_future['ds'], y=pred_future['yhat'], mode='lines',
        name='Forecast 2025', line=dict(color='#A78BFA', width=2, dash='dash'),
        hovertemplate='<b>%{x|%b %Y}</b><br>Prediksi: %{y:,.0f}<extra></extra>'), row=1, col=1)
    fig_all.add_trace(go.Scatter(
        x=forecast_full['ds'], y=forecast_full['trend'], mode='lines',
        name='Tren Prophet', line=dict(color='#F59E0B', width=2.5),
        hovertemplate='<b>%{x|%b %Y}</b><br>Tren: %{y:,.0f}<extra></extra>'), row=1, col=1)
    fig_all.add_trace(go.Scatter(
        x=forecast_full['ds'], y=forecast_full['yearly'], mode='lines',
        name='Musiman Tahunan', line=dict(color='#4ADE80', width=1.8),
        fill='tozeroy', fillcolor='rgba(74,222,128,0.07)',
        hovertemplate='<b>%{x|%b %Y}</b><br>Efek: %{y:+.0f}<extra></extra>'), row=2, col=1)
    fig_all.add_hline(y=0, line_color='#334155', line_width=1, row=2, col=1)

    fig_all.update_layout(**CHART_LAYOUT, height=520, hovermode='x unified')
    fig_all.update_yaxes(title_text="Jumlah Peserta", row=1)
    fig_all.update_yaxes(title_text="Efek Musiman", row=2)
    fig_all.update_xaxes(title_text="Periode", row=2)
    st.plotly_chart(fig_all, use_container_width=True)

    col_int1, col_int2 = st.columns(2)
    with col_int1:
        info_card("""
        <strong>Komponen Tren:</strong> Tren menunjukkan penurunan pada 2022–awal 2023, kemudian berbalik
        meningkat secara signifikan sejak pertengahan 2023. Tren diprediksi terus meningkat menuju 2025.
        """, color="#F59E0B")
    with col_int2:
        info_card("""
        <strong>Komponen Musiman:</strong> Efek musiman positif (peningkatan kepesertaan) terjadi pada
        pertengahan–akhir tahun (Okt–Des). Efek negatif (penurunan) terjadi pada Maret–April setiap tahun.
        """, color="#4ADE80")

    st.markdown('<div class="section-title">📊 Rata-rata Peserta KB per Tahun</div>', unsafe_allow_html=True)
    df_yoy = df_model.copy()
    df_yoy['Tahun'] = df_yoy['ds'].dt.year
    yearly = df_yoy.groupby('Tahun')['y'].agg(['mean','min','max']).round(0).astype(int).reset_index()

    col_y1, col_y2 = st.columns([1.5,1])
    with col_y1:
        fig_yr = go.Figure()
        fig_yr.add_trace(go.Bar(
            x=yearly['Tahun'].astype(str), y=yearly['mean'],
            name='Rata-rata', marker=dict(color=['#60A5FA','#4ADE80','#A78BFA']),
            text=yearly['mean'].apply(lambda v: f"{v:,}"),
            textposition='outside', textfont=dict(color='#0D1B4B', size=12),
            hovertemplate='<b>%{x}</b><br>Rata-rata: %{y:,}<extra></extra>'
        ))
        fig_yr.update_layout(**CHART_LAYOUT,
            title=dict(text='Rata-rata Peserta KB per Tahun', font=dict(size=13, color='#0D1B4B')),
            height=300, xaxis_title='Tahun', yaxis_title='Rata-rata Peserta',
            yaxis_range=[0, yearly['max'].max()*1.2])
        st.plotly_chart(fig_yr, use_container_width=True)
    with col_y2:
        st.dataframe(yearly.rename(columns={'Tahun':'Tahun','mean':'Rata-rata','min':'Minimum','max':'Maksimum'}),
                     use_container_width=True, height=250)

# ===========================================================
# PAGE: PREFERENSI KONTRASEPSI
# ===========================================================
elif "Preferensi" in tahap or "Kontrasepsi" in tahap:
    page_header("💊", "Preferensi Kontrasepsi",
                "Analisis pergeseran proporsi penggunaan 7 jenis kontrasepsi Modern di Kecamatan Anggana")

    info_card("""
    Analisis pergeseran preferensi dilakukan dengan membandingkan rata-rata jumlah peserta per jenis kontrasepsi
    antara tahun 2022 dan 2024. Data menunjukkan pergeseran dari kontrasepsi jangka pendek menuju
    <strong>Metode Kontrasepsi Jangka Panjang (MKJP)</strong>.
    """)

    metode_kb   = ['Suntik','Pil','Kondom','Implant','IUD','MOW','MOP']
    kol_ada     = [k for k in metode_kb if k in df.columns]
    COLORS_KB   = ['#60A5FA','#4ADE80','#F59E0B','#F87171','#C4B5FD','#34D399','#FB923C']

    if not kol_ada:
        st.warning("⚠️ Kolom kontrasepsi tidak ditemukan. Pastikan kolom bernama: Suntik, Pil, Kondom, Implant, IUD, MOW, MOP")
        st.stop()

    total_m  = df[kol_ada].sum()
    grand_t  = total_m.sum()

    # KPI per metode
    cols_kpi = st.columns(len(kol_ada))
    for i, (kb, col) in enumerate(zip(kol_ada, cols_kpi)):
        pct  = total_m[kb] / grand_t * 100
        top  = total_m[kb] == total_m.max()
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="text-align:center; border-top:3px solid {COLORS_KB[i%len(COLORS_KB)]};
                 {'box-shadow:0 0 18px rgba(96,165,250,0.18);' if top else ''}">
                <h4>{kb}{'⭐' if top else ''}</h4>
                <div style="font-size:1.3rem; font-weight:700; color:{COLORS_KB[i%len(COLORS_KB)]};">{total_m[kb]:,}</div>
                <div style="font-size:0.72rem; color:#546E7A; margin-top:3px;">{pct:.1f}% total</div>
            </div>""", unsafe_allow_html=True)

    tab_tren, tab_pergeseran, tab_detail = st.tabs(
        ["📈 Tren per Jenis", "🔄 Pergeseran 2022 vs 2024", "📋 Tabel Detail"]
    )

    with tab_tren:
        col_line, col_pie = st.columns([2,1])
        with col_line:
            fig_kb = go.Figure()
            for i, kb in enumerate(kol_ada):
                fig_kb.add_trace(go.Scatter(
                    x=df['ds'], y=df[kb], mode='lines', name=kb,
                    line=dict(color=COLORS_KB[i%len(COLORS_KB)], width=2),
                    hovertemplate=f'<b>{kb}</b><br>%{{x|%b %Y}}<br>Peserta: %{{y:,}}<extra></extra>'
                ))
            fig_kb.update_layout(**CHART_LAYOUT,
                title=dict(text='Tren Penggunaan Berbagai Jenis Kontrasepsi 2022–2024', font=dict(size=14, color='#0D1B4B')),
                height=380, xaxis_title='Periode', yaxis_title='Jumlah Peserta', hovermode='x unified')
            st.plotly_chart(fig_kb, use_container_width=True)
        with col_pie:
            fig_pie = go.Figure()
            fig_pie.add_trace(go.Pie(
                labels=total_m.index.tolist(), values=total_m.values.tolist(),
                hole=0.45, marker=dict(colors=COLORS_KB[:len(kol_ada)]),
                textinfo='label+percent', textfont=dict(size=11, color='#0D1B4B'),
                hovertemplate='<b>%{label}</b><br>Total: %{value:,}<br>%{percent}<extra></extra>'
            ))
            layout_pie = CHART_LAYOUT.copy()
            layout_pie["margin"] = dict(l=10,r=10,t=50,b=10)
            fig_pie.update_layout(**layout_pie,
                title=dict(text='Proporsi Total 2022–2024', font=dict(size=13, color='#0D1B4B')),
                height=380, showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)

    with tab_pergeseran:
        df['Tahun'] = df['ds'].dt.year
        yr_sum  = df.groupby('Tahun')[kol_ada].mean().round(0).astype(int)

        if 2022 in yr_sum.index and 2024 in yr_sum.index:
            pergeseran = []
            for kb in kol_ada:
                v22 = yr_sum.loc[2022, kb]
                v24 = yr_sum.loc[2024, kb]
                chg = v24 - v22
                pct = (chg / v22 * 100) if v22 > 0 else 0
                pergeseran.append({'Kontrasepsi':kb,'Rata² 2022':v22,'Rata² 2024':v24,
                                   'Perubahan':chg,'Persen (%)':round(pct,1)})
            df_p = pd.DataFrame(pergeseran).sort_values('Persen (%)', ascending=False)

            # Highlight chart
            fig_chg = go.Figure()
            fig_chg.add_trace(go.Bar(
                y=df_p['Kontrasepsi'], x=df_p['Persen (%)'], orientation='h',
                marker=dict(color=df_p['Persen (%)'].apply(lambda v: '#4ADE80' if v >= 0 else '#F87171')),
                text=df_p['Persen (%)'].apply(lambda v: f"{v:+.1f}%"),
                textposition='outside', textfont=dict(color='#0D1B4B', size=11),
                hovertemplate='<b>%{y}</b><br>Perubahan: %{x:+.1f}%<extra></extra>'
            ))
            fig_chg.add_vline(x=0, line_color='#475569', line_width=1.5)
            fig_chg.update_layout(**CHART_LAYOUT,
                title=dict(text='Persentase Perubahan Rata-rata Peserta (2022 → 2024)', font=dict(size=13, color='#0D1B4B')),
                height=320, xaxis_title='Perubahan (%)')
            st.plotly_chart(fig_chg, use_container_width=True)

            # Stacked bar proporsi per tahun
            fig_stack = go.Figure()
            yr_pct = yr_sum.div(yr_sum.sum(axis=1), axis=0) * 100
            for i, kb in enumerate(kol_ada):
                fig_stack.add_trace(go.Bar(
                    name=kb, x=yr_pct.index.astype(str), y=yr_pct[kb].round(1),
                    marker=dict(color=COLORS_KB[i%len(COLORS_KB)]),
                    text=yr_pct[kb].round(1).astype(str)+"%",
                    textposition='inside', textfont=dict(size=10, color='white')
                ))
            fig_stack.update_layout(**CHART_LAYOUT,
                barmode='stack',
                title=dict(text='Proporsi Penggunaan Kontrasepsi per Tahun (%)', font=dict(size=13, color='#0D1B4B')),
                height=320, xaxis_title='Tahun', yaxis_title='Proporsi (%)')
            st.plotly_chart(fig_stack, use_container_width=True)

            # Tabel pergeseran
            st.dataframe(df_p.rename(columns={'Persen (%)':'Perubahan (%)'}), use_container_width=True, height=260)

            top_naik  = df_p[df_p['Persen (%)'] > 0]['Kontrasepsi'].tolist()
            top_turun = df_p[df_p['Persen (%)'] < 0]['Kontrasepsi'].tolist()
            st.success(f"✅ Kontrasepsi yang meningkat: **{', '.join(top_naik)}** — menunjukkan pergeseran ke MKJP")
            if top_turun:
                st.warning(f"⬇️ Kontrasepsi yang menurun: **{', '.join(top_turun)}**")
        else:
            st.warning("Data tahun 2022 atau 2024 tidak ditemukan untuk analisis pergeseran.")

    with tab_detail:
        selected_kb = st.multiselect("Pilih jenis kontrasepsi:", kol_ada, default=kol_ada[:3])
        if selected_kb:
            df_det = df[['ds'] + selected_kb].copy()
            df_det['ds'] = df_det['ds'].dt.strftime('%B %Y')
            df_det.columns = ['Periode'] + selected_kb
            st.dataframe(df_det, use_container_width=True, height=360)
        else:
            st.info("Pilih minimal satu jenis kontrasepsi di atas.")