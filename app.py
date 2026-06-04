import streamlit as st
import numpy as np
import joblib
from PIL import Image

model = joblib.load('model_regresi.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Prediksi Harga Tanah Surakarta", layout="wide")
st.title("🏠 Prediksi Harga Tanah Kota Surakarta")
st.markdown("**Model Regresi Linier Berganda** dengan 9 fitur | R² = 0.7405")

st.sidebar.header("📋 Masukkan Parameter Tanah")
st.sidebar.markdown("Isi data sesuai dengan properti Anda.")

col1, col2 = st.sidebar.columns(2)
with col1:
    luas_tanah = st.number_input("Luas Tanah (m²)", min_value=10.0, max_value=2000.0, value=150.0, step=10.0)
    lebar_jalan = st.number_input("Lebar Jalan (meter)", min_value=1.0, max_value=30.0, value=6.0, step=0.5)
    luas_bangunan = st.number_input("Luas Bangunan (m²)", min_value=0.0, max_value=2000.0, value=120.0, step=10.0)
with col2:
    jumlah_lantai = st.number_input("Jumlah Lantai", min_value=1, max_value=10, value=2, step=1)
    jarak_ke_pusat = st.number_input("Jarak ke Pusat Kota (meter)", min_value=0.0, max_value=30000.0, value=2500.0, step=100.0)
    kelas_jalan = st.selectbox("Kelas Jalan", options=['Arteri', 'Kolektor', 'Lokal', 'Setapak'], 
                               help="Arteri: jalan besar 2 arah; Kolektor: penghubung antar wilayah; Lokal: lingkungan; Setapak: gang kecil")

kelas_arteri = 1 if kelas_jalan == 'Arteri' else 0
kelas_kolektor = 1 if kelas_jalan == 'Kolektor' else 0
kelas_lokal = 1 if kelas_jalan == 'Lokal' else 0
kelas_setapak = 1 if kelas_jalan == 'Setapak' else 0

if st.sidebar.button("🔍 Prediksi Harga", type="primary"):
    X_input = np.array([[luas_tanah, lebar_jalan, jarak_ke_pusat, luas_bangunan,
                         jumlah_lantai, kelas_arteri, kelas_kolektor, kelas_lokal, kelas_setapak]])
    X_scaled = scaler.transform(X_input)
    pred = model.predict(X_scaled)[0]
    st.sidebar.success(f"💰 Estimasi Harga: **Rp {pred:,.2f}**")

tab1, tab2, tab3 = st.tabs(["📖 Penjelasan Model", "📊 Visualisasi Hasil", "ℹ️ Tentang"])

with tab1:
    st.subheader("Apa itu Regresi Linier?")
    st.write("Regresi linier adalah metode statistik untuk memprediksi nilai target (harga tanah) berdasarkan satu atau lebih variabel input (fitur). Model ini mengasumsikan hubungan linear antara fitur dan target. Keunggulannya: mudah diinterpretasi.")
    
    st.subheader("Fitur yang Digunakan")
    st.markdown("- **Luas Tanah (m²)**: Semakin luas, harga cenderung naik.")
    st.markdown("- **Lebar Jalan (meter)**: Akses yang lebih lebar meningkatkan nilai.")
    st.markdown("- **Jarak ke Pusat Kota (meter)**: Semakin dekat ke pusat, harga lebih mahal.")
    st.markdown("- **Luas Bangunan (m²)**: Berpengaruh positif terhadap harga.")
    st.markdown("- **Jumlah Lantai**: Bangunan bertingkat menambah nilai.")
    st.markdown("- **Kelas Jalan**: Kategori akses jalan.")
    
    st.subheader("Penjelasan Kelas Jalan")
    st.info("- **Arteri** : Jalan utama 2 arah, volume kendaraan tinggi, biasanya terhubung ke pusat kota. **Paling strategis** → harga lebih tinggi.
- **Kolektor** : Jalan penghubung antar wilayah, lebih kecil dari arteri. **Cukup strategis**.
- **Lokal** : Jalan lingkungan pemukiman. **Kurang strategis** → harga cenderung lebih rendah.
- **Setapak** : Gang kecil, akses terbatas. **Paling rendah** pengaruhnya terhadap harga.")
    
    st.subheader("Evaluasi Model")
    st.write("- **R² (Koefisien Determinasi) : 0.7405** → Model menjelaskan 74% variasi harga tanah.")
    st.write("- **RMSE (Root Mean Square Error) : Rp 668 juta** → Rata-rata error prediksi sekitar Rp 668 juta.")
    st.write("- **MAE (Mean Absolute Error) : Rp 501 juta** → Rata-rata selisih absolut prediksi dengan aktual.")

with tab2:
    st.subheader("Visualisasi Hasil Model")
    st.image("scatter_actual_vs_pred.png", caption="Gambar 1. Aktual vs Prediksi (semakin dekat ke garis merah, semakin akurat)")
    st.image("residual_plot.png", caption="Gambar 2. Residual Plot (acak di sekitar nol → model baik)")
    st.image("coef_bar.png", caption="Gambar 3. Pengaruh fitur (biru = positif, merah = negatif)")
    st.image("correlation_heatmap.png", caption="Gambar 4. Korelasi antar variabel (nilai mendekati 1/-1 = kuat)")

with tab3:
    st.subheader("Tentang Aplikasi")
    st.write("Aplikasi ini dikembangkan untuk memprediksi harga tanah di Kota Surakarta menggunakan algoritma Regresi Linier. Data yang digunakan adalah 965 transaksi/penawaran tanah dari Kantor Pertanahan Surakarta tahun 2021-2024.")
    st.write("**Dibuat oleh:** Arjun Najah Muhammad Samhan")
    st.write("**Pembimbing:** [Nama Dosen Pembimbing 1], [Nama Dosen Pembimbing 2]")
    st.caption("© 2026 - Universitas Duta Bangsa Surakarta")
