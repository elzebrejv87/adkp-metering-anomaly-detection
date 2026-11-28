import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Konfigurasi Halaman Web
st.set_page_config(
    page_title="Dashboard Deteksi Anomali Metering & Keselamatan Proses",
    page_icon="⚡",
    layout="wide"
)

# Judul & Header
st.title("⚡ Dashboard Deteksi Anomali Aliran Daya & Susut Non-Teknis")
st.markdown("""
**Sistem Metering Transmisi 150 kV — Integrasi Evaluasi Susut & Keselamatan Proses (*Process Safety*)**  
*Model Unsupervised Machine Learning: Isolation Forest | Penyusun: Farid Azhar (S2 Manajemen Keselamatan Proses UI)*
""")
st.divider()

# Sidebar untuk Pengaturan Parameter & Upload Data
st.sidebar.header("⚙️ Parameter Model & Filter")

uploaded_file = st.sidebar.file_uploader(
    "Unggah File CSV Metering Baru (Opsional)",
    type=["csv"],
    help="Jika kosong, dashboard otomatis menggunakan dataset simulasi 30 hari dari GitHub."
)

contamination = st.sidebar.slider(
    "Sensitivitas Deteksi Anomali (Contamination)",
    min_value=0.01,
    max_value=0.10,
    value=0.042,
    step=0.005,
    help="Estimasi proporsi titik anomali dalam dataset (default: 4.2%)."
)

tol_delta = st.sidebar.slider(
    "Batas Toleransi Selisih Main vs Check (%)",
    min_value=0.2,
    max_value=1.0,
    value=0.5,
    step=0.1,
    help="Mengacu kelas akurasi meter transaksi (0.2S = 0.2%, 0.5S = 0.5%)."
)

# Load Data
@st.cache_data
def load_default_data():
    url = "https://raw.githubusercontent.com/elzebrejv87/adkp-metering-anomaly-detection/main/metering_transmission_data.csv"
    return pd.read_csv(url)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = load_default_data()

df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Feature Engineering
I_avg = (df['I_R_A'] + df['I_S_A'] + df['I_T_A']) / 3
I_max_dev = np.maximum.reduce([
    np.abs(df['I_R_A'] - I_avg),
    np.abs(df['I_S_A'] - I_avg),
    np.abs(df['I_T_A'] - I_avg)
])
df['Current_Unbalance_Pct'] = (I_max_dev / I_avg) * 100
df['Energy_Delta_Pct'] = ((df['kWh_Export_Main'] - df['kWh_Export_Check']) / df['kWh_Export_Check']) * 100
df['Q_P_Ratio'] = df['kVARh_Export_Main'] / df['kWh_Export_Main']

# Pelatihan Model Isolation Forest
features = [
    'V_R_kV', 'V_S_kV', 'V_T_kV',
    'I_R_A', 'I_S_A', 'I_T_A',
    'Current_Unbalance_Pct', 'Energy_Delta_Pct', 'Q_P_Ratio'
]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

iso_forest = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
df['Anomaly_Pred'] = iso_forest.fit_predict(X_scaled)
df['Anomaly_Score'] = np.round(iso_forest.decision_function(X_scaled), 4)

# Klasifikasi Diagnostik Tindakan Lapangan
def diagnosa_tindakan(row):
    if row['Anomaly_Pred'] == 1:
        return 'Normal'
    elif abs(row['Energy_Delta_Pct']) > tol_delta and row['Q_P_Ratio'] < 0.45:
        return 'Audit Metering (Susut / Error CT-PT)'
    else:
        return 'Inspeksi Keselamatan (Thermovisi / Uji Isolasi)'

df['Status_Diagnostik'] = df.apply(diagnosa_tindakan, axis=1)

# Filter Kategori di Sidebar
kategori_pilihan = st.sidebar.multiselect(
    "Filter Status Tampilan:",
    options=df['Status_Diagnostik'].unique(),
    default=df['Status_Diagnostik'].unique()
)
df_filtered = df[df['Status_Diagnostik'].isin(kategori_pilihan)]

# Kartu Ringkasan KPI
total_data = len(df)
jml_normal = (df['Status_Diagnostik'] == 'Normal').sum()
jml_metering = (df['Status_Diagnostik'] == 'Audit Metering (Susut / Error CT-PT)').sum()
jml_safety = (df['Status_Diagnostik'] == 'Inspeksi Keselamatan (Thermovisi / Uji Isolasi)').sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Interval Terpantau", f"{total_data:,} titik", "Interval 15 Menit")
col2.metric("Operasi Normal", f"{jml_normal:,} titik", f"{(jml_normal/total_data)*100:.1f}%")
col3.metric("Anomali Metering (Susut)", f"{jml_metering} titik", "Cek Phasor & CT/PT", delta_color="inverse")
col4.metric("Anomali Keselamatan Proses", f"{jml_safety} titik", "Cek Hot-Spot / Isolasi", delta_color="inverse")

st.divider()

# Grafik 1: Profil Energi Ekspor & Titik Anomali Interaktif
st.subheader("1. Profil Energi Ekspor (kWh) & Sebaran Titik Anomali")
fig1 = px.line(
    df, x='Timestamp', y='kWh_Export_Main',
    labels={'Timestamp': 'Waktu Pencatatan', 'kWh_Export_Main': 'Energi Ekspor (kWh / 15m)'},
    color_discrete_sequence=['#4A90E2']
)
fig1.update_traces(opacity=0.6, name='Profil Beban Normal')

anom_only = df[df['Status_Diagnostik'] != 'Normal']
fig_scatter = px.scatter(
    anom_only, x='Timestamp', y='kWh_Export_Main',
    color='Status_Diagnostik',
    color_discrete_map={
        'Audit Metering (Susut / Error CT-PT)': '#FF9800',
        'Inspeksi Keselamatan (Thermovisi / Uji Isolasi)': '#E53935'
    },
    hover_data=['I_R_A', 'I_S_A', 'I_T_A', 'Current_Unbalance_Pct', 'Energy_Delta_Pct']
)
for trace in fig_scatter.data:
    fig1.add_trace(trace)

fig1.update_layout(height=420, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig1, use_container_width=True)

# Grafik 2 & 3 Berdampingan
col_g1, col_g2 = st.columns(2)

with col_g1:
    st.subheader("2. Deviasi Neraca Energi (Main vs Check %)")
    fig2 = px.line(
        df, x='Timestamp', y='Energy_Delta_Pct',
        labels={'Timestamp': 'Waktu', 'Energy_Delta_Pct': 'Delta Energi (%)'},
        color_discrete_sequence=['#FB8C00']
    )
    fig2.add_hline(y=tol_delta, line_dash="dash", line_color="gray", annotation_text=f"+{tol_delta}%")
    fig2.add_hline(y=-tol_delta, line_dash="dash", line_color="gray", annotation_text=f"-{tol_delta}%")
    fig2.update_layout(height=350)
    st.plotly_chart(fig2, use_container_width=True)

with col_g2:
    st.subheader("3. Ketidakseimbangan Arus Fasa (Unbalance %)")
    fig3 = px.line(
        df, x='Timestamp', y='Current_Unbalance_Pct',
        labels={'Timestamp': 'Waktu', 'Current_Unbalance_Pct': 'Current Unbalance (%)'},
        color_discrete_sequence=['#E53935']
    )
    fig3.update_layout(height=350)
    st.plotly_chart(fig3, use_container_width=True)

# Tabel Log Anomali & Tombol Unduh
st.subheader("4. Log Kejadian Anomali & Rekomendasi Tindak Lanjut Lapangan")
tabel_anomali = df_filtered[df_filtered['Status_Diagnostik'] != 'Normal'][[
    'Timestamp', 'Status_Diagnostik', 'I_R_A', 'I_S_A', 'I_T_A',
    'Current_Unbalance_Pct', 'Energy_Delta_Pct', 'Q_P_Ratio', 'Anomaly_Score'
]].round(2)

st.dataframe(tabel_anomali, use_container_width=True, height=300)

csv_download = tabel_anomali.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Unduh Laporan Log Anomali (.CSV)",
    data=csv_download,
    file_name="laporan_anomali_metering_safety.csv",
    mime="text/csv"
)
