# Deteksi Anomali Aliran Daya & Susut Non-Teknis pada Sistem Metering Transmisi

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elzebrejv87/adkp-metering-anomaly-detection/blob/main/Metering_Anomaly_Detection.ipynb)
[![Open Dashboard](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://metering-anomaly-detection.streamlit.app)

**Mata Kuliah:** Analisa Data dan Keselamatan Proses (ADKP)  
**Program Studi:** Magister Manajemen Keselamatan Proses — Departemen Teknik Kimia, Universitas Indonesia  
**Penyusun:** Farid Azhar  

---

## 1. Latar Belakang & Tujuan Proyek
Pada sistem penyaluran tenaga listrik tegangan tinggi (150 kV), akurasi pencatatan energi transaksi (`kWh` dan `kVARh` ekspor-impor) sangat krusial untuk menekan angka **susut energi (*transmission losses*)** sekaligus menjaga keandalan keselamatan peralatan Gardu Induk. Penyimpangan profil beban antar fasa maupun selisih antara *Main Meter* dan *Check Meter* dapat bersumber dari dua kondisi berbeda:
1. **Anomali Metering / Susut Non-Teknis:** Degradasi akurasi *Current Transformer* (CT) / *Potential Transformer* (PT), pergeseran polaritas/sudut fasa, atau longgarnya terminasi rangkaian sekunder meter.
2. **Anomali Keselamatan Proses (*Process Safety*):** Indikasi awal pemanasan lokal (*hot-spot*) akibat resistansi kontak tinggi maupun gejala awal kegagalan isolasi (*incipient insulation fault*) yang berpotensi memicu *internal arcing* dan kebakaran peralatan.

Proyek ini membangun model *unsupervised machine learning* berbasis algoritma **Isolation Forest** untuk mendeteksi anomali aliran daya secara otomatis serta memisahkan rekomendasi tindak lanjut antara audit metering dan inspeksi keselamatan peralatan.

---

## 2. Formulasi Matematis & Metode

### A. Ketidakseimbangan Arus Antar Fasa (*Current Unbalance*)
Digunakan sebagai indikator awal anomali pembebanan fasa maupun potensi *hot-spot* / gangguan isolasi:

$$\text{Current Unbalance (\%)} = \frac{\max\left(\vert{}I_R - I_{\text{avg}}\vert{}, \vert{}I_S - I_{\text{avg}}\vert{}, \vert{}I_T - I_{\text{avg}}\vert{}\right)}{I_{\text{avg}}} \times 100\%$$

dengan $I_{\text{avg}} = \frac{I_R + I_S + I_T}{3}$.

### B. Selisih Neraca Energi (*Main vs Check Meter Delta*)
Digunakan untuk memantau deviasi kelas akurasi sistem metering transaksi (mengacu batas toleransi kelas akurasi 0,2S / 0,5S):

$$\text{Energy Delta (\%)} = \frac{\text{kWh}_{\text{Main}} - \text{kWh}_{\text{Check}}}{\text{kWh}_{\text{Check}}} \times 100\%$$

### C. Rasio Daya Reaktif terhadap Daya Aktif ($Q/P$)
$$\text{Q/P Ratio} = \frac{\text{kVARh}_{\text{Main}}}{\text{kWh}_{\text{Main}}}$$

### D. Algoritma *Isolation Forest*
*Isolation Forest* mengisolasi titik data anomali dengan membentuk pohon keputusan acak (*Isolation Trees*). Titik anomali memiliki karakteristik jauh dari distribusi operasi normal sehingga membutuhkan jumlah partisi (panjang lintasan pohon $h(x)$) yang jauh lebih pendek. Skor anomali dihitung melalui:

$$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$

dengan $E(h(x))$ adalah rata-rata panjang lintasan titik $x$ di seluruh pohon, dan $c(n)$ adalah faktor normalisasi rata-rata panjang lintasan pada *Binary Search Tree*. Nilai $s(x, n)$ yang mendekati $1$ mengindikasikan anomali kuat.

---

## 3. Struktur Repository
* `Metering_Anomaly_Detection.ipynb` — Notebook Google Colab berisi pra-pemrosesan data, *feature engineering*, pelatihan model *Isolation Forest*, logika diagnostik lapangan, dan visualisasi *time-series*.
* `metering_transmission_data.csv` — Dataset simulasi profil beban 15 menitan selama 30 hari (2.880 baris data) pada sistem transaksi 150 kV (non-rahasia / sintetik berbasis fisika kelistrikan).
* `README.md` — Dokumentasi teknis model dan panduan eksekusi.

---

## 4. Cara Menjalankan Kode di Google Colab
1. Klik tombol biru **Open in Colab** di bagian paling atas halaman ini (atau buka langsung melalui [tautan Google Colab ini](https://colab.research.google.com/github/elzebrejv87/adkp-metering-anomaly-detection/blob/main/Metering_Anomaly_Detection.ipynb)).
2. Pastikan sel pemuatan data membaca dataset langsung dari repository ini:
   ```python
   url_csv = "[https://raw.githubusercontent.com/elzebrejv87/adkp-metering-anomaly-detection/main/metering_transmission_data.csv](https://raw.githubusercontent.com/elzebrejv87/adkp-metering-anomaly-detection/main/metering_transmission_data.csv)"
   df = pd.read_csv(url_csv)
