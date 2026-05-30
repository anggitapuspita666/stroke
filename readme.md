# 🧠 Stroke Risk Analytics Dashboard

Dashboard Streamlit untuk analisis faktor risiko stroke berbasis data.

## 📁 Struktur Folder
```
stroke_dashboard/
├── app.py              ← Aplikasi utama Streamlit
├── stroke_data.csv     ← Dataset (14,044 records)
├── requirements.txt    ← Dependensi Python
└── README.md
```

## 🚀 Cara Menjalankan

### 1. Install dependensi
```bash
pip install -r requirements.txt
```

### 2. Jalankan dashboard
```bash
streamlit run app.py
```

### 3. Buka di browser
```
http://localhost:8501
```

---

## 📊 Fitur Dashboard

### Sidebar – Input Pasien
- Nama, Gender, Usia
- Riwayat Hipertensi & Heart Disease
- Status Menikah, Tipe Pekerjaan
- Tipe Tempat Tinggal (Perkotaan/Desa)
- Avg Glukosa Puasa & BMI
- Status Merokok

### Tab 1 – Faktor Risiko
Menjawab: *Faktor apa saja yang memengaruhi risiko stroke?*
- Bar chart stroke rate per faktor
- Correlation heatmap

### Tab 2 – Usia & Stroke
Menjawab: *Apakah usia memiliki hubungan terhadap kejadian stroke?*
- Distribusi usia (histogram overlay)
- Box plot usia vs stroke
- Stroke rate per kelompok usia

### Tab 3 – Hipertensi & Jantung
Menjawab: *Apakah hipertensi dan penyakit jantung meningkatkan risiko stroke?*
- Kombinasi 2x2 faktor
- Grouped bar chart
- Metrik perbandingan

### Tab 4 – BMI & Glukosa
Menjawab: *Bagaimana pengaruh BMI dan kadar glukosa terhadap stroke?*
- Box plots BMI & Glukosa
- Scatter plot BMI vs Glukosa
- Stroke rate per kategori BMI

### Tab 5 – Keseimbangan Data
Menjawab: *Apakah dataset mengalami imbalance sebelum SMOTE?*
- Visualisasi distribusi saat ini (post-SMOTE)
- Simulasi kondisi sebelum SMOTE
- Penjelasan SMOTE

### Tab 6 – Prediksi Pasien
- Skor risiko stroke (0–100)
- Breakdown kontribusi per faktor
- Tabel ringkasan pasien
- Rekomendasi personal

---

## 🛠️ Teknologi
- **Streamlit** — UI dashboard
- **Plotly** — Visualisasi interaktif
- **Pandas / NumPy** — Pengolahan data
- **Scikit-learn** — Preprocessing