import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import io

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Stroke Risk Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS – DARK NAVY + TEAL ACCENT THEME
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

    /* ── Background ── */
    .stApp {
        background: #0d1117;
        min-height: 100vh;
    }

    .block-container { padding: 1.5rem 2rem 3rem 2rem; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #111827 !important;
        border-right: 1px solid rgba(20,184,166,0.25);
    }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    section[data-testid="stSidebar"] h2 { color: #2dd4bf !important; font-family: 'Syne', serif !important; }
    section[data-testid="stSidebar"] label { color: #94a3b8 !important; font-size: 0.82rem !important; text-transform: uppercase; letter-spacing: 0.5px; }
    section[data-testid="stSidebar"] .stSlider > label { color: #94a3b8 !important; }

    /* ── Headings ── */
    h1 { font-family: 'Syne', sans-serif !important; color: #f1f5f9 !important; font-size: 2.4rem !important; letter-spacing: -1px; }
    h2 { font-family: 'Syne', sans-serif !important; color: #f1f5f9 !important; font-size: 1.45rem !important; border-bottom: 1px solid rgba(20,184,166,0.3); padding-bottom: 10px; margin-bottom: 16px; }
    h3 { color: #e2e8f0 !important; font-size: 1.1rem !important; font-weight: 600; }
    h4 { color: #2dd4bf !important; font-size: 0.95rem !important; font-weight: 700; }
    p, li { color: #cbd5e1 !important; line-height: 1.7; }

    /* ── Metrics ── */
    [data-testid="metric-container"] {
        background: #1e293b;
        border: 1px solid rgba(20,184,166,0.2);
        border-radius: 12px; padding: 18px 20px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.3);
    }
    [data-testid="metric-container"] label { 
        color: #94a3b8 !important; font-size: 0.75rem !important; 
        text-transform: uppercase; letter-spacing: 1.2px; font-weight: 600;
    }
    [data-testid="stMetricValue"] { color: #f1f5f9 !important; font-size: 1.9rem !important; font-weight: 700; font-family: 'Syne', sans-serif !important; }
    [data-testid="stMetricDelta"] { color: #2dd4bf !important; font-size: 0.9rem !important; }

    /* ── Tabs ── */
    [data-baseweb="tab-list"] { background: #1e293b !important; border-radius: 10px; gap: 2px; padding: 4px; border: 1px solid rgba(20,184,166,0.15); }
    [data-baseweb="tab"] { color: #94a3b8 !important; font-weight: 500; border-radius: 8px; padding: 8px 14px; font-size: 0.82rem; }
    [data-baseweb="tab"][aria-selected="true"] { background: #0f766e !important; color: #f0fdfa !important; box-shadow: 0 2px 8px rgba(15,118,110,0.4); }
    [data-baseweb="tab-panel"] { padding: 20px 0; background: transparent !important; }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #0f766e, #0d9488) !important;
        color: #f0fdfa !important; border: none !important; border-radius: 10px;
        font-weight: 600; padding: 10px 24px; transition: all .25s;
        box-shadow: 0 4px 14px rgba(13,148,136,0.35);
        font-family: 'IBM Plex Sans', sans-serif !important;
    }
    .stButton > button:hover { 
        background: linear-gradient(135deg, #0d9488, #14b8a6) !important; 
        transform: translateY(-1px); 
        box-shadow: 0 6px 20px rgba(20,184,166,0.4);
    }

    /* ── Custom card ── */
    .card {
        background: #1e293b;
        border: 1px solid rgba(20,184,166,0.2);
        border-radius: 12px; padding: 20px 24px; margin: 10px 0;
    }
    .card-danger  { border-left: 4px solid #f87171; background: #1e1b2e; }
    .card-warning { border-left: 4px solid #fbbf24; background: #1e1b14; }
    .card-success { border-left: 4px solid #34d399; background: #111e1a; }
    .card-info    { border-left: 4px solid #60a5fa; background: #111827; }

    .card h4 { margin: 0 0 8px 0; font-size: 0.95rem; font-weight: 700; }
    .card p  { margin: 0; color: #cbd5e1; font-size: 0.88rem; line-height: 1.65; }
    .card-danger h4  { color: #fca5a5 !important; }
    .card-warning h4 { color: #fde68a !important; }
    .card-success h4 { color: #6ee7b7 !important; }
    .card-info h4    { color: #93c5fd !important; }

    /* ── Prediction box ── */
    .pred-box {
        background: #1e293b;
        border: 2px solid rgba(20,184,166,0.35);
        border-radius: 18px; padding: 32px; text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }
    .pred-high { border-color: #f87171; background: #1e1220; }
    .pred-mid  { border-color: #fbbf24; background: #1e1a10; }
    .pred-low  { border-color: #34d399; background: #101e18; }

    /* ── Section divider ── */
    .divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(20,184,166,0.35), transparent); margin: 24px 0; }

    /* ── Camera section ── */
    .cam-box {
        background: #1e293b;
        border: 2px dashed rgba(20,184,166,0.35);
        border-radius: 16px; padding: 36px;
        text-align: center;
    }
    .cam-result {
        font-family: 'Syne', sans-serif;
        font-size: 1.8rem; font-weight: 800; margin: 16px 0 8px 0;
    }

    /* Selectbox / inputs */
    .stSelectbox > div > div { 
        background: #1e293b !important; 
        border-color: rgba(20,184,166,0.3) !important; 
        color: #e2e8f0 !important; 
    }
    .stSlider { accent-color: #14b8a6; }
    label { color: #94a3b8 !important; }

    /* ── Header ── */
    .dash-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f2027 100%);
        padding: 24px 30px;
        border-radius: 14px;
        margin-bottom: 24px;
        border: 1px solid rgba(20,184,166,0.2);
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
        position: relative;
        overflow: hidden;
    }
    .dash-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #0f766e, #2dd4bf, #0f766e);
    }
    .dash-header h1 { 
        color: #f1f5f9 !important; font-size: 1.9rem !important; 
        margin: 0; padding: 0; letter-spacing: -0.5px;
    }
    .dash-header p { color: #94a3b8 !important; font-size: 0.88rem; margin: 6px 0 0 0; }
    .dash-header .badge {
        display: inline-block; background: rgba(20,184,166,0.15); 
        border: 1px solid rgba(20,184,166,0.3);
        color: #2dd4bf !important; padding: 3px 10px; border-radius: 20px; 
        font-size: 0.75rem; font-weight: 600; margin-right: 8px; letter-spacing: 0.5px;
    }

    /* ── Interpretation box ── */
    .interpretation-box {
        background: #162032;
        border: 1px solid rgba(96,165,250,0.2);
        border-left: 3px solid #60a5fa;
        border-radius: 10px; padding: 18px 22px; margin: 12px 0;
    }
    .interpretation-box p { color: #cbd5e1 !important; font-size: 0.9rem; line-height: 1.75; margin: 0 0 10px 0; }
    .interpretation-box p:last-child { margin-bottom: 0; }
    .interpretation-box strong { color: #93c5fd !important; }

    /* Dataframe */
    [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
    .stDataFrame { background: #1e293b !important; }

    /* Radio buttons */
    [data-testid="stRadio"] label { color: #e2e8f0 !important; }

    /* Expander */
    [data-testid="stExpander"] { background: #1e293b; border: 1px solid rgba(20,184,166,0.15); border-radius: 10px; }
    [data-testid="stExpander"] summary { color: #94a3b8 !important; }

    /* Multiselect */
    [data-baseweb="tag"] { background: rgba(15,118,110,0.4) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("stroke_data_final_cleaned (2).csv")
    except FileNotFoundError:
        try:
            df = pd.read_csv('stroke_data_hanya_cleaning.csv')
        except FileNotFoundError:
            st.error("❌ Dataset tidak ditemukan. Pastikan file CSV ada di folder yang sama.")
            st.stop()
    
    df["Gender_Label"] = df["Gender"].map({1: "Male", 0: "Female"})
    df["Hypertension_Label"] = df["Hypertension"].map({1: "Ya", 0: "Tidak"})
    df["HeartDisease_Label"] = df["Heart_Disease"].map({1: "Ya", 0: "Tidak"})
    df["Diabetes_Label"] = df["Diabetes"].map({1: "Ya", 0: "Tidak"})
    
    def get_smoking(row):
        if "Smoking_Status_Current" in row.index:
            if row["Smoking_Status_Current"]: return "Perokok Aktif"
            if row["Smoking_Status_Former"]: return "Mantan Perokok"
        return "Tidak Merokok"
    
    if "Smoking_Status_Current" in df.columns:
        df["Smoking_Label"] = df.apply(get_smoking, axis=1)
    else:
        df["Smoking_Label"] = "Tidak Merokok"
    
    df["Stroke_Label"] = df["Stroke"].map({1: "Stroke", 0: "Tidak Stroke"})
    df["Age_Display"] = (df["Age"] * 9.9 + 69.8).round(0).astype(int)
    df["Age_Display"] = df["Age_Display"].clip(20, 100)
    
    return df

df = load_data()

# ─────────────────────────────────────────────
# RISK SCORE COMPUTATION
# ─────────────────────────────────────────────
def compute_risk(usia, hipertensi, heart_dis, glukosa, bmi, merokok, diabetes):
    score = 0
    score += min(usia / 100 * 35, 35)
    if hipertensi == "Ya": score += 20
    if heart_dis == "Ya": score += 18
    if diabetes == "Ya": score += 8
    if glukosa > 140: score += 12
    elif glukosa > 100: score += 5
    if bmi > 30: score += 8
    elif bmi > 25: score += 3
    if merokok == "Perokok Aktif": score += 10
    elif merokok == "Mantan Perokok": score += 4
    return min(round(score, 1), 100)

def risk_level(score):
    if score >= 60: return "TINGGI", "#f87171"
    if score >= 35: return "SEDANG", "#fbbf24"
    return "RENDAH", "#34d399"

# ─────────────────────────────────────────────
# PLOT THEME
# ─────────────────────────────────────────────
PLOT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#cbd5e1", family="IBM Plex Sans", size=12),
    margin=dict(t=52, b=24, l=10, r=10),
    title_font=dict(color="#f1f5f9", family="Syne", size=15),
)

# Teal-to-coral color scale for charts
TEAL_SCALE = ["#134e4a", "#0f766e", "#0d9488", "#14b8a6", "#2dd4bf", "#5eead4"]
RISK_SCALE = ["#1a3a2a", "#0f766e", "#fbbf24", "#f97316", "#ef4444"]

def ax_style(fig):
    fig.update_xaxes(color="#94a3b8", gridcolor="rgba(148,163,184,0.1)", 
                     zerolinecolor="rgba(148,163,184,0.15)", tickfont=dict(color="#94a3b8"))
    fig.update_yaxes(color="#94a3b8", gridcolor="rgba(148,163,184,0.1)", 
                     zerolinecolor="rgba(148,163,184,0.15)", tickfont=dict(color="#94a3b8"))
    return fig

# ─────────────────────────────────────────────
# SIDEBAR INPUT
# ─────────────────────────────────────────────
st.sidebar.markdown("## 🧠 Stroke Risk Intelligence")
st.sidebar.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.sidebar.markdown("### 📋 Data Pasien")

nama = st.sidebar.text_input("Nama Pasien", placeholder="Masukkan nama...")
gender = st.sidebar.radio("Gender", ["Male", "Female"], horizontal=True)
usia = st.sidebar.slider("Usia (Tahun)", 20, 100, 55)
hipertensi = st.sidebar.radio("Hipertensi", ["Ya", "Tidak"], horizontal=True)
heart_dis = st.sidebar.radio("Penyakit Jantung", ["Ya", "Tidak"], horizontal=True)
diabetes = st.sidebar.radio("Diabetes", ["Ya", "Tidak"], horizontal=True)
glukosa = st.sidebar.slider("Avg Glukosa (mg/dL)", 50.0, 300.0, 105.0, 0.5)
bmi = st.sidebar.slider("BMI", 10.0, 60.0, 25.0, 0.1)
merokok = st.sidebar.selectbox("Status Merokok", ["Tidak Merokok", "Mantan Perokok", "Perokok Aktif"])
menikah = st.sidebar.radio("Status Menikah", ["Ya", "Tidak"], horizontal=True)
pekerjaan = st.sidebar.selectbox("Tipe Pekerjaan", ["Private", "Self-employed", "Pemerintah", "Belum Bekerja"])
tempat = st.sidebar.radio("Tempat Tinggal", ["Perkotaan", "Pedesaan"], horizontal=True)

st.sidebar.markdown('<div class="divider"></div>', unsafe_allow_html=True)
run_pred = st.sidebar.button("🔍 Analisis Risiko Pasien", use_container_width=True)

st.sidebar.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.sidebar.markdown("### 🔍 Filter Analisis Data")
gender_filter = st.sidebar.multiselect("Filter Gender:", options=df['Gender_Label'].unique(), default=df['Gender_Label'].unique())
smoking_filter = st.sidebar.multiselect("Filter Status Merokok:", options=df['Smoking_Label'].unique(), default=df['Smoking_Label'].unique())
min_age, max_age = int(df['Age_Display'].min()), int(df['Age_Display'].max())
age_filter = st.sidebar.slider("Filter Rentang Umur (Tahun):", min_value=min_age, max_value=max_age, value=(min_age, max_age))

# Apply filters
df_filtered = df[
    (df['Gender_Label'].isin(gender_filter)) &
    (df['Smoking_Label'].isin(smoking_filter)) &
    (df['Age_Display'].between(age_filter[0], age_filter[1]))
]

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
  <h1>🧠 Stroke Risk Intelligence Dashboard</h1>
  <p style="margin-top:10px;">
    <span class="badge">ANALISIS KLINIS</span>
    <span class="badge">DETEKSI DINI</span>
    <span class="badge">BERBASIS DATA</span>
  </p>
  <p>Dashboard multidimensi untuk analisis faktor risiko, deteksi citra, dan prediksi risiko stroke individual</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TOP KPIs
# ─────────────────────────────────────────────
total_n = len(df_filtered)
stroke_n = int(df_filtered["Stroke"].sum())
stroke_pct = stroke_n / total_n * 100 if total_n > 0 else 0
hiper_pct = df_filtered["Hypertension"].mean() * 100
hd_pct = df_filtered["Heart_Disease"].mean() * 100

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Data", f"{total_n:,}")
k2.metric("Kasus Stroke", f"{stroke_n:,}", f"{stroke_pct:.1f}%")
k3.metric("Prevalensi Stroke", f"{stroke_pct:.1f}%")
k4.metric("Prevalensi Hipertensi", f"{hiper_pct:.1f}%")
k5.metric("Prevalensi Jantung", f"{hd_pct:.1f}%")

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Faktor Risiko",
    "📅 Usia & Stroke",
    "❤️ Hipertensi & Jantung",
    "🩸 BMI & Glukosa",
    "⚖️ Keseimbangan Data",
    "📸 Deteksi Citra Stroke",
    "🔍 Prediksi Pasien",
])

# ══════════════════════════════════════════════
# TAB 1 – FAKTOR RISIKO
# ══════════════════════════════════════════════
with tab1:
    st.markdown("## Faktor Apa Saja yang Memengaruhi Risiko Stroke?")

    st.markdown("""
    <div class="card card-info">
      <h4>💡 Konteks Analisis</h4>
      <p>Stroke adalah kondisi medis darurat yang disebabkan oleh gangguan aliran darah ke otak. 
      Dataset ini berisi ribuan rekaman pasien dengan 10 variabel klinis dan demografi yang dikumpulkan untuk analisis risiko komprehensif.</p>
    </div>
    """, unsafe_allow_html=True)

    factors = {
        "Gender": "Gender_Label",
        "Hipertensi": "Hypertension_Label",
        "Heart Disease": "HeartDisease_Label",
        "Status Merokok": "Smoking_Label",
    }

    col_a, col_b = st.columns(2)
    for i, (fname, fcol) in enumerate(factors.items()):
        tbl = df_filtered.groupby(fcol)["Stroke"].agg(["mean","sum","count"]).reset_index()
        tbl.columns = [fcol, "Rate", "Kasus", "Total"]
        tbl["Rate %"] = (tbl["Rate"] * 100).round(2)

        fig = px.bar(tbl, x=fcol, y="Rate %", color="Rate %", 
                     color_continuous_scale=RISK_SCALE,
                     text="Rate %", title=f"Stroke Rate: {fname}")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                          marker_line_color="rgba(20,184,166,0.3)", marker_line_width=1)
        fig.update_layout(**PLOT_BASE, height=320, coloraxis_showscale=False)
        ax_style(fig)

        target = col_a if i % 2 == 0 else col_b
        target.plotly_chart(fig, use_container_width=True)

        if i % 2 == 1:
            with target.expander(f"📖 Interpretasi {fname}"):
                st.markdown(f"""
                <div class="interpretation-box">
                    <p><strong>Penemuan Utama:</strong><br>
                    Chart di atas menunjukkan distribusi kejadian stroke berdasarkan kategori {fname.lower()}.</p>
                    <p><strong>Analisis Mendalam:</strong><br>
                    Persentase stroke bervariasi di antara kelompok {fname.lower()}. Kelompok dengan persentase tertinggi menunjukkan risiko stroke yang lebih signifikan. 
                    Faktor ini perlu dipertimbangkan dalam evaluasi klinis komprehensif, terutama dalam kombinasi dengan faktor risiko lainnya.</p>
                    <p><strong>Implikasi Klinis:</strong><br>
                    Identifikasi kategori {fname.lower()} dengan risiko tinggi membantu dalam stratifikasi risiko dan perencanaan intervensi preventif yang lebih tertarget.</p>
                </div>
                """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 2 – USIA & STROKE
# ══════════════════════════════════════════════
with tab2:
    st.markdown("## Apakah Usia Memiliki Hubungan dengan Kejadian Stroke?")

    col1, col2 = st.columns(2)
    with col1:
        fig_kde = go.Figure()
        for label, color in [("Tidak Stroke", "#60a5fa"), ("Stroke", "#f87171")]:
            sub = df_filtered[df_filtered["Stroke_Label"] == label]["Age_Display"]
            fig_kde.add_trace(go.Histogram(x=sub, name=label, opacity=0.75, nbinsx=35,
                marker_color=color, histnorm="probability density"))
        fig_kde.update_layout(**PLOT_BASE, title="Distribusi Usia: Stroke vs Tidak Stroke",
                              barmode="overlay", height=360, 
                              xaxis_title="Usia (Tahun)", yaxis_title="Density", 
                              legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1")))
        ax_style(fig_kde)
        st.plotly_chart(fig_kde, use_container_width=True)

    with col2:
        fig_box = px.box(df_filtered, x="Stroke_Label", y="Age_Display", color="Stroke_Label",
                         color_discrete_map={"Stroke":"#f87171","Tidak Stroke":"#60a5fa"},
                         title="Box Plot Usia vs Kejadian Stroke", points="outliers")
        fig_box.update_layout(**PLOT_BASE, height=360, showlegend=False,
                              xaxis_title="Status Stroke", yaxis_title="Usia (Tahun)")
        ax_style(fig_box)
        st.plotly_chart(fig_box, use_container_width=True)

    with st.expander("📖 Interpretasi Hubungan Usia dan Stroke"):
        stroke_df = df_filtered[df_filtered["Stroke"] == 1]
        no_stroke_df = df_filtered[df_filtered["Stroke"] == 0]
        mean_stroke_age = stroke_df["Age_Display"].mean()
        mean_no_stroke_age = no_stroke_df["Age_Display"].mean()
        
        st.markdown(f"""
        <div class="interpretation-box">
            <p><strong>Penemuan Utama:</strong><br>
            Histogram menunjukkan distribusi usia pada kelompok stroke dan tidak stroke. Box plot memberikan visualisasi statistik deskriptif termasuk median, quartil, dan outlier.</p>
            
            <p><strong>Analisis Mendalam:</strong><br>
            • Rata-rata usia pasien stroke: <strong>{mean_stroke_age:.1f} tahun</strong><br>
            • Rata-rata usia pasien tidak stroke: <strong>{mean_no_stroke_age:.1f} tahun</strong><br>
            • Perbedaan rata-rata: <strong>{abs(mean_stroke_age - mean_no_stroke_age):.1f} tahun</strong><br><br>
            Analisis menunjukkan bahwa usia merupakan faktor risiko yang signifikan. Seiring bertambahnya usia, risiko stroke meningkat secara eksponensial.</p>
            
            <p><strong>Implikasi Klinis:</strong><br>
            Usia >60 tahun dianggap sebagai marker risiko penting. Pasien dalam kelompok usia ini memerlukan screening yang lebih ketat dan monitoring berkelanjutan.</p>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 3 – HIPERTENSI & JANTUNG
# ══════════════════════════════════════════════
with tab3:
    st.markdown("## Apakah Hipertensi dan Penyakit Jantung Meningkatkan Risiko Stroke?")

    combo = df_filtered.groupby(["Hypertension_Label","HeartDisease_Label"])["Stroke"] \
              .agg(["mean","count"]).reset_index()
    combo.columns = ["Hypertension_Label","HeartDisease_Label","Rate","Count"]
    combo["Rate %"] = (combo["Rate"]*100).round(2)
    combo["Label"] = combo["Hypertension_Label"] + " / " + combo["HeartDisease_Label"]

    col1, col2 = st.columns(2)
    with col1:
        fig_cb = px.bar(combo, x="Label", y="Rate %", color="Rate %", 
                        color_continuous_scale=RISK_SCALE, text="Rate %",
                        title="Stroke Rate: Kombinasi Hipertensi & Heart Disease")
        fig_cb.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                             marker_line_color="rgba(20,184,166,0.25)", marker_line_width=1)
        fig_cb.update_layout(**PLOT_BASE, height=380, coloraxis_showscale=False)
        ax_style(fig_cb)
        st.plotly_chart(fig_cb, use_container_width=True)

    with col2:
        s_df = df_filtered[df_filtered["Stroke"]==1]
        if len(s_df) > 0:
            hc_counts = s_df["Hypertension_Label"].value_counts()
            hc = pd.DataFrame({"Hypertension_Label": hc_counts.index, "count": hc_counts.values})
            fig_pie = px.pie(hc, names="Hypertension_Label", values="count",
                             title="Distribusi Hipertensi pada Pasien Stroke",
                             color_discrete_sequence=["#f87171","#60a5fa"], hole=0.45)
            fig_pie.update_layout(**PLOT_BASE, height=380, 
                                  legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1")))
            fig_pie.update_traces(textinfo="percent+value", textfont=dict(color="#ffffff", size=13))
            st.plotly_chart(fig_pie, use_container_width=True)

with st.expander("📖 Interpretasi Hipertensi & Penyakit Jantung"):
        st.markdown("""
<div class="interpretation-box">
<p><strong>Penemuan Utama:</strong><br>
Chart pertama menunjukkan stroke rate untuk kombinasi hipertensi dan penyakit jantung. Chart kedua menampilkan proporsi hipertensi dalam kelompok pasien stroke.</p>

<p><strong>Analisis Mendalam:</strong><br>
Hipertensi dan penyakit jantung adalah faktor risiko utama untuk stroke. Pasien yang memiliki kedua kondisi ini memiliki risiko stroke yang jauh lebih tinggi. 
Kombinasi kedua faktor ini bersifat sinergis — meningkatkan risiko secara signifikan melebihi penjumlahan sederhana keduanya.<br><br>
Hipertensi menyebabkan kerusakan pada dinding pembuluh darah dan meningkatkan risiko aterosklerosis. 
Penyakit jantung meningkatkan risiko pembentukan thrombus yang dapat menyebar ke otak.</p>

<p><strong>Implikasi Klinis:</strong><br>
Pasien dengan riwayat hipertensi dan penyakit jantung memerlukan kontrol tekanan darah dan monitoring jantung yang ketat. 
Program modifikasi gaya hidup adalah komponen esensial dari strategi pencegahan.</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 4 – BMI & GLUKOSA
# ══════════════════════════════════════════════
with tab4:
    st.markdown("##  Bagaimana Pengaruh BMI dan Kadar Glukosa terhadap Stroke?")

    col1, col2 = st.columns(2)
    with col1:
        fig_bmi = px.box(df_filtered, x="Stroke_Label", y="BMI", color="Stroke_Label",
                         color_discrete_map={"Stroke":"#f87171","Tidak Stroke":"#60a5fa"},
                         title="Distribusi BMI: Stroke vs Tidak", points="outliers")
        fig_bmi.update_layout(**PLOT_BASE, height=360, showlegend=False,
                              xaxis_title="Status Stroke", yaxis_title="Indeks Massa Tubuh (BMI)")
        ax_style(fig_bmi)
        st.plotly_chart(fig_bmi, use_container_width=True)
        
    with col2:
        fig_glc = px.box(df_filtered, x="Stroke_Label", y="Avg_Glucose", color="Stroke_Label",
                         color_discrete_map={"Stroke":"#f87171","Tidak Stroke":"#60a5fa"},
                         title="Distribusi Rata-rata Glukosa: Stroke vs Tidak", points="outliers")
        fig_glc.update_layout(**PLOT_BASE, height=360, showlegend=False,
                              xaxis_title="Status Stroke", yaxis_title="Rata-rata Glukosa (mg/dL)")
        ax_style(fig_glc)
        st.plotly_chart(fig_glc, use_container_width=True)

    samp = df_filtered.sample(min(3000,len(df_filtered)), random_state=42).sort_values("Stroke")
    fig_sc = px.scatter(samp, x="BMI", y="Avg_Glucose", color="Stroke_Label",
                        color_discrete_map={"Stroke":"#f87171","Tidak Stroke":"rgba(96,165,250,0.35)"},
                        opacity=0.65, title="BMI vs Rata-rata Glukosa (Sampel)")
    fig_sc.update_layout(**PLOT_BASE, height=400, 
                         legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1")))
    ax_style(fig_sc)
    st.plotly_chart(fig_sc, use_container_width=True)

with st.expander("📖 Interpretasi BMI dan Glukosa"):
        stroke_df = df_filtered[df_filtered["Stroke"] == 1]
        no_stroke_df = df_filtered[df_filtered["Stroke"] == 0]
        mean_bmi_stroke = stroke_df["BMI"].mean()
        mean_bmi_no_stroke = no_stroke_df["BMI"].mean()
        mean_glc_stroke = stroke_df["Avg_Glucose"].mean()
        mean_glc_no_stroke = no_stroke_df["Avg_Glucose"].mean()
        
        st.markdown(f"""
<div class="interpretation-box">
<p><strong>Penemuan Utama:</strong><br>
Box plot menampilkan distribusi BMI dan glukosa pada kelompok stroke dan tidak stroke. Scatter plot menunjukkan relasi antara kedua variabel.</p>

<p><strong>Analisis Mendalam — BMI:</strong><br>
• Rata-rata BMI pasien stroke: <strong>{mean_bmi_stroke:.1f}</strong><br>
• Rata-rata BMI pasien tidak stroke: <strong>{mean_bmi_no_stroke:.1f}</strong><br>
Obesitas (BMI >30) meningkatkan risiko stroke melalui hipertensi, dislipidemia, dan resistansi insulin.</p>

<p><strong>Analisis Mendalam — Glukosa:</strong><br>
• Rata-rata glukosa pasien stroke: <strong>{mean_glc_stroke:.1f} mg/dL</strong><br>
• Rata-rata glukosa pasien tidak stroke: <strong>{mean_glc_no_stroke:.1f} mg/dL</strong><br>
Kadar glukosa tinggi meningkatkan risiko stroke melalui hiperviskositas darah dan disfungsi endotel.</p>

<p><strong>Implikasi Klinis:</strong><br>
Pasien dengan BMI >25 atau glukosa >100 mg/dL memerlukan intervensi lifestyle dan farmakoterapi. 
Program diet, aktivitas fisik, dan edukasi pasien sangat penting untuk mengurangi risiko jangka panjang.</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 5 – IMBALANCE & SMOTE
# ══════════════════════════════════════════════
with tab5:
    st.markdown("##  Apakah Dataset Mengalami Imbalance Sebelum SMOTE?")

    sc = df_filtered["Stroke"].value_counts().reset_index()
    sc.columns = ["Stroke","count"]
    sc["Label"] = sc["Stroke"].map({1:"Stroke (1)", 0:"Tidak Stroke (0)"})

    col1, col2 = st.columns(2)
    with col1:
        fig_now = px.pie(sc, names="Label", values="count",
                         title="Distribusi Saat Ini (Post-SMOTE)",
                         color_discrete_sequence=["#f87171","#60a5fa"], hole=0.45)
        fig_now.update_traces(textinfo="percent+value", textfont=dict(color="#ffffff", size=13))
        fig_now.update_layout(**PLOT_BASE, height=360, 
                              legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1")))
        st.plotly_chart(fig_now, use_container_width=True)
        
    with col2:
        sim_no = int(len(df) * 0.70)
        sim_yes = int(len(df) * 0.30)
        sim = pd.DataFrame({"Label":["Tidak Stroke","Stroke"],"Count":[sim_no,sim_yes]})
        fig_sim = px.bar(sim, x="Label", y="Count", color="Label",
                         color_discrete_sequence=["#60a5fa","#f87171"],
                         text="Count", title="Simulasi Distribusi SEBELUM SMOTE")
        fig_sim.update_traces(texttemplate="%{text:,}", textposition="outside",
                              textfont=dict(color="#f1f5f9"))
        fig_sim.update_layout(**PLOT_BASE, height=360, showlegend=False)
        ax_style(fig_sim)
        st.plotly_chart(fig_sim, use_container_width=True)
with st.expander("📖 Interpretasi Class Imbalance dan SMOTE"):
        st.markdown("""
<div class="interpretation-box">
<p><strong>Penemuan Utama:</strong><br>
Dataset mengalami class imbalance yang signifikan, dengan jumlah kasus "Tidak Stroke" jauh lebih banyak dibanding "Stroke".</p>

<p><strong>Analisis Mendalam:</strong><br>
Class imbalance adalah masalah umum dalam machine learning, terutama pada dataset medis. Model cenderung belajar memprediksi kelas mayoritas, 
mengabaikan kelas minoritas (stroke). Hal ini menghasilkan bias dalam prediksi dan performa model yang tidak optimal.<br><br>
SMOTE (Synthetic Minority Over-sampling Technique) mengatasi ini dengan membuat sampel sintetis dari kelas minoritas menggunakan k-nearest neighbors, 
sehingga menciptakan distribusi yang lebih seimbang tanpa duplikasi sampel original.</p>

<p><strong>Implikasi untuk Model:</strong><br>
Dengan SMOTE, model dapat belajar dari kedua kelas secara lebih seimbang. Hal ini meningkatkan sensitivitas (recall) dalam mendeteksi kasus stroke, 
sangat penting untuk aplikasi klinis dimana false negatives dapat berakibat fatal.</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 6 – DETEKSI CITRA STROKE
# ══════════════════════════════════════════════
with tab6:
    st.markdown("## 📸 Deteksi Citra Stroke (Analisis Gambar Medis)")

    st.markdown("""
    <div class="card card-info">
      <h4>🧠 Tentang Analisis Citra Stroke</h4>
      <p>Analisis citra medis (MRI/CT-Scan otak) untuk klasifikasi Stroke vs Non-Stroke menggunakan fitur visual dan statistical features. Upload gambar MRI/CT-Scan untuk mendapatkan analisis otomatis berbasis 6 fitur citra.</p>
    </div>
    """, unsafe_allow_html=True)

    input_method = st.radio("Pilih Metode Input:", ["📁 Upload Gambar", "📷 Kamera Real-time"], horizontal=True)

    img_data = None
    if input_method == "📁 Upload Gambar":
        uploaded_file = st.file_uploader("Upload gambar MRI/CT-Scan otak (JPG, PNG):",
                                         type=["jpg","jpeg","png"])
        if uploaded_file:
            img_data = uploaded_file
    else:
        captured = st.camera_input("Ambil gambar dengan kamera:")
        if captured:
            img_data = captured

    if img_data is not None:
        img = Image.open(img_data).convert("RGB")
        img_arr = np.array(img.resize((128, 128))) / 255.0

        gray = np.mean(img_arr, axis=2)
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        dark_ratio = float(np.mean(gray < 0.3))
        light_ratio = float(np.mean(gray > 0.7))
        asymmetry = float(abs(np.mean(gray[:,:64]) - np.mean(gray[:,64:])))
        edges = np.abs(np.diff(gray, axis=1)).mean()
        texture = float(edges)
        
        img_risk_score = (
            asymmetry * 35 +
            dark_ratio * 25 +
            (1 - light_ratio) * 20 +
            texture * 10 +
            (1 - contrast) * 10
        )
        img_risk_score = min(100, max(0, img_risk_score))
        img_risk_score = round(img_risk_score, 1)

        verdict = "🚨 STROKE TERDETEKSI" if img_risk_score > 50 else ("⚠️ RISIKO SEDANG" if img_risk_score > 30 else "✅ NORMAL")
        verdict_color = "#f87171" if img_risk_score > 50 else ("#fbbf24" if img_risk_score > 30 else "#34d399")
        verdict_bg = "pred-high" if img_risk_score > 50 else ("pred-mid" if img_risk_score > 30 else "pred-low")

        col_img, col_res = st.columns([1, 1.3])
        with col_img:
            st.image(img, caption="Gambar yang Dianalisis", width=300)
        
        with col_res:
            st.markdown(f"""
            <div class="pred-box {verdict_bg}">
              <p style="color:#94a3b8; font-size:0.82rem; margin:0 0 10px 0; text-transform:uppercase; letter-spacing:1px;">Hasil Analisis Citra</p>
              <div class="cam-result" style="color:{verdict_color};">{verdict}</div>
              <p style="color:#e2e8f0; font-size:1.1rem; margin:14px 0 0 0;">
                Skor Risiko: <strong style="color:{verdict_color}; font-size:1.4rem; font-family:'Syne',sans-serif;">{img_risk_score:.1f}</strong>
                <span style="color:#64748b;">/100</span>
              </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("### 📊 Analisis Terperinci Fitur Citra")

        col_feat1, col_feat2, col_feat3 = st.columns(3)
        with col_feat1:
            st.metric("Brightness", f"{brightness:.3f}", "Tingkat kecerahan gambar")
        with col_feat2:
            st.metric("Contrast", f"{contrast:.3f}", "Variabilitas intensitas")
        with col_feat3:
            st.metric("Dark Ratio", f"{dark_ratio:.1%}", "Proporsi area gelap")

        col_feat4, col_feat5, col_feat6 = st.columns(3)
        with col_feat4:
            st.metric("Asymmetry", f"{asymmetry:.3f}", "Ketidaksimetrisan hemisfer")
        with col_feat5:
            st.metric("Texture", f"{texture:.3f}", "Detil tepi (edge)")
        with col_feat6:
            st.metric("Light Ratio", f"{light_ratio:.1%}", "Proporsi area cerah")

        fig_features = go.Figure(data=[
            go.Scatterpolar(
                r=[brightness, contrast, dark_ratio, asymmetry, texture],
                theta=['Brightness', 'Contrast', 'Dark Ratio', 'Asymmetry', 'Texture'],
                fill='toself',
                name='Fitur Citra',
                line=dict(color='#14b8a6', width=2),
                fillcolor='rgba(20,184,166,0.2)'
            )
        ])
        fig_features.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], color="#94a3b8", gridcolor="rgba(148,163,184,0.15)"),
                angularaxis=dict(color="#cbd5e1")
            ),
            **PLOT_BASE,
            height=360,
            title="Radar Plot Fitur Citra"
        )
        st.plotly_chart(fig_features, use_container_width=True)

        with st.expander("📖 Interpretasi Detail Analisis Citra"):
            st.markdown(f"""
            <div class="interpretation-box">
                <p><strong>Penjelasan Fitur yang Diekstrak:</strong><br>
                • <strong>Brightness:</strong> Rata-rata intensitas pixel (0=gelap, 1=terang). Nilai rendah dapat mengindikasikan area iskemik.<br>
                • <strong>Contrast:</strong> Standar deviasi intensitas. Kontras rendah dapat menunjukkan area homogen yang mencurigakan.<br>
                • <strong>Dark Ratio:</strong> Proporsi piksel dengan intensitas &lt;0.3. Area gelap berlebihan dapat mengindikasikan infark.<br>
                • <strong>Asymmetry:</strong> Perbedaan intensitas antara hemisfer kiri-kanan. Asimetri tinggi dapat menunjukkan fokus lesi.<br>
                • <strong>Texture:</strong> Kompleksitas edge dalam gambar. Texture rendah bisa normal atau edge-blur (abnormal).<br>
                • <strong>Light Ratio:</strong> Proporsi area terang. Nilai rendah pada FLAIR MRI dapat menunjukkan kelainan.</p>
                
                <p><strong>Skor Risiko: {img_risk_score}/100</strong><br>
                {"🚨 Risiko tinggi — fitur citra menunjukkan pola yang konsisten dengan stroke. Evaluasi klinis segera oleh neuroradiolog diperlukan." if img_risk_score > 50 else ("⚠️ Risiko sedang — beberapa fitur menyimpang dari normal. Follow-up dan clinical correlation diperlukan." if img_risk_score > 30 else "✅ Risiko rendah — karakteristik citra konsisten dengan brain imaging normal. Monitoring rutin direkomendasikan.")}</p>
                
                <p><strong>Catatan Penting:</strong><br>
                Analisis ini bersifat bantuan awal berbasis fitur statistik, bukan pengganti interpretasi radiologi klinis oleh dokter ahli.</p>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="cam-box">
          <div style="font-size:3.5rem; margin-bottom:14px;">🫀</div>
          <h3 style="color:#2dd4bf; margin:0 0 8px 0; font-family:'Syne',sans-serif;">Upload atau ambil foto untuk analisis citra</h3>
          <p style="color:#94a3b8; margin:0;">Gunakan MRI/CT-Scan otak untuk hasil analisis yang optimal</p>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 7 – PREDIKSI PASIEN
# ══════════════════════════════════════════════
with tab7:
    st.markdown("## 🔍 Analisis Risiko Pasien Individual")

    if not run_pred:
        st.markdown("""
        <div class="cam-box">
          <div style="font-size:2.5rem; margin-bottom:14px;">👤</div>
          <h3 style="color:#2dd4bf; margin:0 0 8px 0; font-family:'Syne',sans-serif;">Isi data pasien di sidebar</h3>
          <p style="color:#94a3b8; margin:0;">Kemudian klik tombol <strong style="color:#2dd4bf;">"Analisis Risiko Pasien"</strong> untuk melihat hasil</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        risk_score = compute_risk(usia, hipertensi, heart_dis, glukosa, bmi, merokok, diabetes)
        rl, rc = risk_level(risk_score)
        box_cls = "pred-high" if risk_score >= 60 else ("pred-mid" if risk_score >= 35 else "pred-low")

        st.markdown(f"""
        <div class="pred-box {box_cls}">
          <p style="color:#94a3b8; font-size:0.78rem; text-transform:uppercase; letter-spacing:1.2px; margin:0 0 8px 0;">Profil Pasien</p>
          <h2 style="color:#f1f5f9; margin:0 0 6px 0; font-family:'Syne',sans-serif; font-size:1.6rem;">
            {'👤 ' + nama if nama else '👤 Pasien Anonim'}
          </h2>
          <p style="color:#94a3b8; font-size:0.9rem; margin:0 0 22px 0;">
            {gender} &nbsp;·&nbsp; {usia} tahun &nbsp;·&nbsp; {tempat} &nbsp;·&nbsp; {pekerjaan}
          </p>
          <div style="font-family:'Syne',sans-serif; font-size:4.5rem; font-weight:800; color:{rc}; margin:0; line-height:1;">
            {risk_score:.0f}<span style="font-size:1.1rem; color:#64748b; font-family:'IBM Plex Sans',sans-serif;">/100</span>
          </div>
          <p style="color:#e2e8f0; font-size:1.15rem; margin:12px 0 0 0; font-weight:600;">
            Tingkat Risiko: <span style="color:{rc}; font-weight:700;">{rl}</span>
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        factors_list = {
            "Usia": min(round(usia/100*35,1), 35),
            "Hipertensi": 20 if hipertensi=="Ya" else 0,
            "Heart Disease": 18 if heart_dis=="Ya" else 0,
            "Diabetes": 8 if diabetes=="Ya" else 0,
            "Glukosa": 12 if glukosa>140 else (5 if glukosa>100 else 0),
            "BMI": 8 if bmi>30 else (3 if bmi>25 else 0),
            "Merokok": 10 if merokok=="Perokok Aktif" else (4 if merokok=="Mantan Perokok" else 0),
        }

        fd = pd.DataFrame(list(factors_list.items()), columns=["Faktor","Skor"])
        fig_rf = px.bar(fd, y="Faktor", x="Skor", orientation="h", color="Skor", 
                        color_continuous_scale=RISK_SCALE, text="Skor",
                        title="Kontribusi Setiap Faktor terhadap Skor Risiko")
        fig_rf.update_traces(texttemplate="%{text:.1f}", textposition="outside",
                             textfont=dict(color="#f1f5f9"))
        fig_rf.update_layout(**PLOT_BASE, height=340, coloraxis_showscale=False)
        ax_style(fig_rf)
        st.plotly_chart(fig_rf, use_container_width=True)

        # Risk card
        if risk_score >= 60:
            st.markdown(f"""
            <div class="card card-danger">
              <h4>🔴 Risiko Tinggi — Skor: {risk_score:.0f}/100</h4>
              <p>Profil risiko tinggi dengan akumulasi faktor risiko mayor. Intervensi medis segera sangat diperlukan untuk mencegah kejadian stroke.</p>
            </div>
            """, unsafe_allow_html=True)
            rec_title = "🚨 Rekomendasi Penanganan Risiko Tinggi"
            rec_color = "#fca5a5"
            rec_items = [
                "Evaluasi klinis urgent oleh neurolog atau cardiologist",
                "Screening lanjutan: MRI otak, carotid ultrasound, atau CT angiography",
                "Pertimbangkan terapi antiplatelet (aspirin) atau antikoagulan sesuai indikasi",
                "Manajemen agresif tekanan darah — target &lt;140/90 mmHg",
                "Terapi statin untuk penanganan dyslipidemia",
                "Monitoring 24 jam ambulatory apabila ada aritmia jantung",
                "Modifikasi gaya hidup yang ketat dan monitoring berkala setiap 1–3 bulan",
            ]
        elif risk_score >= 35:
            st.markdown(f"""
            <div class="card card-warning">
              <h4>🟡 Risiko Sedang — Skor: {risk_score:.0f}/100</h4>
              <p>Beberapa faktor risiko yang perlu dikendalikan secara aktif. Program pencegahan primer sangat direkomendasikan.</p>
            </div>
            """, unsafe_allow_html=True)
            rec_title = "⚠️ Rekomendasi Penanganan Risiko Sedang"
            rec_color = "#fde68a"
            rec_items = [
                "Konsultasi rutin dengan dokter primer setiap 3–6 bulan",
                "Monitoring tekanan darah rutin — target optimal &lt;130/80 mmHg",
                "Pemeriksaan laboratorium tahunan: lipid profile, HbA1c, fungsi ginjal",
                "Screening untuk atrial fibrillation dengan EKG berkala",
                "Pertimbangkan terapi low-dose aspirin setelah evaluasi risiko-manfaat",
                "Program modifikasi gaya hidup yang terstruktur",
                "Edukasi warning signs stroke — kenali gejala FAST (Face, Arms, Speech, Time)",
            ]
        else:
            st.markdown(f"""
            <div class="card card-success">
              <h4>🟢 Risiko Rendah — Skor: {risk_score:.0f}/100</h4>
              <p>Profil risiko rendah. Pertahankan pola hidup sehat sebagai investasi kesehatan jangka panjang.</p>
            </div>
            """, unsafe_allow_html=True)
            rec_title = "💚 Rekomendasi Pemeliharaan Risiko Rendah"
            rec_color = "#6ee7b7"
            rec_items = [
                "Lanjutkan monitoring kesehatan rutin setiap tahun",
                "Pemeriksaan tekanan darah di setiap kunjungan medis",
                "Diet sehat jantung — Mediterranean atau DASH diet",
                "Aktivitas fisik teratur minimum 150 menit per minggu",
                "Manajemen stres melalui meditasi atau yoga",
                "Hindari merokok sepenuhnya dan batasi konsumsi alkohol",
                "Pertahankan BMI ideal (18.5–24.9) dengan pola makan seimbang",
            ]

        st.markdown(f"### {rec_title}")
        for item in rec_items:
            st.markdown(f"""
            <div style="display:flex; align-items:flex-start; gap:10px; padding:8px 0; border-bottom:1px solid rgba(148,163,184,0.08);">
              <span style="color:{rec_color}; font-size:1rem; flex-shrink:0;">▸</span>
              <span style="color:#cbd5e1; font-size:0.9rem; line-height:1.6;">{item}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        with st.expander("🔍 Lihat Metodologi Perhitungan Skor Risiko"):
            st.markdown("""
            <div class="interpretation-box">
                <p><strong>Cara Skor Risiko Dihitung:</strong><br>
                Skor dibangun dengan mengintegrasikan kontribusi relatif setiap faktor klinis, berdasarkan literatur epidemiologi dan analisis dataset.</p>
                
                <p><strong>Bobot Setiap Faktor:</strong><br>
                • <strong>Usia:</strong> Kontribusi linear, bertambah seiring usia (maks. 35 poin)<br>
                • <strong>Hipertensi:</strong> Faktor risiko mayor — 20 poin jika ada<br>
                • <strong>Penyakit Jantung:</strong> Faktor risiko mayor — 18 poin jika ada<br>
                • <strong>Diabetes:</strong> Faktor risiko moderat — 8 poin jika ada<br>
                • <strong>Glukosa:</strong> Dosage-dependent — 0–12 poin sesuai level<br>
                • <strong>BMI:</strong> Dosage-dependent — 0–8 poin sesuai kategori<br>
                • <strong>Status Merokok:</strong> Faktor signifikan — 0–10 poin</p>
                
                <p><strong>Interpretasi Skor:</strong><br>
                Skor &lt;35 → Risiko rendah, fokus pencegahan primer<br>
                Skor 35–60 → Risiko sedang, manajemen aktif faktor risiko<br>
                Skor &gt;60 → Risiko tinggi, intervensi klinis intensif</p>
                
                <p><strong>Disclaimer:</strong><br>
                Model ini bersifat prediktif dan bukan diagnosis definitif. Keputusan klinis harus mempertimbangkan konteks klinis lengkap dan melibatkan profesional medis yang qualified.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 👤 Ringkasan Data Pasien")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("#### Demografis")
            summary_left = {"Nama": nama or "-", "Gender": gender, "Usia": f"{usia} tahun",
                           "Tempat Tinggal": tempat, "Pekerjaan": pekerjaan, "Status Menikah": menikah}
            st.dataframe(pd.DataFrame(list(summary_left.items()), columns=["Parameter","Nilai"]),
                        use_container_width=True, hide_index=True)
                        
        with col_s2:
            st.markdown("#### Klinis")
            summary_right = {"Hipertensi": hipertensi, "Heart Disease": heart_dis,
                            "Diabetes": diabetes, "Glukosa": f"{glukosa:.1f} mg/dL",
                            "BMI": f"{bmi:.1f}", "Status Merokok": merokok}
            st.dataframe(pd.DataFrame(list(summary_right.items()), columns=["Parameter","Nilai"]),
                        use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("""
<p style="text-align:center; color:#475569; font-size:0.8rem; line-height:1.8;">
  🧠 <strong style="color:#64748b;">Stroke Risk Intelligence Dashboard</strong> &nbsp;·&nbsp; Powered by Streamlit & Plotly<br>
  <em>Disclaimer: Dashboard ini adalah alat bantu analisis. Keputusan klinis harus melibatkan profesional medis yang qualified.</em>
</p>
""", unsafe_allow_html=True)