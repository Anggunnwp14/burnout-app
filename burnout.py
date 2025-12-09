import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os
import base64
from streamlit_option_menu import option_menu 

# ==========================================
# 1. KONFIGURASI HALAMAN & CSS
# ==========================================
st.set_page_config(
    page_title="Sistem Analisis Burnout",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Custom untuk Tampilan Cantik
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
    
    .home-title { text-align: center; font-size: 2.5rem; font-weight: 700; color: #1E293B; margin-bottom: 5px; }
    .home-subtitle { text-align: center; font-size: 1.1rem; color: #64748B; margin-bottom: 40px; }
    
    .member-card {
        background-color: white; border-radius: 15px; padding: 20px;
        text-align: center; box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        transition: transform 0.3s ease; border: 1px solid #E2E8F0;
    }
    .member-card:hover { transform: translateY(-5px); box-shadow: 0 10px 20px rgba(0,0,0,0.1); border-color: #3B82F6; }
    
    .member-avatar {
        width: 120px; height: 120px; border-radius: 50%; object-fit: cover;
        margin-bottom: 15px; border: 3px solid #3B82F6; padding: 2px;
    }
    .stButton>button { width: 100%; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. INISIALISASI SESSION STATE
# ==========================================
if 'df' not in st.session_state: st.session_state['df'] = None
if 'model' not in st.session_state: st.session_state['model'] = None
if 'features' not in st.session_state: st.session_state['features'] = []
if 'target' not in st.session_state: st.session_state['target'] = None
if 'le_dict' not in st.session_state: st.session_state['le_dict'] = {}

# ==========================================
# 3. FUNGSI BANTUAN
# ==========================================
def get_img_as_base64(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return ""

def generate_dummy_data():
    np.random.seed(42)
    n = 200
    data = {
        'Jam_Belajar': np.random.randint(2, 14, n),
        'Jam_Tidur': np.random.randint(3, 9, n),
        'Ikut_Organisasi': np.random.choice(['Ya', 'Tidak'], n),
        'Tekanan_Tugas': np.random.choice(['Ringan', 'Sedang', 'Berat'], n)
    }
    return pd.DataFrame(data)

# ==========================================
# 4. SIDEBAR NAVIGASI
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3437/3437364.png", width=70)
    st.markdown("<h3>Analisis Burnout</h3>", unsafe_allow_html=True)
    
    selected = option_menu(
        menu_title=None,
        options=["Beranda", "Input Data", "Preprocessing", "Analisis Model", "Visualisasi"],
        icons=["house", "cloud-upload", "gear", "cpu", "bar-chart"], 
        menu_icon="cast", 
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "nav-link-selected": {"background-color": "#3B82F6"},
        }
    )
    st.caption("© 2025 Kelompok 3")

# ==========================================
# 5. KONTEN UTAMA
# ==========================================

# --- HALAMAN 1: BERANDA ---
if selected == "Beranda":
    st.markdown('<div class="home-title">Analisis Risiko Burnout pada Mahasiswa</div>', unsafe_allow_html=True)
    st.markdown('<div class="home-subtitle">Menggunakan metode Decision Tree Classification</div>', unsafe_allow_html=True)

    st.info("web app ini digunakan untuk memprediksi tingkat risiko burnout pada mahasiswa dengan metode Decision Tree melalui analisis faktor-faktor seperti beban kuliah, waktu istirahat, keaktifan organisasi, dan dampak aktivitas harian.")

    st.markdown("### Anggota Kelompok 3")
    col1, col2, col3 = st.columns(3)
    
    # Ganti 'fotosalsa.jpg' dll dengan nama file foto asli Anda jika ada
    def_av = "https://cdn-icons-png.flaticon.com/512/4140/4140048.png"
    img1 = f"data:image/jpg;base64,{get_img_as_base64('fotosalsa.jpg')}" if get_img_as_base64('fotosalsa.jpg') else def_av
    img2 = f"data:image/jpg;base64,{get_img_as_base64('fotoupa.jpg')}" if get_img_as_base64('fotoupa.jpg') else def_av
    img3 = f"data:image/jpg;base64,{get_img_as_base64('fotoanggun1.jpg')}" if get_img_as_base64('fotoanggun1.jpg') else def_av

    with col1:
        st.markdown(f"""<div class="member-card"><img src="{img1}" class="member-avatar"><h3>Salsabilla</h3><p>NIM: 2311522020</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="member-card"><img src="{img2}" class="member-avatar"><h3>Alya Zulhanifa</h3><p>NIM: 2311523028</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="member-card"><img src="{img3}" class="member-avatar"><h3>Anggun Weldiana Putri</h3><p>NIM: 2311523040</p></div>""", unsafe_allow_html=True)

# --- HALAMAN 2: INPUT DATA (PERBAIKAN UTAMA DISINI) ---
elif selected == "Input Data":
    st.title("Input Data")
    c1, c2 = st.columns([1, 2])
    with c1:
        opt = st.radio("Sumber Data:", ["Upload CSV", "Data Dummy"])
        if opt == "Upload CSV":
            uploaded_file = st.file_uploader("Upload File CSV", type=["csv"])
            
            if uploaded_file:
                try:
                    # FIX: Gunakan engine='python' dan sep=None agar otomatis mendeteksi koma atau titik koma
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, sep=None, engine='python')
                    
                    # FIX: Bersihkan nama kolom dari spasi berlebih (misal: " Nama " jadi "Nama")
                    df.columns = df.columns.str.strip()
                    
                    st.session_state['df'] = df
                    st.success(f"✅ CSV Berhasil Dimuat! (Ditemukan {len(df.columns)} kolom)")
                except Exception as e:
                    st.error(f"Gagal membaca file: {e}")
                    
        else:
            if st.button("Buat Data Dummy"):
                st.session_state['df'] = generate_dummy_data()
                st.success("✅ Data Dummy Berhasil Dibuat!")
                
    with c2:
        if st.session_state['df'] is not None:
            st.write("Preview Data:")
            st.dataframe(st.session_state['df'].head(), use_container_width=True)

# --- HALAMAN 3: PREPROCESSING ---
elif selected == "Preprocessing":
    st.title("Preprocessing & Kalkulasi")
    
    if st.session_state['df'] is None:
        st.warning("Data belum ada. Silakan Input Data dulu.")
    else:
        df = st.session_state['df'].copy()
        
        tab_clean, tab_calc, tab_encode = st.tabs(["1. Pembersihan", "2. Hitung Label Burnout", "3. Encoding (Angka)"])
        
        # TAB 1: DATA CLEANING
        with tab_clean:
            st.subheader("Cek Data Kosong")
            if df.isnull().sum().sum() > 0:
                st.write(df.isnull().sum())
                if st.button("Hapus Baris Kosong"):
                    st.session_state['df'] = df.dropna()
                    st.rerun()
            else:
                st.success("Data Bersih dari nilai kosong.")

        # TAB 2: KALKULATOR RISIKO
        with tab_calc:
            st.subheader("Kalkulator Risiko Burnout")
            st.info("Pilih kolom yang paling sesuai dengan indikator di bawah ini.")
            
            # Dropdown dinamis
            cols = df.columns.tolist()
            c1, c2 = st.columns(2)
            with c1:
                # Coba cari kolom yang mirip namanya secara otomatis
                idx_belajar = next((i for i, c in enumerate(cols) if 'belajar' in c.lower() or 'kuliah' in c.lower()), 0)
                idx_tidur = next((i for i, c in enumerate(cols) if 'tidur' in c.lower() or 'istirahat' in c.lower()), 1 if len(cols)>1 else 0)
                
                col_belajar = st.selectbox("Indikator Beban/Belajar (Angka)", cols, index=idx_belajar)
                col_tidur = st.selectbox("Indikator Tidur (Angka/Jam)", cols, index=idx_tidur)
            with c2:
                idx_org = next((i for i, c in enumerate(cols) if 'organisasi' in c.lower()), 2 if len(cols)>2 else 0)
                idx_tekanan = next((i for i, c in enumerate(cols) if 'tekanan' in c.lower() or 'tugas' in c.lower() or 'stres' in c.lower()), 3 if len(cols)>3 else 0)
                
                col_org = st.selectbox("Indikator Organisasi (Ya/Tidak)", cols, index=idx_org)
                col_tekanan = st.selectbox("Indikator Tekanan/Stres (Teks)", cols, index=idx_tekanan)

            if st.button("🚀 Hitung Total Risiko Burnout"):
                def hitung_skor_total(row):
                    skor = 0
                    # 1. Faktor Belajar/Beban (Jika nilai tinggi = burnout tinggi)
                    try:
                        v = float(row[col_belajar])
                        # Asumsi: Jika input jam (>7), atau skala 1-5 (>3)
                        if v > 10: skor += 2        # Untuk data Jam
                        elif v > 7: skor += 1       # Untuk data Jam
                        elif v >= 4: skor += 2      # Untuk Skala 1-5 (4,5)
                        elif v >= 3: skor += 1      # Untuk Skala 1-5 (3)
                    except: pass 

                    # 2. Faktor Tidur (Jika nilai rendah = burnout tinggi)
                    try:
                        # Cek jika data berupa teks (misal: "Buruk")
                        val_str = str(row[col_tidur]).lower()
                        if 'buruk' in val_str: skor += 2
                        elif 'sedang' in val_str or 'cukup' in val_str: skor += 1
                        else:
                            # Jika data berupa angka jam
                            v = float(row[col_tidur])
                            if v < 5: skor += 2
                            elif v < 7: skor += 1
                    except: pass

                    # 3. Organisasi
                    if 'ya' in str(row[col_org]).lower(): skor += 1

                    # 4. Tekanan
                    tek = str(row[col_tekanan]).lower()
                    if 'berat' in tek or 'tinggi' in tek: skor += 2
                    elif 'sedang' in tek: skor += 1
                    
                    if skor >= 5: return "Tinggi"
                    elif skor >= 3: return "Sedang"
                    else: return "Rendah"

                df['Risiko_Burnout_Final'] = df.apply(hitung_skor_total, axis=1)
                st.session_state['df'] = df
                st.success("Label 'Risiko_Burnout_Final' berhasil dibuat!")
                st.dataframe(df.head(), use_container_width=True)

        # TAB 3: ENCODING
        with tab_encode:
            st.subheader("Ubah Teks Menjadi Angka")
            cat_cols = df.select_dtypes(include=['object']).columns
            st.write(f"Kolom Teks terdeteksi: {list(cat_cols)}")
            
            if st.button("Lakukan Encoding"):
                le_dict = {}
                for col in cat_cols:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    le_dict[col] = le
                
                st.session_state['df'] = df
                st.session_state['le_dict'] = le_dict
                st.success("Encoding Selesai!")
                st.dataframe(df.head(), use_container_width=True)

# --- HALAMAN 4: TRAINING MODEL ---
elif selected == "Analisis Model":
    st.title("Training Model")
    
    if st.session_state['df'] is not None:
        df = st.session_state['df']
        if df.select_dtypes(include=['object']).shape[1] > 0:
            st.error("Masih ada kolom Teks! Silakan kembali ke Preprocessing -> Tab 3 Encoding.")
        else:
            c1, c2 = st.columns(2)
            with c1: 
                # Coba cari 'Risiko_Burnout_Final', kalau tidak ada pakai kolom terakhir
                target_options = list(df.columns)
                default_idx = target_options.index('Risiko_Burnout_Final') if 'Risiko_Burnout_Final' in target_options else len(target_options)-1
                target = st.selectbox("Target (Label)", target_options, index=default_idx)
            with c2: 
                # Fitur adalah semua kolom KECUALI target
                feature_options = [c for c in df.columns if c != target]
                feats = st.multiselect("Fitur (Kriteria)", feature_options, default=feature_options)
            
            if st.button("Mulai Training Model", type="primary"):
                if not feats:
                    st.error("Pilih minimal 1 fitur!")
                else:
                    try:
                        X, y = df[feats], df[target]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        
                        clf = DecisionTreeClassifier(max_depth=4, random_state=42)
                        clf.fit(X_train, y_train)
                        
                        st.session_state.update({'model': clf, 'features': feats, 'target': target, 'X_test': X_test, 'y_test': y_test})
                        
                        acc = accuracy_score(y_test, clf.predict(X_test))
                        st.success(f"Model Berhasil Dilatih! Akurasi: {acc:.1%}")
                    except Exception as e:
                        st.error(f"Error Training: {e}")
    else:
        st.warning("Data kosong.")

# --- HALAMAN 5: VISUALISASI ---
elif selected == "Visualisasi":
    st.title("Visualisasi Hasil")
    
    if st.session_state['model']:
        clf = st.session_state['model']
        target_col = st.session_state['target']
        feats = st.session_state['features']
        
        class_names = [str(c) for c in clf.classes_]
        if target_col in st.session_state['le_dict']:
            try:
                class_names = [str(c) for c in st.session_state['le_dict'][target_col].inverse_transform(clf.classes_)]
            except: pass
            
        t1, t2, t3, t4 = st.tabs(["Pohon Keputusan", "Logika (Rules)", "Feature Importance", "Prediksi"])
        
        with t1:
            fig = plt.figure(figsize=(15, 8))
            plot_tree(clf, feature_names=feats, class_names=class_names, filled=True)
            st.pyplot(fig)
        
        with t2:
            st.code(export_text(clf, feature_names=feats))
            
        with t3:
            imp = pd.DataFrame({'Fitur': feats, 'Importance': clf.feature_importances_}).sort_values('Importance', ascending=False)
            fig, ax = plt.subplots()
            sns.barplot(x='Importance', y='Fitur', data=imp, palette='viridis', ax=ax)
            st.pyplot(fig)
            
        with t4:
            st.subheader("Simulasi Prediksi")
            inputs = {}
            cols = st.columns(2)
            for i, f in enumerate(feats):
                with cols[i%2]: inputs[f] = st.number_input(f"Nilai {f}", value=0)
            
            if st.button("Prediksi"):
                res_idx = clf.predict([list(inputs.values())])[0]
                res_proba = clf.predict_proba([list(inputs.values())])[0]
                
                res_label = str(res_idx)
                if target_col in st.session_state['le_dict']:
                    try: res_label = st.session_state['le_dict'][target_col].inverse_transform([res_idx])[0]
                    except: pass
                
                st.success(f"Hasil: **{res_label}**")
                prob_df = pd.DataFrame({"Kategori": class_names, "Probabilitas": res_proba})
                st.dataframe(prob_df)
    else:
        st.error("Latih Model dulu.")