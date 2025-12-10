import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
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

# CSS Custom
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
        'Nama': [f'Mhs_{i}' for i in range(1, n+1)],
        'Jam_Belajar': np.random.randint(2, 14, n),
        'Jam_Tidur': np.random.randint(3, 9, n),
        'Ikut_Organisasi': np.random.choice(['Ya', 'Tidak'], n),
        'Tekanan_Tugas': np.random.choice(['Ringan', 'Sedang', 'Berat'], n),
        'Kelelah_Emosional': np.random.choice(['Jarang', 'Sering', 'Selalu'], n)
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

    st.info("Web app ini digunakan untuk memprediksi tingkat risiko burnout pada mahasiswa dengan metode Decision Tree melalui analisis faktor-faktor seperti beban kuliah, waktu istirahat, keaktifan organisasi, dan dampak aktivitas harian.")

    st.markdown("### Anggota Kelompok 3")
    col1, col2, col3 = st.columns(3)
    
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

# --- HALAMAN 2: INPUT DATA ---
elif selected == "Input Data":
    st.title("Input Data")
    c1, c2 = st.columns([1, 2])
    with c1:
        opt = st.radio("Sumber Data:", ["Upload CSV", "Data Dummy"])
        if opt == "Upload CSV":
            uploaded_file = st.file_uploader("Upload File CSV", type=["csv"])
            
            if uploaded_file:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, sep=None, engine='python')
                    df.columns = df.columns.str.strip() 
                    st.session_state['df'] = df
                    st.success(f"✅ CSV Berhasil Dimuat! ({len(df.columns)} kolom)")
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
        
        tab_clean, tab_calc, tab_encode, tab_feature = st.tabs(["1. Pembersihan", "2. Hitung Label Burnout", "3. Encoding (Angka)", "4. Seleksi Fitur Penting"])
        
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
            st.subheader("Kalkulator Risiko Burnout (Metode Agregat)")
            st.info("Pilih kolom dari dataset Anda yang merupakan indikator burnout.")
            
            abaikan_otomatis = ['nama', 'risiko_burnout_final', 'total_skor', 'nim', 'id', 'no', 'email', 'timestamp']
            opsi_kolom = [c for c in df.columns if c.lower() not in abaikan_otomatis]
            
            target_cols = st.multiselect(
                label="Pilih Indikator Penentu Burnout:",
                options=opsi_kolom,
                default=opsi_kolom,
                help="Centang variabel yang mempengaruhi burnout. Hapus yang tidak relevan."
            )
            
            if target_cols:
                st.write(f"Variabel terpilih: *{len(target_cols)} indikator*")
            else:
                st.warning("Silakan pilih minimal satu variabel.")

            st.markdown("---")

            if st.button("Hitung Total Risiko Burnout"):
                if not target_cols:
                    st.error("Pilih indikator terlebih dahulu.")
                else:
                    def hitung_skor_semua(row):
                        total_skor = 0
                        for col in target_cols:
                            val = row[col]
                            # A. Logic Angka
                            if isinstance(val, (int, float, np.number)):
                                if 'tidur' in col.lower() or 'istirahat' in col.lower():
                                    if val < 5: total_skor += 3
                                    elif val < 7: total_skor += 1
                                else:
                                    if val >= 4: total_skor += 3
                                    elif val >= 2: total_skor += 1
                            # B. Logic Teks
                            elif isinstance(val, str):
                                v = val.lower()
                                if any(x in v for x in ['ya', 'berat', 'tinggi', 'buruk', 'sering', 'sangat', 'selalu']):
                                    total_skor += 3
                                elif any(x in v for x in ['sedang', 'cukup', 'kadang']):
                                    total_skor += 1
                        return total_skor

                    df['Total_Skor'] = df.apply(hitung_skor_semua, axis=1)
                    q1 = df['Total_Skor'].quantile(0.33)
                    q2 = df['Total_Skor'].quantile(0.66)
                    
                    def get_label(skor):
                        if skor <= q1: return "Rendah"
                        elif skor <= q2: return "Sedang"
                        else: return "Tinggi"

                    df['Risiko_Burnout_Final'] = df['Total_Skor'].apply(get_label)
                    st.session_state['df'] = df
                    st.success(f"Selesai! Batas Skor: Rendah (≤{q1:.0f}), Sedang ({q1:.0f}-{q2:.0f}), Tinggi (>{q2:.0f})")
                    st.dataframe(df.head(), use_container_width=True)

        # TAB 3: ENCODING
        with tab_encode:
            st.subheader("Ubah Teks Menjadi Angka")
            cat_cols = df.select_dtypes(include=['object']).columns
            st.write(f"Kolom Teks: {list(cat_cols)}")
            
            if st.button("Lakukan Encoding"):
                le_dict = {}
                for col in cat_cols:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    le_dict[col] = le
                
                st.session_state['df'] = df
                st.session_state['le_dict'] = le_dict
                st.success("Encoding Selesai! Data siap untuk training.")
                st.dataframe(df.head(), use_container_width=True)

        # TAB 4: FEATURE SELECTION
        with tab_feature:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 15px; margin-bottom: 20px;'>
                <h2 style='color: white; margin: 0; text-align: center;'> Seleksi Fitur Penting</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("Sistem akan menganalisis importance setiap fitur.")
            
            if 'Risiko_Burnout_Final' not in df.columns:
                st.warning("⚠ Kolom 'Risiko_Burnout_Final' belum ada. Silakan hitung label burnout di Tab 2.")
            else:
                if df.select_dtypes(include=['object']).shape[1] > 0:
                    st.warning("⚠ Masih ada kolom teks. Silakan encoding di Tab 3.")
                else:
                    ignore_cols = ['Risiko_Burnout_Final', 'Total_Skor', 'total_skor', 'Nama', 'nama']
                    feature_cols = [c for c in df.columns if c not in ignore_cols]
                    
                    if len(feature_cols) == 0:
                        st.error("Tidak ada fitur yang bisa digunakan.")
                    else:
                        st.markdown("### Pengaturan Threshold")
                        threshold = st.slider("Threshold Importance (%)", 0.0, 20.0, 5.0, 0.5)

                        if st.button("Analisis Importance Fitur", type="primary", use_container_width=True):
                            try:
                                X_temp = df[feature_cols]
                                y_temp = df['Risiko_Burnout_Final']
                                
                                clf_temp = DecisionTreeClassifier(max_depth=4, random_state=42)
                                clf_temp.fit(X_temp, y_temp)
                                
                                importances_temp = clf_temp.feature_importances_
                                feat_imp_df = pd.DataFrame({
                                    'Fitur': feature_cols,
                                    'Importance (%)': importances_temp * 100
                                }).sort_values('Importance (%)', ascending=False)
                                
                                st.session_state['feature_importance_analysis'] = feat_imp_df
                                
                                important_feats = feat_imp_df[feat_imp_df['Importance (%)'] >= threshold]
                                unimportant_feats = feat_imp_df[feat_imp_df['Importance (%)'] < threshold]
                                
                                st.markdown("---")
                                st.markdown("### Hasil Analisis")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.success(f"✅ Fitur Penting: {len(important_feats)}")
                                with col2:
                                    st.error(f"❌ Fitur Tidak Penting: {len(unimportant_feats)}")

                                # Visualisasi
                                fig_temp = px.bar(
                                    feat_imp_df.sort_values('Importance (%)'),
                                    x='Importance (%)', y='Fitur', orientation='h',
                                    title=f"Feature Importance (Threshold: {threshold}%)",
                                    color='Importance (%)', color_continuous_scale='Viridis'
                                )
                                fig_temp.add_vline(x=threshold, line_dash="dash", line_color="red")
                                st.plotly_chart(fig_temp, use_container_width=True)
                                
                                # Tabel
                                st.dataframe(feat_imp_df.style.format({"Importance (%)": "{:.2f}%"}), use_container_width=True)

                                if not unimportant_feats.empty:
                                    st.markdown("### 🗑 Aksi")
                                    if st.button("Hapus Fitur Tidak Penting", type="primary"):
                                        cols_to_drop = [c for c in unimportant_feats['Fitur'].tolist() if c in df.columns]
                                        if cols_to_drop:
                                            df_cleaned = df.drop(columns=cols_to_drop)
                                            st.session_state['df'] = df_cleaned
                                            st.success(f"Berhasil menghapus {len(cols_to_drop)} fitur!")
                                            st.rerun()

                            except Exception as e:
                                st.error(f"Error: {e}")

# --- HALAMAN 4: TRAINING MODEL ---
elif selected == "Analisis Model":
    st.title("Training Model")
    
    if st.session_state['df'] is not None:
        df = st.session_state['df']
        
        if df.select_dtypes(include=['object']).shape[1] > 0:
            st.error("⚠ Masih ada kolom Teks! Silakan kembali ke Preprocessing -> Tab 3 Encoding.")
        else:
            c1, c2 = st.columns(2)
            with c1: 
                target_options = list(df.columns)
                default_idx = target_options.index('Risiko_Burnout_Final') if 'Risiko_Burnout_Final' in target_options else len(target_options)-1
                target = st.selectbox("Target (Label Output)", target_options, index=default_idx)
            
            with c2: 
                ignore_cols = [target, 'Total_Skor', 'total_skor'] 
                feature_options = [c for c in df.columns if c not in ignore_cols]
                
                default_features = feature_options
                if 'feature_importance_analysis' in st.session_state:
                    important_feats = st.session_state['feature_importance_analysis']
                    important_feats = important_feats[important_feats['Importance (%)'] >= 5.0]['Fitur'].tolist()
                    important_feats = [f for f in important_feats if f in feature_options]
                    if important_feats:
                        default_features = important_feats
                        st.info(f"✅ Menggunakan {len(important_feats)} fitur penting otomatis.")
                
                feats = st.multiselect("Fitur (Data Input)", feature_options, default=default_features)
                st.caption("ℹ 'Total_Skor' otomatis disembunyikan.")
            
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
                        st.success(f"✅ Model Berhasil Dilatih! Akurasi: {acc:.1%}")
                    except Exception as e:
                        st.error(f"Error Training: {e}")
    else:
        st.warning("Data kosong.")

# --- HALAMAN 5: VISUALISASI ---
elif selected == "Visualisasi":
    st.title("Visualisasi Hasil Analisis")
    
    if st.session_state['model']:
        clf = st.session_state['model']
        target_col = st.session_state['target']
        feats = st.session_state['features']
        
        class_names = [str(c) for c in clf.classes_]
        if target_col in st.session_state['le_dict']:
            try:
                class_names = [str(c) for c in st.session_state['le_dict'][target_col].inverse_transform(clf.classes_)]
            except: pass
            
        t1, t2, t3 = st.tabs([" Visualisasi & Distribusi", " Pohon Keputusan", " Simulasi Prediksi"])
        
        with t1:
            st.subheader("Faktor Paling Berpengaruh")
            
            importances = clf.feature_importances_
            imp_df = pd.DataFrame({
                'Faktor': feats, 
                'Pentingnya (%)': importances * 100
            }).sort_values('Pentingnya (%)', ascending=True) 
            
            fig = px.bar(
                imp_df, x='Pentingnya (%)', y='Faktor', orientation='h',
                text_auto='.1f', color='Pentingnya (%)',
                color_continuous_scale='Viridis',
                title="Kontribusi Faktor Terhadap Burnout"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.subheader(" Distribusi Dataset")
            
            if st.session_state['df'] is not None:
                df = st.session_state['df']
                target_col_name = target_col
                
                if target_col_name in df.columns:
                    distribusi = df[target_col_name].value_counts()
                    
                    if target_col_name in st.session_state.get('le_dict', {}):
                        try:
                            le = st.session_state['le_dict'][target_col_name]
                            distribusi.index = le.inverse_transform(distribusi.index)
                        except: pass
                    
                    distribusi_df = pd.DataFrame({'Kategori': distribusi.index, 'Jumlah': distribusi.values})
                    
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        fig_pie = px.pie(distribusi_df, values='Jumlah', names='Kategori', title="Proporsi Kategori Burnout")
                        st.plotly_chart(fig_pie, use_container_width=True)
                    with c2:
                        st.write("Detail Data:")
                        st.dataframe(distribusi_df, use_container_width=True)

        with t2:
            st.subheader(" Pohon Keputusan")
            fig = plt.figure(figsize=(25, 12))
            plot_tree(clf, feature_names=feats, class_names=class_names, filled=True, rounded=True, fontsize=10)
            st.pyplot(fig)

        with t3:
            st.subheader(" Simulasi Prediksi Manual")
            st.info("Masukkan nilai antara 0 sampai 5.")
            
            inputs = {}
            num_cols = min(3, len(feats))
            cols = st.columns(num_cols)
            
            for i, f in enumerate(feats):
                with cols[i % num_cols]: 
                    inputs[f] = st.number_input(f"{f}", value=0.0, min_value=0.0, max_value=5.0, step=1.0)
            
            st.markdown("---")
            if st.button("🔍 Prediksi Risiko", type="primary"):
                res_idx = clf.predict([list(inputs.values())])[0]
                res_proba = clf.predict_proba([list(inputs.values())])[0]
                
                res_label = str(res_idx)
                if target_col in st.session_state['le_dict']:
                    try: res_label = st.session_state['le_dict'][target_col].inverse_transform([res_idx])[0]
                    except: pass
                
                prob_df = pd.DataFrame({
                    "Kategori": class_names, 
                    "Probabilitas (%)": res_proba * 100
                }).sort_values('Probabilitas (%)', ascending=False)
                
                col1, col2, col3 = st.columns(3)
                
                # Default values
                p_tinggi = p_sedang = p_rendah = 0.0
                
                for idx, row in prob_df.iterrows():
                    cat = str(row['Kategori'])
                    val = row['Probabilitas (%)']
                    if "Tinggi" in cat or "Berat" in cat: p_tinggi = val
                    elif "Sedang" in cat: p_sedang = val
                    else: p_rendah = val
                
                with col1:
                    st.markdown(f"<div style='background:#ff4444;padding:20px;border-radius:10px;text-align:center;color:white'><h3>TINGGI</h3><h1>{p_tinggi:.1f}%</h1></div>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"<div style='background:#ffaa00;padding:20px;border-radius:10px;text-align:center;color:white'><h3>SEDANG</h3><h1>{p_sedang:.1f}%</h1></div>", unsafe_allow_html=True)
                with col3:
                    st.markdown(f"<div style='background:#00cc00;padding:20px;border-radius:10px;text-align:center;color:white'><h3>RENDAH</h3><h1>{p_rendah:.1f}%</h1></div>", unsafe_allow_html=True)
                
                st.markdown("---")
                st.subheader(f"Hasil Akhir: {res_label}")

    else:
        st.error("⚠ Model belum dilatih. Silakan ke menu 'Analisis Model'.")