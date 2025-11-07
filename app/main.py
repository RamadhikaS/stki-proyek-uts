# Import library
import streamlit as st
import os
import sys

# --- Penting: Menambahkan 'src' ke Path ---
# Ini memberitahu Streamlit di mana harus mencari file 'src' Anda
# (../ artinya "keluar satu folder" dari 'app' ke 'stki-uts-project')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import fungsi-fungsi Anda dari folder 'src'
from src.preprocess import setup_preprocessing, preprocess_text
from src.boolean_ir import create_inverted_index, search_and, search_or, search_not
from src.vsm_ir import create_vsm, search_vsm

# --- Setup Awal Aplikasi (Hanya berjalan sekali) ---
# @st.cache_resource adalah "jurus" agar Streamlit tidak mengulang
# proses yang berat (seperti setup stemmer) setiap kali user klik
@st.cache_resource
def load_models_and_data():
    """
    Memuat data, setup preprocessing, dan membuat model VSM & Boolean.
    Berjalan sekali saja saat aplikasi dimulai.
    """
    # 1. Setup Preprocessing (dari src/preprocess.py)
    stemmer, list_stopword = setup_preprocessing()
    
    # 2. Load Data Korpus
    data_path = "data/" # Path relatif dari file main.py
    korpus_mentah = []
    dokumen_names = []
    for nama_file in sorted(os.listdir(data_path)):
        if nama_file.endswith(".txt"):
            with open(os.path.join(data_path, nama_file), 'r') as f:
                korpus_mentah.append(f.read())
                dokumen_names.append(nama_file)
                
    # 3. Preprocessing Korpus
    korpus_bersih = []
    for doc in korpus_mentah:
        korpus_bersih.append(preprocess_text(doc, stemmer, list_stopword))
        
    # 4. Buat Model VSM (Gunakan Skema TERBAIK Anda: Sublinear TF)
    # (dari src/vsm_ir.py)
    # Kita modifikasi create_vsm untuk menerima argumen sublinear
    vectorizer, tfidf_matrix = create_vsm(korpus_bersih, sublinear=True) # Kita akan update vsm_ir.py
    
    # 5. Buat Model Boolean (dari src/boolean_ir.py)
    inverted_index = create_inverted_index(korpus_bersih)
    
    # Kembalikan semua yang kita butuhkan
    return {
        "stemmer": stemmer,
        "list_stopword": list_stopword,
        "korpus_mentah": korpus_mentah,
        "dokumen_names": dokumen_names,
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "inverted_index": inverted_index
    }

# --- Jalankan setup ---
data_bundle = load_models_and_data()

# --- Tampilan UI (User Interface) Streamlit ---

st.title("Mesin Pencari STKI (Boolean & VSM)")
st.write("Dibuat oleh: [NIM Anda] - [Nama Anda]")
st.write(f"Menggunakan korpus {len(data_bundle['korpus_mentah'])} dokumen.")

# Sidebar untuk memilih model
st.sidebar.header("Pengaturan Model")
model_type = st.sidebar.selectbox(
    "Pilih Model Retrieval:",
    ("Vector Space Model (VSM)", "Boolean Retrieval")
)

# Input Query
query_mentah = st.text_input("Masukkan query Anda:", "informasi mahasiswa")
k = st.sidebar.slider("Jumlah hasil (Top-K)", 1, 5, 3)

# Tombol Cari
if st.button("Cari"):
    
    if not query_mentah:
        st.warning("Silakan masukkan query.")
    else:
        st.subheader("Hasil Pencarian:")
        
        # --- LOGIKA VSM ---
        if model_type == "Vector Space Model (VSM)":
            st.write(f"Menampilkan Top-{k} hasil untuk VSM (Sublinear TF)...")
            
            # Panggil fungsi search_vsm (dari src/vsm_ir.py)
            hasil_vsm = search_vsm(
                query_mentah=query_mentah,
                preprocessor_func=preprocess_text,
                stemmer=data_bundle["stemmer"],
                stopwords=data_bundle["list_stopword"],
                vectorizer=data_bundle["vectorizer"],
                tfidf_matrix=data_bundle["tfidf_matrix"],
                k=k
            )
            
            # Tampilkan hasil VSM
            if not hasil_vsm:
                st.write("Tidak ada dokumen yang relevan.")
            else:
                for rank, (doc_id, score) in enumerate(hasil_vsm):
                    st.markdown(f"**Rank {rank+1} (Skor: {score:.4f}) - {data_bundle['dokumen_names'][doc_id]}**")
                    st.info(data_bundle['korpus_mentah'][doc_id][:200] + "...") # Tampilkan snippet

        # --- LOGIKA BOOLEAN ---
        elif model_type == "Boolean Retrieval":
            st.write("Menampilkan hasil untuk Boolean (Model AND)...")
            
            # Peringatan: Model Boolean ini disederhanakan (hanya AND 2 kata)
            # Ini untuk memenuhi syarat, tapi VSM lebih utama
            query_terms = query_mentah.split()
            
            if len(query_terms) < 2:
                st.warning("Query Boolean sederhana ini memerlukan 2 kata (misal: 'informatika mahasiswa') untuk operasi AND.")
            else:
                # Ambil 2 kata pertama, stem
                q_term1 = data_bundle["stemmer"].stem(query_terms[0])
                q_term2 = data_bundle["stemmer"].stem(query_terms[1])
                
                # Panggil fungsi search_and (dari src/boolean_ir.py)
                hasil_bool = search_and(q_term1, q_term2, data_bundle["inverted_index"])
                
                if not hasil_bool:
                    st.write(f"Tidak ada dokumen yang mengandung '{q_term1}' DAN '{q_term2}'.")
                else:
                    st.write(f"Ditemukan {len(hasil_bool)} dokumen mengandung '{q_term1}' DAN '{q_term2}':")
                    for doc_id in hasil_bool:
                        st.markdown(f"**- {data_bundle['dokumen_names'][doc_id]}**")
                        st.info(data_bundle['korpus_mentah'][doc_id][:200] + "...")