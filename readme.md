# 🚀 Proyek Pra-UTS Sistem Temu Kembali Informasi (STKI)
Implementasi Model Boolean dan Vector Space Model (VSM)

Proyek ini merupakan implementasi *mini search engine* untuk memenuhi tugas Pra-UTS mata kuliah STKI. Sistem ini mencakup dua model pencarian: **Model Boolean** (yang diimplementasikan secara manual) dan **Model Ruang Vektor (VSM)** (diimplementasikan menggunakan `scikit-learn` dengan pembobotan TF-IDF).

## 🧑‍🎓 Author
* **Nama:** RAMADHIKA SURYA PURMIADANU
* **NIM:** A11.2022.14116
* **Mata Kuliah:** Sistem Temu Kembali Informasi (STKI)

---

## 📚 Fitur Utama
Sistem ini mencakup beberapa komponen inti STKI:

* **Preprocessing Teks:** *Pipeline* lengkap (Tokenisasi, Case-Folding, Stopword Removal, dan Stemming Sastrawi).
* **Model Boolean:**
    * Implementasi `Incidence Matrix` (untuk konsep).
    * Implementasi `Inverted Index` (untuk pencarian).
    * Mendukung *query* `AND`, `OR`, dan `NOT`.
* **Model Vector Space (VSM):**
    * Pembobotan `TF-IDF` menggunakan `TfidfVectorizer`.
    * Perankingan (Ranking) hasil pencarian menggunakan `Cosine Similarity`.
* **Evaluasi Model:**
    * **Boolean:** Menggunakan `Precision`, `Recall`, dan `F1-Score`.
    * **VSM:** Menggunakan `Precision@k` (P@k) dan `Mean Average Precision@k` (MAP@k).
* **Perbandingan Skema:** Menganalisis perbedaan performa antara TF-IDF Standar dan **Sublinear TF-IDF**.
* **Antarmuka (UI):** Aplikasi web interaktif sederhana menggunakan **Streamlit** untuk menguji model VSM terbaik.

---

## 📁 Struktur Folder
stki-uts-A11202214116-Ramadhika/ 
├── src/ # Modul Python (digunakan oleh app/main.py) 
│ ├── preprocess.py # Fungsi preprocessing 
│ ├── boolean_ir.py # Logika model boolean 
│ ├── vsm_ir.py # Logika model VSM 
│ └── eval.py # Fungsi evaluasi
├── app/ # Folder aplikasi Streamlit 
│ ├── main.py # Kode UI Streamlit 
│ ├── vectorizer.pkl # Model Vectorizer (hasil notebook) 
│ ├── tfidf_matrix.pkl # Matriks TF-IDF (hasil notebook) 
│ ├── korpus_mentah.pkl # Data korpus (hasil notebook) 
│ └── dokumen_names.pkl # Data nama file (hasil notebook) 
├── data/ # Korpus 5 dokumen .txt mentah 
│ ├── doc1.txt 
│ └── ... 
├── notebooks/ # File Colab/Jupyter 
│ └── UTS_STKI_A11.2022.14116.ipynb 
├── reports/ # Laporan 
│ └── laporan.pdf 
├── requirements.txt # Daftar dependensi (library) 
└── readme.md # File ini
---

## ⚙️ Instalasi
Proyek ini membutuhkan beberapa *library* Python.

1.  Buat *virtual environment* (dianjurkan):
    ```bash
    python -m venv venv
    source venv/bin/activate  # (atau venv\Scripts\activate di Windows)
    ```
2.  Install semua dependensi yang dibutuhkan:
    ```bash
    pip install -r requirements.txt
    ```

---

## 🏃 Cara Menjalankan

Ada dua cara utama untuk menjalankan proyek ini:

### 1. Menjalankan Analisis & Evaluasi (Notebook)
Cara ini digunakan untuk melihat semua proses, dari *preprocessing*, pembuatan model, hingga evaluasi (P/R/F1 dan MAP@k).

1.  Buka file `notebooks/UTS_STKI_[nim].ipynb` menggunakan **Google Colab** atau **Jupyter Notebook**.
2.  Pastikan Anda telah mengunggah folder `data/` atau menghubungkan Google Drive.
3.  Jalankan semua sel *notebook* secara berurutan dari atas ke bawah.
4.  Hasil analisis dan evaluasi (termasuk skor P/R/F1 dan MAP@k) akan tercetak di *output* sel.

### 2. Menjalankan Aplikasi Web (Streamlit)
Cara ini digunakan untuk berinteraksi langsung dengan model VSM terbaik (Sublinear TF-IDF) yang sudah dilatih.

1.  Pastikan Anda sudah menjalankan *notebook* (Opsi 1) setidaknya satu kali untuk menghasilkan file model (`.pkl`) di dalam folder `app/`.
2.  Buka terminal dan arahkan ke *root* folder proyek ini.
3.  Jalankan perintah Streamlit:
    ```bash
    streamlit run app/main.py
    ```
4.  Buka *browser* Anda dan akses alamat lokal yang diberikan (biasanya `http://localhost:8501`).
