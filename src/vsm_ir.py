from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Fungsi Pembuat VSM (DIPERBARUI) ---
def create_vsm(corpus, sublinear=False):
    """
    Membuat TF-IDF Vectorizer dan Matriks VSM.
    
    Args:
    corpus (list): List dokumen bersih.
    sublinear (bool): Apakah akan menggunakan Sublinear TF Scaling.
    """
    # Gunakan sublinear_tf=True JIKA 'sublinear' adalah True
    vectorizer = TfidfVectorizer(sublinear_tf=sublinear)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf_matrix

# --- Fungsi Pencarian VSM (Tetap sama) ---
def search_vsm(query_mentah, preprocessor_func, stemmer, stopwords, vectorizer, tfidf_matrix, k=3):
    """
    Mencari dan merangking dokumen menggunakan VSM.
    """
    # 1. Proses query
    query_bersih = preprocessor_func(query_mentah, stemmer, stopwords)
    
    # 2. Ubah query jadi vektor
    query_vector = vectorizer.transform([query_bersih])
    
    # 3. Hitung similarity
    scores = cosine_similarity(query_vector, tfidf_matrix)
    skor_dokumen = scores[0]
    
    # 4. Ranking
    paired_scores = list(enumerate(skor_dokumen))
    sorted_scores = sorted(paired_scores, key=lambda x: x[1], reverse=True)
    
    # 5. Ambil Top-k
    top_k_scores = sorted_scores[:k]
    
    return top_k_scores