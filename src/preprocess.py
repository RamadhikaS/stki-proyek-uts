import re
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# --- Fungsi Inisialisasi ---
# Kita akan butuh ini agar stemmer & stopwords siap
def setup_preprocessing():
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    list_stopword = stopwords.words('indonesian')

    return stemmer, list_stopword

# --- Fungsi Utama Preprocessing ---
# (Ini adalah salinan dari Langkah 6 Anda)
def preprocess_text(text, stemmer, list_stopword):
    # 1. Case Folding
    text = text.lower()

    # 2. Cleaning
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # 3. Tokenizing
    tokens = nltk.tokenize.word_tokenize(text)

    # 4. Stopword Removal
    cleaned_tokens = [token for token in tokens if token not in list_stopword]

    # 5. Stemming
    stemmed_tokens = [stemmer.stem(token) for token in cleaned_tokens]

    return " ".join(stemmed_tokens)