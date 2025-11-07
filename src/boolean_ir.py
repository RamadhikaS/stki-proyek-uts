# --- Fungsi Pembuat Index ---
# (Salinan dari Langkah 9)
def create_inverted_index(corpus):
    inverted_index = {}
    for doc_id, doc in enumerate(corpus):
        terms = doc.split()
        for term in terms:
            if term not in inverted_index:
                inverted_index[term] = set()
            inverted_index[term].add(doc_id)
    return inverted_index

# --- Fungsi Pencarian Boolean ---
# (Salinan dari Langkah 11)
def search_and(term1, term2, inverted_index):
    set1 = inverted_index.get(term1, set())
    set2 = inverted_index.get(term2, set())
    return set1 & set2

def search_or(term1, term2, inverted_index):
    set1 = inverted_index.get(term1, set())
    set2 = inverted_index.get(term2, set())
    return set1 | set2

def search_not(term1, term2, inverted_index):
    set1 = inverted_index.get(term1, set())
    set2 = inverted_index.get(term2, set())
    return set1 - set2