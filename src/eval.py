# --- FUNGSI DARI SOAL 03 (Boolean) ---
# (Salinan dari Langkah 14)
def calculate_precision_recall_f1(retrieved_docs, relevant_docs):
    """
    Menghitung Precision, Recall, dan F1-Score untuk model Boolean.
    """
    tp = len(retrieved_docs & relevant_docs)
    fp = len(retrieved_docs - relevant_docs)
    fn = len(relevant_docs - retrieved_docs)
    
    if (tp + fp) == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)
        
    if (tp + fn) == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)
        
    if (precision + recall) == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
        
    return precision, recall, f1


# --- FUNGSI BARU DARI SOAL 04 (VSM) ---
# (Salinan dari Langkah 23)

def calculate_precision_at_k(retrieved_doc_ids, relevant_doc_ids, k):
    """
    Menghitung Precision@k untuk VSM.
    """
    top_k_retrieved = retrieved_doc_ids[:k]
    relevant_and_retrieved_at_k = [doc_id for doc_id in top_k_retrieved if doc_id in relevant_doc_ids]
    
    if k == 0:
        return 0.0
        
    precision_at_k = len(relevant_and_retrieved_at_k) / k
    return precision_at_k

def calculate_average_precision_at_k(retrieved_doc_ids, relevant_doc_ids, k):
    """
    Menghitung Average Precision (AP@k) untuk VSM.
    """
    top_k_retrieved = retrieved_doc_ids[:k]
    
    if not relevant_doc_ids:
        return 0.0

    score = 0.0
    num_relevant_docs_found = 0
    
    for i, doc_id in enumerate(top_k_retrieved):
        rank = i + 1
        
        if doc_id in relevant_doc_ids:
            num_relevant_docs_found += 1
            precision_at_current_rank = num_relevant_docs_found / rank
            score += precision_at_current_rank
            
    if num_relevant_docs_found == 0:
        return 0.0
        
    average_precision = score / len(relevant_doc_ids)
    return min(average_precision, 1.0)