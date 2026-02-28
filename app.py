"""
Search Engine using Inverted Index + TF-IDF + Cosine Similarity
Flask Web Application
"""

import math
import os
from collections import defaultdict, Counter
from flask import Flask, render_template, request

app = Flask(__name__)

# ---------------------------
# STOPWORDS LIST
# ---------------------------
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "it", "its", "as", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do",
    "does", "did", "will", "would", "could", "should", "may", "might",
    "this", "that", "these", "those", "they", "them", "their", "there",
    "we", "our", "you", "your", "he", "she", "his", "her", "i", "my",
    "not", "no", "so", "than", "then", "too", "very", "just", "also",
    "about", "into", "more", "even", "both", "each", "such", "how",
    "who", "what", "which", "any", "all", "one", "can", "now", "how"
}

# ---------------------------
# 1. LOAD DOCUMENTS
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
doc_files = ["d1.txt", "d2.txt", "d3.txt", "d4.txt", "d5.txt"]
documents = []
doc_ids = []

for file in doc_files:
    filepath = os.path.join(BASE_DIR, file)
    with open(filepath, "r", encoding="utf-8") as f:
        documents.append(f.read().strip())
    doc_ids.append(os.path.splitext(file)[0].upper())

# ---------------------------
# 2. PREPROCESSING
# ---------------------------
def preprocess(text):
    tokens = text.lower().split()
    cleaned = []
    for word in tokens:
        word = word.strip(".,!?;:()")
        if word and word not in STOPWORDS:
            cleaned.append(word)
    return cleaned

tokenized_docs = [preprocess(doc) for doc in documents]
vocab = sorted(set(word for doc in tokenized_docs for word in doc))

# ---------------------------
# 3. INVERTED INDEX
# ---------------------------
inverted_index = defaultdict(list)
for i, doc in enumerate(tokenized_docs):
    for word in set(doc):
        inverted_index[word].append(doc_ids[i])

# ---------------------------
# 4. TF
# ---------------------------
tf_data = []
for doc in tokenized_docs:
    counter = Counter(doc)
    tf_data.append({term: counter.get(term, 0) for term in vocab})

# ---------------------------
# 5. IDF
# ---------------------------
N = len(documents)
idf_values = {}
df_counts = {}
for term in vocab:
    df = sum(1 for doc in tokenized_docs if term in doc)
    df_counts[term] = df
    idf_values[term] = round(math.log10(N / df), 4)

# ---------------------------
# 6. TF-IDF
# ---------------------------
tfidf_data = []
for doc in tokenized_docs:
    counter = Counter(doc)
    tfidf_data.append([round(counter.get(t, 0) * idf_values[t], 4) for t in vocab])

# ---------------------------
# 7. COSINE SIMILARITY
# ---------------------------
def cosine_similarity(vec1, vec2):
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot / (norm1 * norm2)

# ---------------------------
# FLASK ROUTES
# ---------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    query_input = ""
    query_terms = []
    inverted_index_display = []
    tf_table_data = []
    idf_table_data = []
    tfidf_table_data = []
    query_vector_display = []

    if request.method == "POST":
        query_input = request.form.get("query", "").strip()
        query_terms = preprocess(query_input)

        # Query TF-IDF vector
        query_counter = Counter(query_terms)
        query_vector = [round(query_counter.get(t, 0) * idf_values.get(t, 0), 4) for t in vocab]

        # Cosine similarities
        similarities = [round(cosine_similarity(query_vector, row), 4) for row in tfidf_data]
        ranked = sorted(zip(doc_ids, similarities), key=lambda x: x[1], reverse=True)

        # Determine result message
        best_score = ranked[0][1]
        if best_score == 0:
            result_msg = "No relevant documents found for the query."
            top_docs = []
        else:
            top_docs = [d for d, s in ranked if s == best_score]
            if len(top_docs) == 1:
                result_msg = f"'{query_input}' is most relevant to {top_docs[0]} with a cosine similarity score of {best_score}."
            else:
                result_msg = f"'{query_input}' is equally relevant to {', '.join(top_docs)} with a cosine similarity score of {best_score}."

        results = {
            "query_terms": query_terms,
            "ranked": ranked,
            "result_msg": result_msg,
            "best_score": best_score,
            "top_docs": top_docs,
        }

        # Build display tables (only terms in query for brevity)
        display_terms = [t for t in vocab if t in query_terms] or vocab[:20]

        inverted_index_display = [(t, ", ".join(inverted_index[t])) for t in display_terms if t in inverted_index]
        tf_table_data = [(doc_id, [(t, tf_data[i].get(t, 0)) for t in display_terms]) for i, doc_id in enumerate(doc_ids)]
        idf_table_data = [(t, df_counts.get(t, 0), idf_values.get(t, 0)) for t in display_terms]
        tfidf_table_data = [(doc_id, [(t, round(tf_data[i].get(t, 0) * idf_values.get(t, 0), 4)) for t in display_terms]) for i, doc_id in enumerate(doc_ids)]
        query_vector_display = [(t, round(query_counter.get(t, 0) * idf_values.get(t, 0), 4)) for t in display_terms]

    return render_template(
        "index.html",
        query_input=query_input,
        query_terms=query_terms,
        results=results,
        inverted_index_display=inverted_index_display,
        tf_table_data=tf_table_data,
        idf_table_data=idf_table_data,
        tfidf_table_data=tfidf_table_data,
        query_vector_display=query_vector_display,
        display_terms=[t for t in vocab if t in query_terms] or [],
        doc_ids=doc_ids,
        vocab_size=len(vocab),
    )


@app.route("/documents")
def view_documents():
    docs_preview = []
    for doc_id, doc in zip(doc_ids, documents):
        lines = [l for l in doc.split('\n') if l.strip()]
        docs_preview.append((doc_id, lines[:2]))
    return render_template("documents.html", docs_preview=docs_preview, documents=list(zip(doc_ids, documents)))


if __name__ == "__main__":
    app.run(debug=True)


# Routes: / for search, /documents for document viewer
# Search engine completed and tested