# IR Search Engine — Inverted Index + TF-IDF

A Flask web application implementing a Search Engine using an Inverted Index and TF-IDF ranking with Cosine Similarity.

## Objective

This application demonstrates core Information Retrieval (IR) concepts:
- **Inverted Index** construction from a document collection
- **TF-IDF** (Term Frequency–Inverse Document Frequency) weighting
- **Cosine Similarity** for query-document ranking
- **Stopword removal** during preprocessing

## Tools and Technologies

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| Flask 3.x | Web framework |
| HTML/CSS | Frontend UI |
| math, collections | TF-IDF + Cosine Similarity |

## Project Structure

```
search_engine_flask/
├── app.py                  # Main Flask application + IR logic
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── d1.txt                  # Document 1 — Big Data Analytics
├── d2.txt                  # Document 2 — Big Data & Data Trees
├── d3.txt                  # Document 3 — Data Science & Trees
├── d4.txt                  # Document 4 — Tree Data Structures
├── d5.txt                  # Document 5 — Books for Newbies
└── templates/
    ├── index.html          # Search interface
    └── documents.html      # Document viewer
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/search-engine-flask.git
   cd search-engine-flask
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS/Linux
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

```bash
python app.py
```

Then open your browser and go to: **http://127.0.0.1:5000**

## How It Works

1. **Document Loading** — Five `.txt` documents are loaded at startup
2. **Preprocessing** — Text is lowercased, punctuation stripped, and stopwords removed
3. **Inverted Index** — Maps each term to the list of documents containing it
4. **TF Calculation** — Raw count of each term per document
5. **IDF Calculation** — `IDF = log₁₀(N / DF)` where N = total docs, DF = docs containing the term
6. **TF-IDF** — `TF × IDF` gives the weight of each term in each document
7. **Query Processing** — The query is preprocessed and converted to a TF-IDF vector
8. **Cosine Similarity** — Each document vector is compared to the query vector
9. **Ranking** — Documents sorted by similarity score (highest = most relevant)

## Sample Queries

| Query | Expected Top Result |
|-------|-------------------|
| `data processing` | D1, D2 |
| `tree structure` | D3, D4 |
| `newbie book` | D4, D5 |
| `machine learning` | D1 |
| `data science` | D3 |

## Routes

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET, POST | Main search interface |
| `/documents` | GET | View all documents in the collection |
> Project completed and tested successfully.