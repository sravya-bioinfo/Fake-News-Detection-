# Fake News Detection
**Industry:** Media | **Domain:** AI & ML | **Tools:** Python, Scikit-learn, NLTK

---

## Project Overview

This project builds an NLP-based machine learning model to classify news articles as **REAL** or **FAKE**. It demonstrates end-to-end machine learning: data loading, text preprocessing, model training, evaluation, error analysis, and a prediction interface.

---

## Folder Structure

```
fake_news_detection/
├── fake_news_detection.py   ← Main script (all steps)
├── train.csv                ← Kaggle dataset (download separately)
└── README.md                ← This file
```

---

## Dataset

Download from Kaggle:  
🔗 https://www.kaggle.com/c/fake-news/data

Place `train.csv` in the same folder as the script.  
**Columns used:** `title`, `author`, `text`, `label` (0=REAL, 1=FAKE)

> If the CSV is not found, the script auto-generates synthetic demo data so you can still run it.

---

## How to Run

### 1. Install dependencies
```bash
pip install pandas numpy scikit-learn nltk
```

### 2. Run the script
```bash
python fake_news_detection.py
```

---

## Pipeline Steps

| Step | Description |
|------|-------------|
| 1 | **Load Data** — Read CSV, combine title + author + text |
| 2 | **Preprocess** — Lowercase, remove URLs/HTML/punctuation, remove stopwords |
| 3 | **Train** — TF-IDF Vectorizer + Logistic Regression (80/20 split) |
| 4 | **Evaluate** — Accuracy, Precision, Recall, F1, Confusion Matrix |
| 5 | **Error Analysis** — Review false positives and false negatives |
| 6 | **Predict** — Demo interface to classify your own news text |

---

## Key Concepts

- **TF-IDF (Term Frequency–Inverse Document Frequency):** Converts text into numbers by weighting words that are common in a document but rare across the corpus — great for distinguishing writing styles.
- **Logistic Regression:** A fast, interpretable classifier that works very well on high-dimensional TF-IDF features.
- **Precision vs Recall trade-off:** Precision = how many flagged articles are truly fake; Recall = how many fake articles we successfully caught.

---

## Sample Output

```
══════════════════════════════════════════════════
  MODEL EVALUATION RESULTS
══════════════════════════════════════════════════
  Accuracy  : 0.9823  (98.23%)
  Precision : 0.9811
  Recall    : 0.9836
  F1 Score  : 0.9823
══════════════════════════════════════════════════
```

---

## Extending the Project

- **Use BERT embeddings** (HuggingFace `transformers`) for higher accuracy
- **Add an LSTM** layer for sequence-aware classification
- **Deploy as a web app** using Flask or Streamlit

