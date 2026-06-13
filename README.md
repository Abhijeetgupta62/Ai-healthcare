

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch everything (trains model + builds index + starts server)
python run.py

# 3. Open browser
http://localhost:5000
```

---

## 📁 Project Structure

```
healthcare-chatbot/
│
├── healthcare_dataset.json   # 22-disease knowledge base (hand-crafted)
├── train_model.py            # TF-IDF + Logistic Regression trainer
├── rag_engine.py             # RAG retrieval engine (TF-IDF + cosine similarity)
├── response_generator.py     # Intent detection + answer composition
├── app.py                    # Flask web server
├── run.py                    # One-click launcher
├── requirements.txt          # Python dependencies
│
├── templates/
│   └── index.html            # Full chat UI (dark medical theme)
│
├── static/
│   └── style.css             # Extra CSS
│
└── README.md
```

Generated after first run:
```
├── ml_pipeline.pkl           # Trained sklearn Pipeline
├── tfidf_vectorizer.pkl      # TF-IDF feature extractor
├── disease_classifier.pkl    # LogisticRegression model
├── label_encoder.pkl         # Class label encoder
├── rag_index.pkl             # FAISS-style TF-IDF RAG index
└── training_report.txt       # Model accuracy report
```

## 🧠 How It Works

### 1. Dataset (`healthcare_dataset.json`)
Hand-crafted knowledge base with 22 diseases, each containing:
- Symptoms, Causes, Precautions, Home Remedies
- Medical Treatments, Emergency Signs & Actions
- Diet Advice, Recovery Time, When to See a Doctor

### 2. ML Model (`train_model.py`)
- **TF-IDF Vectorizer** (n-gram 1–3, 20,000 features)
- **Logistic Regression** classifier
- **97.33% accuracy** on test set, 94.65% cross-validation
- Trained on 374 augmented samples across 22 disease classes

### 3. RAG Engine (`rag_engine.py`)
- Splits each disease into 11 topical **chunks**
- Builds a **TF-IDF + cosine similarity** search index (242 chunks)
- Returns top-k most relevant chunks for any query
- 100% offline — no internet or GPU required

### 4. Response Generator (`response_generator.py`)
- **Intent detection** from query keywords
- **Hybrid retrieval**: RAG + ML classifier blend
- Structured markdown answers with emoji headers
- Special emergency detection → immediate first-aid response

### 5. Flask UI (`app.py` + `templates/index.html`)
- Dark medical-themed chat interface
- Sidebar with disease list and quick-action buttons
- Suggestion chips, typing indicator, emergency banner
- Markdown rendering in chat bubbles
- Fully responsive design



## 🛠️ Tech Stack

- **Python 3.10+**
- **Flask** — Web framework
- **scikit-learn** — TF-IDF vectorization, Logistic Regression
- **NumPy** — Matrix operations
- **Vanilla HTML/CSS/JS** — Zero-dependency frontend

---

*Built with ❤️ for healthcare awareness*
