"""
train_model.py
==============
Trains a lightweight intent / disease classifier on the healthcare dataset.
Saves:
  • tfidf_vectorizer.pkl   – TF-IDF feature extractor
  • disease_classifier.pkl – LogisticRegression multi-class model
  • label_encoder.pkl      – maps class indices ↔ disease names
  • training_report.txt    – accuracy + classification report

Run once:  python train_model.py
"""

import json, pickle, os, datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline


# ── 1. Load dataset ────────────────────────────────────────────────
print("[Trainer] Loading dataset …")
with open("healthcare_dataset.json", encoding="utf-8") as f:
    data = json.load(f)


# ── 2. Build training corpus ───────────────────────────────────────
def record_to_texts(rec: dict) -> list[str]:
    """Generate multiple training sentences per disease for richer coverage."""
    d = rec["disease"]
    samples = []

    def add(tmpl, content):
        if isinstance(content, list):
            samples.append(tmpl + " " + " ".join(content))
        elif isinstance(content, str) and content:
            samples.append(tmpl + " " + content)

    add(f"{d} symptoms include", rec.get("symptoms", []))
    add(f"Signs of {d} are", rec.get("symptoms", []))
    add(f"How do I know if I have {d}", rec.get("symptoms", []))
    add(f"{d} is caused by", rec.get("causes", ""))
    add(f"Why does {d} happen", rec.get("causes", ""))
    add(f"Prevent {d} by", rec.get("precautions", []))
    add(f"Precautions for {d}", rec.get("precautions", []))
    add(f"Home remedy for {d}", rec.get("home_remedies", []))
    add(f"Natural treatment for {d}", rec.get("home_remedies", []))
    add(f"Medicine for {d}", rec.get("medical_treatment", []))
    add(f"Treat {d} with", rec.get("medical_treatment", []))
    add(f"Emergency {d}", rec.get("emergency_signs", []))
    add(f"Danger signs of {d}", rec.get("emergency_signs", []))
    add(f"Diet for {d}", rec.get("diet_advice", ""))
    add(f"What to eat with {d}", rec.get("diet_advice", ""))
    add(f"Recovery from {d}", rec.get("recovery_time", ""))
    # category-level samples
    add(rec["category"], rec.get("symptoms", []))

    return samples


texts, labels = [], []
for rec in data:
    for text in record_to_texts(rec):
        texts.append(text)
        labels.append(rec["disease"])

print(f"[Trainer] Corpus: {len(texts)} samples across {len(set(labels))} classes")


# ── 3. Encode labels ───────────────────────────────────────────────
le = LabelEncoder()
y  = le.fit_transform(labels)


# ── 4. Train / test split ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    texts, y, test_size=0.20, random_state=42, stratify=y
)


# ── 5. Build pipeline: TF-IDF + LogReg ────────────────────────────
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=20_000,
        sublinear_tf=True,
        min_df=1,
        analyzer="word",
        stop_words="english",
    )),
    ("clf", LogisticRegression(
        max_iter=2000,
        C=5.0,
        solver="lbfgs",
        
        random_state=42,
    )),
])

print("[Trainer] Training TF-IDF + LogisticRegression …")
pipeline.fit(X_train, y_train)


# ── 6. Evaluate ────────────────────────────────────────────────────
y_pred   = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cv_scores = cross_val_score(pipeline, texts, y, cv=5, scoring="accuracy")
report   = classification_report(y_test, y_pred, target_names=le.classes_)

print(f"\n[Trainer] Test Accuracy : {accuracy*100:.2f}%")
print(f"[Trainer] CV Mean±Std   : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
print("\n" + report)


# ── 7. Save artefacts ──────────────────────────────────────────────
tfidf_vec = pipeline.named_steps["tfidf"]
clf_model = pipeline.named_steps["clf"]

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf_vec, f)

with open("disease_classifier.pkl", "wb") as f:
    pickle.dump(clf_model, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# Save full pipeline too (convenient)
with open("ml_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

# Human-readable report
report_txt = (
    f"Healthcare Chatbot — ML Training Report\n"
    f"Generated : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    f"{'='*55}\n"
    f"Training samples : {len(X_train)}\n"
    f"Test samples     : {len(X_test)}\n"
    f"Classes          : {len(le.classes_)}\n"
    f"Test Accuracy    : {accuracy*100:.2f}%\n"
    f"CV Mean (5-fold) : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%\n"
    f"{'='*55}\n\n"
    f"Classification Report:\n{report}\n"
    f"\nModel Config:\n"
    f"  Vectorizer : TF-IDF  ngram=(1,3)  max_features=20000\n"
    f"  Classifier : LogisticRegression  C=5.0  solver=lbfgs\n"
)
with open("training_report.txt", "w", encoding="utf-8") as f:
    f.write(report_txt)

print("[Trainer] Saved: tfidf_vectorizer.pkl, disease_classifier.pkl,")
print("          label_encoder.pkl, ml_pipeline.pkl, training_report.txt")
print("[Trainer] Training complete ✓")
