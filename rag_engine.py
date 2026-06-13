"""
rag_engine.py  —  Local RAG engine (TF-IDF + cosine similarity, no internet)
Builds a searchable index over the healthcare dataset chunks.
"""

import json, pickle, os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATASET_PATH = "healthcare_dataset.json"
INDEX_PATH   = "rag_index.pkl"


def build_chunks(data):
    chunks = []
    for rec in data:
        disease, cat = rec["disease"], rec["category"]
        def add(section, content):
            if isinstance(content, list):
                text = f"{disease} {section}: " + " ".join(content)
            else:
                text = f"{disease} {section}: {content}"
            chunks.append({"disease": disease, "category": cat,
                            "section": section, "text": text,
                            "full_record": rec})
        add("symptoms",             rec.get("symptoms", []))
        add("causes",               rec.get("causes", ""))
        add("precautions",          rec.get("precautions", []))
        add("home_remedies",        rec.get("home_remedies", []))
        add("medical_treatment",    rec.get("medical_treatment", []))
        add("emergency_signs",      rec.get("emergency_signs", []))
        add("emergency_action",     rec.get("emergency_action", ""))
        add("diet_advice",          rec.get("diet_advice", ""))
        add("advantages_of_treatment", rec.get("advantages_of_treatment", []))
        add("when_to_see_doctor",   rec.get("when_to_see_doctor", ""))
        add("recovery_time",        rec.get("recovery_time", ""))
    return chunks


def build_index(force=False):
    if not force and os.path.exists(INDEX_PATH):
        print("[RAG] Loading existing index …")
        with open(INDEX_PATH, "rb") as f:
            obj = pickle.load(f)
        return obj["vectorizer"], obj["matrix"], obj["chunks"]

    print("[RAG] Building TF-IDF index …")
    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    chunks = build_chunks(data)
    texts  = [c["text"] for c in chunks]
    vec    = TfidfVectorizer(ngram_range=(1, 3), max_features=30000,
                              sublinear_tf=True, stop_words="english")
    matrix = vec.fit_transform(texts)
    with open(INDEX_PATH, "wb") as f:
        pickle.dump({"vectorizer": vec, "matrix": matrix, "chunks": chunks}, f)
    print(f"[RAG] Index built: {len(chunks)} chunks")
    return vec, matrix, chunks


class RAGRetriever:
    def __init__(self):
        self.vec, self.matrix, self.chunks = build_index()

    def retrieve(self, query, top_k=8):
        q_vec  = self.vec.transform([query])
        scores = cosine_similarity(q_vec, self.matrix).flatten()
        top_idx = scores.argsort()[::-1][:top_k]
        results = []
        for i in top_idx:
            if scores[i] > 0:
                c = dict(self.chunks[i])
                c["score"] = float(scores[i])
                results.append(c)
        return results

    def get_disease_record(self, name):
        nl = name.lower()
        for c in self.chunks:
            if nl in c["disease"].lower() and c["section"] == "symptoms":
                return c["full_record"]
        return None


if __name__ == "__main__":
    r = RAGRetriever()
    for res in r.retrieve("fever chills muscle pain")[:3]:
        print(res["disease"], "→", res["section"], f"[{res['score']:.3f}]")
