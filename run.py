"""
run.py  —  One-click launcher for the Healthcare Chatbot
Usage:  python run.py
"""

import os, sys

def check_requirements():
    missing = []
    for pkg in ["flask", "sklearn", "numpy", "faiss"]:
        try:
            __import__(pkg if pkg != "sklearn" else "sklearn")
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[!] Missing packages: {', '.join(missing)}")
        print("[!] Run:  pip install -r requirements.txt")
        sys.exit(1)

def ensure_model():
    if not os.path.exists("ml_pipeline.pkl"):
        print("[Setup] Training ML model (first run only) …")
        import subprocess
        subprocess.run([sys.executable, "train_model.py"], check=True)
    else:
        print("[Setup] ML model found ✓")

def ensure_rag_index():
    if not os.path.exists("rag_index.pkl"):
        print("[Setup] Building RAG index (first run only) …")
        from rag_engine import build_index
        build_index(force=True)
    else:
        print("[Setup] RAG index found ✓")

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  🏥  Healthcare AI Chatbot")
    print("="*55)

    check_requirements()
    ensure_model()
    ensure_rag_index()

    print("\n[Launch] Starting Flask server …")
    print("[Launch] Open your browser at: http://localhost:5000\n")

    # Import and run app
    from app import app
    app.run(debug=False, host="0.0.0.0", port=5000)
