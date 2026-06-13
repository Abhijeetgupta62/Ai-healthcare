"""
response_generator.py  —  Composes answers from RAG chunks + ML classifier
"""

import re, pickle
from rag_engine import RAGRetriever

# ── Intent keywords ───────────────────────────────────────────────
INTENT_MAP = {
    "symptoms":             ["symptom","sign","feel","feeling","experience","indicate","showing"],
    "causes":               ["cause","reason","why","how get","origin","risk factor","how does"],
    "precautions":          ["precaution","prevent","prevention","avoid","protect","safe","stop"],
    "home_remedies":        ["home remedy","natural","herbal","ayurvedic","without medicine","घरेलू","desi"],
    "medical_treatment":    ["treat","treatment","medicine","medication","drug","cure","hospital","antibiotic","doctor give"],
    "emergency_signs":      ["emergency","danger","critical","urgent","serious","alarming","warning","rush"],
    "emergency_action":     ["what to do","first aid","immediately","help now","action","quick","fast"],
    "diet_advice":          ["diet","food","eat","drink","nutrition","avoid eating","what to eat","khana"],
    "advantages_of_treatment":["advantage","benefit","why treat","importance","why medicine","good"],
    "when_to_see_doctor":   ["doctor","consult","visit","when","should i go","checkup","physician"],
    "recovery_time":        ["recover","recovery","how long","duration","days","weeks","heal","time"],
}

EMERGENCY_TRIGGERS = [
    "chest pain","can't breathe","difficulty breathing","not breathing","unconscious",
    "heart attack","stroke","seizure","severe bleeding","faint","collapsed","overdose",
    "choking","severe head injury","emergency","ambulance","help immediately"
]

GREETINGS = ["hello","hi","hey","namaste","good morning","good afternoon","good evening",
             "hii","helo","sup","greetings","howdy"]

HELP_WORDS = ["help","what can you do","capabilities","features","what do you know",
              "list disease","all disease","disease list"]


def detect_intent(text):
    tl = text.lower()
    intents = []
    for intent, kw in INTENT_MAP.items():
        if any(k in tl for k in kw):
            intents.append(intent)
    # default: show overview when no specific intent found
    if not intents:
        intents = ["symptoms", "precautions", "medical_treatment", "home_remedies"]
    return intents


def is_emergency(text):
    tl = text.lower()
    return any(t in tl for t in EMERGENCY_TRIGGERS)


def is_greeting(text):
    tl = text.strip().lower()
    return any(tl.startswith(g) for g in GREETINGS) and len(tl.split()) <= 5


def is_help(text):
    tl = text.lower()
    return any(w in tl for w in HELP_WORDS)


def fmt_list(items):
    if isinstance(items, list):
        return "\n".join(f"  • {i}" for i in items)
    return f"  {items}"


# ── Pre-built fixed responses ─────────────────────────────────────
WELCOME = """👋 **Hello! Welcome to the Healthcare AI Assistant.**

I can help you with information about **22+ diseases** including:

  • 🤒 **Symptoms** — Know what signs to look for
  • 🔬 **Causes & Risk Factors** — Understand why diseases happen
  • 🛡️ **Precautions** — How to prevent illness
  • 🌿 **Home Remedies** — Natural & ayurvedic treatments
  • 💊 **Medical Treatments** — Medicines & procedures
  • 🚨 **Emergency Signs & First Aid** — Critical life-saving info
  • 🥗 **Diet Advice** — What to eat and avoid
  • ⏱️ **Recovery Time** — How long healing takes

**Try asking:**
  _"What are the symptoms of dengue?"_
  _"How to treat diabetes at home?"_
  _"Emergency signs of a heart attack"_
  _"Precautions for tuberculosis"_

How can I help you today? 😊"""

HELP_MSG = """🏥 **Healthcare Chatbot — 22 Diseases Covered:**

**Viral & Bacterial Infections:**
  Influenza, COVID-19, Dengue, Malaria, Typhoid, Tuberculosis, Pneumonia, Chickenpox, UTI

**Chronic & Metabolic:**
  Diabetes (Type 2), Hypertension, Asthma, Rheumatoid Arthritis, Anemia

**Cardiovascular & Neurological:**
  Heart Attack, Stroke, Migraine

**Gastrointestinal & Urological:**
  Peptic Ulcer, Kidney Stones, Appendicitis, Jaundice

**Mental Health:**
  Depression

**I can answer about:** Symptoms • Causes • Precautions • Home Remedies • Treatments • Emergency First Aid • Diet • Recovery Time

Just ask naturally — e.g. _"dengue symptoms"_ or _"how to prevent TB"_ 🩺"""

GLOBAL_EMERGENCY = """🚨 **EMERGENCY DETECTED**

**📞 Call 112 immediately (India Emergency)**
**📞 Ambulance: 108**
**📞 Mental Health Crisis: iCall 9152987821**

---

**🆘 Universal First Aid Steps:**
  • Keep the person calm and still
  • Check airway, breathing, and circulation (ABC)
  • Do NOT move if spinal injury suspected
  • Begin CPR if unresponsive and not breathing normally
  • Do NOT give food or water
  • Stay on the line with emergency services
  • Send someone to guide the ambulance

---

_Tell me the **disease or condition** for specific emergency guidance._"""


def emergency_card(record):
    signs  = fmt_list(record.get("emergency_signs", ["Seek immediate medical help"]))
    action = record.get("emergency_action", "Call 112 immediately.")
    return (
        f"🚨 **EMERGENCY — {record['disease'].upper()}**\n\n"
        f"**⚠️ Warning Signs:**\n{signs}\n\n"
        f"**🆘 Immediate Action:**\n  {action}\n\n"
        f"---\n"
        f"📞 **Emergency (India):** 112 | **Ambulance:** 108 | **iCall (Mental Health):** 9152987821\n\n"
        f"_Do NOT wait — call for help now._"
    )


# section config: (header emoji + title, record key)
SECTION_CONFIG = {
    "symptoms":             ("🤒 Symptoms",                "symptoms"),
    "causes":               ("🔬 Causes & Risk Factors",    "causes"),
    "precautions":          ("🛡️ Precautions",              "precautions"),
    "home_remedies":        ("🌿 Home Remedies",            "home_remedies"),
    "medical_treatment":    ("💊 Medical Treatment",        "medical_treatment"),
    "emergency_signs":      ("🚨 Emergency Warning Signs",  "emergency_signs"),
    "emergency_action":     ("🆘 Emergency Action",         "emergency_action"),
    "diet_advice":          ("🥗 Diet & Nutrition",         "diet_advice"),
    "advantages_of_treatment":("✅ Advantages of Treatment","advantages_of_treatment"),
    "when_to_see_doctor":   ("👨‍⚕️ When to See a Doctor",   "when_to_see_doctor"),
    "recovery_time":        ("⏱️ Recovery Time",            "recovery_time"),
}


def compose_response(chunks, intents):
    if not chunks:
        return ("I'm sorry, I couldn't find relevant information for your query.\n"
                "Please try rephrasing, or ask about a specific disease like _'dengue symptoms'_.")

    # pick the top disease by highest score
    best = chunks[0]
    record = best["full_record"]

    parts = [f"## 🏥 {record['disease']}\n_{record['category']}_\n"]

    for intent in intents:
        if intent not in SECTION_CONFIG:
            continue
        title, key = SECTION_CONFIG[intent]
        content = record.get(key, None)
        if not content:
            continue
        parts.append(f"### {title}\n{fmt_list(content)}")

    # always append emergency helpline footer when emergency intent present
    if "emergency_signs" in intents or "emergency_action" in intents:
        parts.append(
            "---\n📞 **Helplines (India):** Ambulance **108** | Emergency **112** | "
            "Mental Health **iCall 9152987821**"
        )
        
    return "\n\n".join(parts)


# ── Classifier (ML model for disease detection) ───────────────────
def load_classifier():
    try:
        with open("ml_pipeline.pkl", "rb") as f:
            pipeline = pickle.load(f)
        with open("label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
        return pipeline, le
    except Exception:
        return None, None


class HealthcareResponder:
    def __init__(self):
        print("[Bot] Loading RAG engine …")
        self.retriever = RAGRetriever()
        print("[Bot] Loading ML classifier …")
        self.pipeline, self.le = load_classifier()
        print("[Bot] Ready ✓")

    def _ml_disease_hint(self, text):
        """Use ML classifier to get top disease prediction."""
        if self.pipeline is None:
            return None, 0.0
        proba = self.pipeline.predict_proba([text])[0]
        best_idx = proba.argmax()
        confidence = proba[best_idx]
        if confidence > 0.40:
            return self.le.classes_[best_idx], confidence
        return None, 0.0

    def respond(self, user_message):
        msg = user_message.strip()
        if not msg:
            return "Please type a message to get started. 😊"

        # Greeting
        if is_greeting(msg):
            return WELCOME

        # Help / capabilities
        if is_help(msg):
            return HELP_MSG

        # Global emergency (no disease context)
        if is_emergency(msg):
            chunks = self.retriever.retrieve(msg, top_k=5)
            if chunks:
                top = chunks[0]["full_record"]
                # only show disease card if score is meaningful
                if chunks[0]["score"] > 0.10:
                    return emergency_card(top)
            return GLOBAL_EMERGENCY

        # RAG retrieval
        chunks  = self.retriever.retrieve(msg, top_k=8)
        intents = detect_intent(msg)

        # Boost with ML classifier hint
        ml_disease, ml_conf = self._ml_disease_hint(msg)
        if ml_disease and chunks:
            top_disease = chunks[0]["disease"]
            # if ML strongly disagrees with RAG top pick, blend in ML result
            if ml_disease.lower() != top_disease.lower() and ml_conf > 0.60:
                rec = self.retriever.get_disease_record(ml_disease)
                if rec:
                    # inject a synthetic top chunk for the ML-predicted disease
                    synth = {
                        "disease": ml_disease,
                        "category": rec["category"],
                        "section": "symptoms",
                        "text": "",
                        "score": ml_conf,
                        "full_record": rec,
                    }
                    chunks.insert(0, synth)

        # Emergency intent → show emergency card
        if "emergency_signs" in intents or "emergency_action" in intents:
            if chunks:
                return emergency_card(chunks[0]["full_record"])

        return compose_response(chunks, intents)


if __name__ == "__main__":
    bot = HealthcareResponder()
    tests = [
        "hello",
        "symptoms of dengue fever",
        "how to treat high blood pressure",
        "emergency signs of heart attack",
        "diet for diabetes",
        "home remedy for flu",
    ]
    for t in tests:
        print(f"\n{'='*50}\nQ: {t}\n{'-'*50}")
        print(bot.respond(t)[:400])
