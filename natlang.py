# natlang.py
# Assignment 6 — Natlang minimal app (fixed string literals)
# - Emotion analysis (anger, urgency, sarcasm) + coarse sentiment
# - Intent detection (refund, defect, cancellation)
# - Retrieval grounding with simple Jaccard
# - Response via OpenAI (if OPENAI_API_KEY is set) or graceful local fallback
# - Demo saves five "screenshot" PNGs into ./natlang_artifacts

from dataclasses import dataclass
from typing import List, Dict, Tuple
from datetime import datetime
import os, re, uuid, json

# ------------- Lexicons & Patterns -------------
ANGER_WORDS = set("""
angry furious pissed livid unacceptable infuriating ridiculous annoyed awful
useless broken trash terrible worst hate scam fraud nightmare
""".split())

URGENCY_WORDS = set("""
now asap immediately urgent today right-away rightaway emergency can't-wait cannot-wait
""".split())

SARCASM_MARKERS = [
    r"yeah\s+right",
    r"just\s+great",
    r"wonderful\s+job",
    r"thanks\s+for\s+nothing",
    r"what\s+a\s+surprise",
]

PROFANITY = set("damn hell crap shit fuck freaking fricking".split())

REFUND_PATTERNS = [
    r"\brefund\b",
    r"\bchargeback\b",
    r"\bmy\s+money\s+back\b",
]

DEFECT_PATTERNS = [
    r"\bbroken\b",
    r"\bdefect\w*\b",
    r"\bdoes(n't| not)\s+work\b",
    r"\bfaulty\b",
    r"\bcrash\w*\b",
    r"\boverheat\w*\b",
]

# ------------- Models -------------
@dataclass
class Analysis:
    sentiment: str
    emotions: Dict[str, float]
    intents: List[str]
    characteristics: List[str]

@dataclass
class PastComplaint:
    text: str
    category: str
    resolution: str

PAST_COMPLAINTS = [
    PastComplaint("Battery dies after 2 hours, device overheats.", "battery", "Offered replacement and extended warranty"),
    PastComplaint("App crashes on checkout, lost my cart twice.", "app_crash", "Issued 15% off code and logged bug ticket"),
    PastComplaint("Package arrived late and box was damaged.", "shipping", "Refunded shipping and reshipped item"),
    PastComplaint("Support never replied, waited 5 days!", "support_delay", "Escalated to Tier 2 and apologized"),
    PastComplaint("Wrong size sent even after confirmation email.", "fulfillment", "Sent prepaid return label and correct size"),
]

# ------------- Helpers -------------
def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9\s']", " ", text.lower())

def score_anger(tokens: List[str]) -> float:
    hits = sum(1 for t in tokens if t in ANGER_WORDS or t in PROFANITY)
    return min(1.0, hits / max(1, len(tokens)) * 5)

def score_urgency(tokens: List[str]) -> float:
    hits = sum(1 for t in tokens if t in URGENCY_WORDS)
    return min(1.0, hits / max(1, len(tokens)) * 8)

def detect_sarcasm(text: str) -> float:
    return 1.0 if any(re.search(p, text, flags=re.I) for p in SARCASM_MARKERS) else 0.0

def classify_sentiment(emotions: Dict[str, float]) -> str:
    if emotions["anger"] > 0.5: return "negative"
    if emotions["sarcasm"] >= 1.0: return "negative"
    if emotions["urgency"] > 0.4: return "tense"
    return "neutral"

def analyze_emotions(text: str) -> Dict[str, float]:
    toks = normalize(text).split()
    emotions = {
        "anger": score_anger(toks),
        "urgency": score_urgency(toks),
        "sarcasm": detect_sarcasm(text),
    }
    emotions["sentiment"] = classify_sentiment(emotions)
    return emotions

def detect_intents(text: str) -> List[str]:
    intents = []
    if any(re.search(p, text, flags=re.I) for p in REFUND_PATTERNS): intents.append("refund_request")
    if any(re.search(p, text, flags=re.I) for p in DEFECT_PATTERNS): intents.append("product_defect")
    if re.search(r"\bcancel\b", text, flags=re.I): intents.append("cancellation")
    return intents

def characteristics_from(emotions: Dict[str, float], intents: List[str]) -> List[str]:
    chars = []
    if emotions["anger"] > 0.3: chars.append("angry")
    if emotions["urgency"] > 0.3: chars.append("urgent")
    if emotions["sarcasm"] >= 1.0: chars.append("sarcastic")
    if "refund_request" in intents: chars.append("refund_seeking")
    if "product_defect" in intents: chars.append("defect_reported")
    if "cancellation" in intents: chars.append("cancellation_intent")
    return chars

def jaccard(a: set, b: set) -> float:
    return (len(a & b) / len(a | b)) if a and b else 0.0

def find_similar(user_text: str, k: int = 2) -> List[PastComplaint]:
    tokens = set(normalize(user_text).split())
    scored: List[Tuple[float, PastComplaint]] = []
    for pc in PAST_COMPLAINTS:
        s = jaccard(tokens, set(normalize(pc.text).split()))
        scored.append((s, pc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [pc for s, pc in scored[:k] if s > 0]

def generate_reference_code() -> str:
    return f"NC-{datetime.utcnow().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"

# ------------- Response (OpenAI or fallback) -------------
def _llm_openai(user_text: str, analysis: Analysis, similar_tags: str, ref: str) -> str:
    try:
        from openai import OpenAI
        import os as _os
        if not _os.getenv("OPENAI_API_KEY"): raise RuntimeError("OPENAI_API_KEY not set")
        client = OpenAI()
        sys = (
            "You are a de-escalation and solutions specialist. Be empathetic, concise, and action-oriented. "
            "Use at most one apology. If urgency is detected, state same-day prioritization. "
            "Always include the provided reference code verbatim."
        )
        characteristics = ", ".join(analysis.characteristics) or "general inquiry"
        prompt = (
            f"Complaint: {user_text}\n"
            f"Sentiment: {analysis.sentiment}\n"
            f"Characteristics: {characteristics}\n"
            f"Intents: {', '.join(analysis.intents) or 'none'}\n"
            f"Similar cases: {similar_tags}\n"
            f"Reference: {ref}\n"
            "Write 2–3 sentences: acknowledge appropriately (<=1 apology), propose concrete next step(s), "
            "mention prioritization if urgent, and include the reference code."
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":sys},
                      {"role":"user","content":prompt}],
            temperature=0.2
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""  # empty => triggers fallback

def _llm_fallback(user_text: str, analysis: Analysis, similar_tags: str, ref: str) -> str:
    apology = ("We're truly sorry" if ("angry" in analysis.characteristics or analysis.sentiment == "negative") else "Thanks for reaching out")
    actions = []
    if "refund_request" in analysis.intents: actions.append("I can start a refund right away.")
    if "product_defect" in analysis.intents: actions.append("Let's replace the item and log this defect for engineering.")
    if "urgent" in analysis.characteristics: actions.append("I'll prioritize this and get back today.")
    if not actions: actions.append("I'll investigate and update you with options.")
    return (
        f"{apology} about this experience. I'll keep the tone calm and empathetic.\n"
        f"Summary I captured: {', '.join(analysis.characteristics) or 'general inquiry'}.\n"
        f"What I can do now: {' '.join(actions)}\n"
        f"Similar past issues consulted: {similar_tags or 'none'}.\n"
        f"Reference: {ref}"
    )

def llm_craft_response(user_text: str, analysis: Analysis, similar: List[PastComplaint]) -> str:
    ref = generate_reference_code()
    similar_tags = "; ".join({pc.category for pc in similar}) if similar else "none"
    out = _llm_openai(user_text, analysis, similar_tags, ref)
    if not out:
        out = _llm_fallback(user_text, analysis, similar_tags, ref)
    return out

# ------------- Orchestrator -------------
def natlang_handle(user_text: str) -> Dict:
    emo = analyze_emotions(user_text)
    intents = detect_intents(user_text)
    chars = characteristics_from(emo, intents)
    analysis = Analysis(
        sentiment=emo["sentiment"],
        emotions={"anger": emo["anger"], "urgency": emo["urgency"], "sarcasm": emo["sarcasm"]},
        intents=intents,
        characteristics=chars
    )
    similar = find_similar(user_text, k=2)
    reply = llm_craft_response(user_text, analysis, similar)
    return {
        "input": user_text,
        "analysis": {
            "sentiment": analysis.sentiment,
            "emotions": analysis.emotions,
            "intents": analysis.intents,
            "characteristics": analysis.characteristics
        },
        "similar_cases": [{"text": pc.text, "category": pc.category, "resolution": pc.resolution} for pc in similar],
        "output": reply
    }

# ------------- Demo (5 examples + PNGs) -------------
def _render_io_image(idx: int, record: Dict, outdir: str) -> str:
    try:
        import matplotlib.pyplot as plt, textwrap
    except Exception:
        return ""
    wrapped_in = textwrap.fill("INPUT: " + record["input"], width=78)
    meta = record["analysis"]
    meta_txt = f"Sentiment: {meta['sentiment']} | Intents: {', '.join(meta['intents']) or 'none'} | Chars: {', '.join(meta['characteristics']) or 'none'}"
    wrapped_meta = textwrap.fill(meta_txt, width=78)
    wrapped_out = textwrap.fill("OUTPUT:\n" + record["output"], width=78)
    fig = plt.figure(figsize=(10, 8)); plt.axis('off')
    txt = f"Natlang Execution — Example {idx+1}\n\n{wrapped_in}\n\n{wrapped_meta}\n\n{wrapped_out}"
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"example_{idx+1}.png")
    plt.text(0.01, 0.98, txt, va='top', fontsize=12, family="monospace")
    plt.savefig(path, bbox_inches="tight", dpi=200); plt.close(fig)
    return path

def demo():
    examples = [
        "This is ridiculous — the app crashed at checkout again and I want a refund now. It’s basically broken.",
        "Package showed up 5 days late with a crushed box. Yeah right, 'premium shipping'.",
        "I need to cancel my order today, wrong size was sent even though the confirmation showed the right one.",
        "The device overheats after about 20 minutes of video calls. Can you help replace it?",
        "I’d just like a refund for this order; it isn’t what I expected."
    ]
    results = [natlang_handle(t) for t in examples]
    for i, r in enumerate(results):
        print(f"\n--- Example {i+1} ---"); print(json.dumps(r, indent=2))
    outdir = "./natlang_artifacts"; os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "natlang_examples.json"), "w") as f:
        json.dump(results, f, indent=2)
    for i, rec in enumerate(results):
        p = _render_io_image(i, rec, outdir)
        if p: print(f"Saved screenshot: {p}")
        else: print("matplotlib not available: skipping PNG generation.")

if __name__ == "__main__":
    demo()
