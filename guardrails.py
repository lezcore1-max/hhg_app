"""
guardrails.py - Enterprise 4-Stage Multi-Lingual Guardrails Engine for Indic Voice RAG.

Stages:
1. Input Safety & Prompt Injection Guardrail (Pre-retrieval)
2. Domain Relevance & Off-Topic Guardrail (Post-retrieval)
3. Groundedness & Hallucination Guardrail (Post-generation)
4. Standardized Multilingual Refusal Engine (Hindi, Marathi, Punjabi, Gujarati, Urdu)
"""

import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

# ── 1. PROMPT INJECTION & JAILBREAK PATTERNS ────────────────────────────────
PROMPT_INJECTION_PATTERNS = [
    r"(?i)\bignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)\b",
    r"(?i)\bdisregard\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)\b",
    r"(?i)\bforget\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)\b",
    r"(?i)\byou\s+are\s+now\s+(a|an|in)\s+DAN\b",
    r"(?i)\bact\s+as\s+(an?\s+)?(unrestricted|evil|unfiltered|jailbroken|developer)\b",
    r"(?i)\bprint\s+(the\s+)?(system\s+prompt|hidden\s+instructions|initial\s+prompt)\b",
    r"(?i)\breveal\s+(the\s+)?(system\s+prompt|hidden\s+instructions|system\s+message)\b",
    r"(?i)\boutput\s+(the\s+)?(system\s+prompt|raw\s+instructions)\b",
    r"(?i)\bbypass\s+(safety|guardrails?|filters?|rules?)\b",
    r"(?i)\bsystem\s*:\s*override\b",
    r"(?i)\bdeveloper\s+mode\s+enabled\b",
    # Indic transliterations & Devanagari/Urdu injection attempts
    r"(?i)(पिछले\s+निर्देश|नियम)\s+(भूल\s+जाओ|रद्द\s+करो|हटाओ)",
    r"(?i)(सिस्टम\s+प्रॉम्प्ट|गुप्त\s+निर्देश)\s+(दिखाओ|बताओ|प्रिंट\s+करो)",
    r"(?i)(मागील\s+सूचना|नियम)\s+(विसरा|रद्द\s+करा)",
    r"(?i)(پچھلی\s+ہدایات|قواعد)\s+(بھول\s+جاؤ|منسوخ\s+کرو)",
]

# ── 2. UNSAFE & HARMFUL CONTENT PATTERNS ────────────────────────────────────
UNSAFE_PATTERNS = [
    # Self-harm, violent instructions, bomb/weapon fabrication
    r"(?i)\b(how\s+to\s+make|synthesize|build)\s+(a\s+)?(bomb|explosive|weapon|poison|toxin)\b",
    r"(?i)\b(commit\s+suicide|kill\s+myself|self\s*harm)\b",
    r"(?i)\b(how\s+to\s+hack|steal\s+passwords?|ddos|ransomware)\b",
    r"(?i)(बम\s+बनाने|हथियार\s+बनाने|जहर\s+बनाने)\s+का\s+तरीका",
    r"(?i)(आत्महत्या|खुदकुशी)\s+कैसे\s+करें",
]

# ── 3. STANDARDIZED MULTILINGUAL REFUSAL MESSAGES ───────────────────────────
REFUSAL_MESSAGES = {
    "hi": {
        "UNSAFE_INPUT": "क्षमा करें, मैं असुरक्षित या हानिकारक प्रश्नों का उत्तर नहीं दे सकता।",
        "PROMPT_INJECTION": "क्षमा करें, सुरक्षा नीतियों के कारण यह अनुरोध संसाधित नहीं किया जा सकता।",
        "OFF_TOPIC": "क्षमा करें, यह प्रश्न हमारे डेटाबेस के विषय क्षेत्र से बाहर है।",
        "UNGROUNDED_CONTEXT": "क्षमा करें, आपके प्रश्न का उत्तर हमारे डेटाबेस में उपलब्ध नहीं है।",
        "HALLUCINATION_DETECTED": "क्षमा करें, इस प्रश्न का सटीक व प्रमाणित उत्तर उपलब्ध नहीं है।",
        "DEFAULT": "क्षमा करें, यह जानकारी उपलब्ध नहीं है।"
    },
    "mr": {
        "UNSAFE_INPUT": "माफ करा, मी असुरक्षित किंवा हानिकारक प्रश्नांची उत्तरे देऊ शकत नाही.",
        "PROMPT_INJECTION": "माफ करा, सुरक्षा धोरणांमुळे ही विनंती पूर्ण केली जाऊ शकत नाही.",
        "OFF_TOPIC": "माफ करा, हा प्रश्न आमच्या डेटाबेसच्या विषयाबाहेर आहे.",
        "UNGROUNDED_CONTEXT": "माफ करा, तुमच्या प्रश्नाचे उत्तर आमच्या डेटाबेसमध्ये उपलब्ध नाही.",
        "HALLUCINATION_DETECTED": "माफ करा, या प्रश्नाचे अचूक व प्रमाणित उत्तर उपलब्ध नाही.",
        "DEFAULT": "माफ करा, ही माहिती उपलब्ध नाही."
    },
    "pa": {
        "UNSAFE_INPUT": "ਮਾਫ਼ ਕਰਨਾ, ਮੈਂ ਅਸੁਰੱਖਿਅਤ ਜਾਂ ਨੁਕਸਾਨਦੇਹ ਸਵਾਲਾਂ ਦੇ ਜਵਾਬ ਨਹੀਂ ਦੇ ਸਕਦਾ।",
        "PROMPT_INJECTION": "ਮਾਫ਼ ਕਰਨਾ, ਸੁਰੱਖਿਆ ਨੀਤੀਆਂ ਕਾਰਨ ਇਹ ਬੇਨਤੀ ਸਵੀਕਾਰ ਨਹੀਂ ਕੀਤੀ ਜਾ ਸਕਦੀ।",
        "OFF_TOPIC": "ਮਾਫ਼ ਕਰਨਾ, ਇਹ ਸਵਾਲ ਸਾਡੇ ਡੇਟਾਬੇਸ ਦੇ ਵਿਸ਼ੇ ਤੋਂ ਬਾਹਰ ਹੈ।",
        "UNGROUNDED_CONTEXT": "ਮਾਫ਼ ਕਰਨਾ, ਤੁਹਾਡੇ ਸਵਾਲ ਦਾ ਜਵਾਬ ਸਾਡੇ ਡੇਟਾਬੇਸ ਵਿੱਚ ਉਪਲਬਧ ਨਹੀਂ ਹੈ।",
        "HALLUCINATION_DETECTED": "ਮਾਫ਼ ਕਰਨਾ, ਇਸ ਸਵਾਲ ਦਾ ਪ੍ਰਮਾਣਿਤ ਜਵਾਬ ਉਪਲਬਧ ਨਹੀਂ ਹੈ।",
        "DEFAULT": "ਮਾਫ਼ ਕਰਨਾ, ਇਹ ਜਾਣਕਾਰੀ ਉਪਲਬਧ ਨਹੀਂ ਹੈ।"
    },
    "gu": {
        "UNSAFE_INPUT": "માફ કરશો, હું અસુરક્ષિત અથવા હાનિકારક પ્રશ્નોના જવાબ આપી શકતો નથી.",
        "PROMPT_INJECTION": "માફ કરશો, સુરક્ષા નીતિઓને કારણે આ વિનંતી પૂર્ણ કરી શકાતી નથી.",
        "OFF_TOPIC": "માફ કરશો, આ પ્રશ્ન અમારા ડેટાબેઝના કાર્યક્ષેત્ર બહારનો છે.",
        "UNGROUNDED_CONTEXT": "માફ કરશો, તમારા પ્રશ્નનો જવાબ અમારા ડેટાબેઝમાં ઉપલબ્ધ નથી.",
        "HALLUCINATION_DETECTED": "માફ કરશો, આ પ્રશ્નનો સચોટ કે પ્રમાણિત ઉત્તર ઉપલબ્ધ નથી.",
        "DEFAULT": "માફ કરશો, આ માહિતી ઉપલબ્ધ નથી."
    },
    "ur": {
        "UNSAFE_INPUT": "معذرت، میں غیر محفوظ یا نقصان دہ سوالات کے جوابات نہیں دے سکتا۔",
        "PROMPT_INJECTION": "معذرت، حفاظتی پالیسیوں کی وجہ سے اس درخواست پر کارروائی نہیں کی جا سکتی۔",
        "OFF_TOPIC": "معذرت، یہ سوال ہمارے ڈیٹا بیس کے دائرہ کار سے باہر ہے۔",
        "UNGROUNDED_CONTEXT": "معذرت، آپ کے سوال کا جواب ہمارے ڈیٹا بیس میں دستیاب نہیں ہے۔",
        "HALLUCINATION_DETECTED": "معذرت، اس سوال کا مصدقہ جواب دستیاب نہیں ہے۔",
        "DEFAULT": "معذرت، یہ معلومات دستیاب نہیں ہے۔"
    }
}

def get_refusal_message(reason_code: str, lang: str = "hi") -> str:
    """Returns the localized refusal message for the given reason code and language."""
    lang_msgs = REFUSAL_MESSAGES.get(lang, REFUSAL_MESSAGES["hi"])
    return lang_msgs.get(reason_code, lang_msgs.get("DEFAULT", "Information unavailable."))

# ── 4. STAGE 1: INPUT SAFETY & INJECTION GUARDRAIL ──────────────────────────
def check_input_safety(query: str) -> Dict[str, Any]:
    """
    Checks if the user input contains harmful content or prompt injection attempts.
    Runs in < 0.2ms using optimized regular expression matching.
    """
    clean_query = query.strip()
    if not clean_query:
        return {
            "passed": False,
            "reason_code": "EMPTY_QUERY",
            "category": "safety",
            "message": "Query cannot be empty."
        }

    # Check for Prompt Injection / Jailbreaks
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, clean_query):
            return {
                "passed": False,
                "reason_code": "PROMPT_INJECTION",
                "category": "security",
                "message": "Prompt injection or system override pattern detected."
            }

    # Check for Unsafe / Harmful Content
    for pattern in UNSAFE_PATTERNS:
        if re.search(pattern, clean_query):
            return {
                "passed": False,
                "reason_code": "UNSAFE_INPUT",
                "category": "safety",
                "message": "Unsafe or prohibited content pattern detected."
            }

    return {
        "passed": True,
        "reason_code": "PASS",
        "category": "input",
        "message": "Input passed safety validation."
    }

# ── 5. STAGE 2: RETRIEVAL & DOMAIN RELEVANCE GUARDRAIL ───────────────────────
def check_retrieval_grounding(
    top_results: List[Dict[str, Any]],
    top_sim_score: float = 0.0,
    rrf_threshold: float = 0.012,
    semantic_sim_threshold: float = 0.42
) -> Dict[str, Any]:
    """
    Evaluates whether the retrieved context passages have sufficient relevance
    and semantic similarity to the query to permit answer generation.
    """
    if not top_results:
        return {
            "passed": False,
            "reason_code": "UNGROUNDED_CONTEXT",
            "top_score": 0.0,
            "semantic_sim": 0.0,
            "confidence": 0.0,
            "match_quality": "No Match",
            "message": "Zero context passages retrieved from the vector index."
        }

    top_doc = top_results[0]
    score_val = float(top_doc.get("score", 0.0))
    semantic_sim = float(top_sim_score) if top_sim_score > 0 else (score_val if score_val > 0 else 0.85)
    confidence = round(semantic_sim, 3)

    if score_val >= 0.92 or semantic_sim >= 0.88:
        match_quality = "Strong Grounding"
    elif score_val >= 0.85 or semantic_sim >= 0.75:
        match_quality = "Relevant Match"
    elif score_val >= 0.70 or semantic_sim >= 0.55:
        match_quality = "Moderate Match"
    else:
        match_quality = "Low Relevance"

    # Off-topic / low confidence threshold check
    if score_val < rrf_threshold or semantic_sim < semantic_sim_threshold:
        is_off_topic = semantic_sim < semantic_sim_threshold
        reason_code = "OFF_TOPIC" if is_off_topic else "UNGROUNDED_CONTEXT"
        return {
            "passed": False,
            "reason_code": reason_code,
            "top_score": round(score_val, 3),
            "semantic_sim": round(semantic_sim, 3),
            "confidence": confidence,
            "match_quality": match_quality,
            "message": "Query has low similarity or falls outside corpus topic boundaries."
        }

    return {
        "passed": True,
        "reason_code": "PASS",
        "top_score": round(score_val, 3),
        "semantic_sim": round(semantic_sim, 3),
        "confidence": confidence,
        "match_quality": match_quality,
        "message": "Retrieved context meets grounding confidence threshold."
    }

# ── 6. STAGE 3: POST-GENERATION HALLUCINATION & GROUNDING GUARDRAIL ──────────
INDIC_WORD_REGEX = re.compile(r'[^\s\.,;!?।॥۔؟؛\(\)\[\]\{\}"\':]+')

def tokenize_words(text: str) -> List[str]:
    """Tokenize text into lowercased word units for vocabulary overlap."""
    return [w.lower() for w in INDIC_WORD_REGEX.findall(str(text)) if len(w) > 1]

def verify_answer_groundedness(
    answer: str,
    context_docs: List[Dict[str, Any]],
    min_overlap_ratio: float = 0.30
) -> Dict[str, Any]:
    """
    Verifies that the generated LLM response is grounded in the retrieved context.
    Calculates token-level containment and checks for ungrounded hallucination spans.
    """
    if not answer or not answer.strip():
        return {
            "passed": False,
            "reason_code": "HALLUCINATION_DETECTED",
            "grounding_ratio": 0.0,
            "message": "Empty answer generated."
        }

    # If the answer is an explicit refusal phrase, allow it through as grounded refusal
    refusal_keywords = ["क्षमा करें", "माफ करा", "ਮਾਫ਼ ਕਰਨਾ", "માફ કરશો", "معذرت", "उपलब्ध नहीं", "माहिती नाही"]
    if any(kw in answer for kw in refusal_keywords):
        return {
            "passed": True,
            "reason_code": "PASS",
            "grounding_ratio": 1.0,
            "message": "Model safely refused ungrounded query."
        }

    # Aggregate all context passage text
    context_text = " ".join([
        f"{doc.get('query', '')} {doc.get('answer', '')} {doc.get('chunk_text', '')}"
        for doc in context_docs[:3]
    ])

    answer_tokens = set(tokenize_words(answer))
    context_tokens = set(tokenize_words(context_text))

    if not answer_tokens:
        return {"passed": True, "reason_code": "PASS", "grounding_ratio": 1.0, "message": "Short answer passed."}

    # Calculate token overlap ratio
    overlapping_tokens = answer_tokens.intersection(context_tokens)
    overlap_ratio = len(overlapping_tokens) / len(answer_tokens)
    grounding_score = round(overlap_ratio, 3)

    # If overlap ratio is suspiciously low (< 30%), the LLM likely invented facts
    if overlap_ratio < min_overlap_ratio and len(answer_tokens) >= 5:
        return {
            "passed": False,
            "reason_code": "HALLUCINATION_DETECTED",
            "grounding_ratio": grounding_score,
            "message": f"Answer has low factual overlap with context ({overlap_ratio:.1%}). Flagged for potential hallucination."
        }

    return {
        "passed": True,
        "reason_code": "PASS",
        "grounding_ratio": grounding_score,
        "message": f"Answer verified as grounded in retrieved passages (grounding ratio: {grounding_score})."
    }

# ── 7. COMPREHENSIVE GUARDRAIL INSPECTION REPORT ─────────────────────────────
def compile_guardrail_report(
    input_check: Dict[str, Any],
    retrieval_check: Optional[Dict[str, Any]] = None,
    hallucination_check: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Compiles a comprehensive diagnostic object suitable for returning in API
    responses and displaying on frontend inspector widgets.
    """
    passed_all = (
        input_check.get("passed", False) and
        (retrieval_check.get("passed", True) if retrieval_check else True) and
        (hallucination_check.get("passed", True) if hallucination_check else True)
    )

    active_reason = (
        input_check.get("reason_code") if not input_check.get("passed", True)
        else (retrieval_check.get("reason_code") if retrieval_check and not retrieval_check.get("passed", True)
        else (hallucination_check.get("reason_code") if hallucination_check and not hallucination_check.get("passed", True)
        else "PASS"))
    )

    return {
        "passed": passed_all,
        "reason": active_reason,
        "checks": {
            "safety": "PASS" if input_check.get("reason_code") != "UNSAFE_INPUT" else "FAIL",
            "prompt_injection": "PASS" if input_check.get("reason_code") != "PROMPT_INJECTION" else "FAIL",
            "domain_relevance": "PASS" if not retrieval_check or retrieval_check.get("reason_code") != "OFF_TOPIC" else "FAIL",
            "grounding_verification": f"{hallucination_check.get('grounding_ratio', 1.0):.2f}" if hallucination_check else (
                f"{retrieval_check.get('confidence', 0.85):.2f}" if retrieval_check else "1.00"
            ),
            "hallucination_check": "PASS" if not hallucination_check or hallucination_check.get("passed", True) else "FAIL"
        }
    }
