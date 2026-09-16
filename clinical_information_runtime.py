"""Klinik Chong V17 clinical information extraction runtime.

Extracts user-reported symptoms and duration, accepts only explicitly stated severity,
and generates a read-only English clinical note for booking.
"""

from __future__ import annotations

import json
import os
import re
import threading
from typing import Any, Dict, List, Optional

_MODEL = None
_MODEL_LOCK = threading.Lock()


def _openai_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if key:
        return key
    try:
        from google.colab import userdata
        key = str(userdata.get("OPENAI_API_KEY") or "").strip()
    except Exception:
        key = ""
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY not found. In Colab, load the OPENAI_API_KEY secret "
            "into os.environ before starting model_server_ver17.py."
        )
    return key


def _model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("langchain-openai is required for clinical extraction.") from exc
        _MODEL = ChatOpenAI(
            api_key=_openai_api_key(),
            model="gpt-4.1-mini",
            temperature=0,
        )
        return _MODEL


def _strip_json_fence(raw: str) -> str:
    value = str(raw or "").strip()
    value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.I)
    value = re.sub(r"\s*```$", "", value)
    return value.strip()


def explicit_severity_from_text(text: str) -> Optional[int]:
    """Extract only an explicit 1-10 severity value; never infer from adjectives."""
    value = str(text or "")
    patterns = [
        r"(?i)\b(?:severity|pain|level|score|tahap)\s*(?:is|=|:)?\s*(10|[1-9])(?:\s*/\s*10)?\b",
        r"(?:严重程度|嚴重程度|程度|疼痛程度|痛感)\s*(?:是|为|為|=|:)?\s*(10|[1-9])(?:\s*/\s*10)?",
        r"(?<!\d)(10|[1-9])\s*/\s*10(?!\d)",
    ]
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            return int(match.group(1))
    return None


def extract_clinical_information(user_input: str) -> Dict[str, Any]:
    prompt = f"""Extract only the user's explicitly reported clinical information.

Return JSON only in this format:
{{"symptoms": [], "duration": null}}

Rules:
- Extract only symptoms explicitly reported by the user.
- Do not diagnose or infer a condition.
- Do not add medical information that the user did not mention.
- Normalize duration into simple English when possible.
- Examples: 三天 -> 3 days; dua hari -> 2 days; semalam -> 1 day; 一星期 -> 1 week.
- If symptoms are missing, return an empty list.
- If duration is missing, return null.
- Support Chinese, Malay, English, Pinyin and mixed-language input.

User message:
{user_input}
"""
    response = _model().invoke(prompt)
    raw = str(getattr(response, "content", response)).strip()
    try:
        data = json.loads(_strip_json_fence(raw))
    except Exception as exc:
        raise RuntimeError(f"Clinical extraction returned invalid JSON: {raw}") from exc

    symptoms = data.get("symptoms", []) if isinstance(data, dict) else []
    if not isinstance(symptoms, list):
        symptoms = []
    symptoms = [str(item).strip() for item in symptoms if str(item).strip()]

    duration = data.get("duration") if isinstance(data, dict) else None
    if duration is not None:
        duration = str(duration).strip() or None

    return {
        "symptoms": symptoms,
        "duration": duration,
        "severity": explicit_severity_from_text(user_input),
        "parse_error": False,
    }


def generate_clinical_note(
    symptoms: List[str],
    duration: str,
    severity: int,
    descriptions: Optional[List[str]] = None,
    rating: Optional[int] = None,
) -> str:
    if not symptoms or not duration or severity is None:
        raise ValueError("Symptoms, duration and severity are required before generating a clinical note.")
    if not 0 <= int(severity) <= 10:
        raise ValueError("Severity must be between 0 and 10.")

    descriptions = [str(item).strip() for item in (descriptions or []) if str(item).strip()]
    prompt = f"""Generate a concise English clinical booking note using only the user-reported information below.
Do not diagnose. Do not infer any condition. Do not add information that was not provided.
Mention all reported symptoms, duration and severity. Preserve useful user-reported details without inventing facts.

Symptoms: {', '.join(symptoms)}
Duration: {duration}
Severity: {severity}/10
User-reported details: {' | '.join(descriptions) if descriptions else 'None'}
"""
    response = _model().invoke(prompt)
    note = str(getattr(response, "content", response)).strip()
    if rating is not None:
        if not 1 <= int(rating) <= 5:
            raise ValueError("Rating must be between 1 and 5.")
        note = note.rstrip() + f"\nrate to clinic: {int(rating)} stars"
    return note

