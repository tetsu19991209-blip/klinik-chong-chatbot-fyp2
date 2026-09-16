"""
Klinik Chong Chatbot V22 model inference server.

Loads:
1. mBert_language_detect_model/best_model
2. pinyin2hanzi_hmm_model (joblib HMM bundle)
3. models/xlmr (eight-class intention classifier)

The server also hosts the V22 frontend, so open http://127.0.0.1:8000/
instead of opening the HTML with file://.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sqlite3
import threading
import unicodedata
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import quote as url_quote

try:
    from twilio.rest import Client as TwilioClient
except ImportError:  # The non-emergency chatbot can still start without Twilio.
    TwilioClient = None

try:
    import torch
except ImportError:  # Booking-only/mock mode must work without model packages.
    torch = None
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
except ImportError:  # Environment variables can still be supplied by the host.
    load_dotenv = None


if load_dotenv is not None:
    load_dotenv()


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_DRIVE_ROOT = Path("/content/drive/MyDrive/FYP2/Trained Model")
DEFAULT_DATABASE_PATH = Path(
    "/content/drive/MyDrive/FYP2/klinik_chong_database/klinik_chong.db"
)

# Twilio is optional. Public source code must never contain account credentials
# or phone numbers, so all values are supplied by the deployer at runtime.
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "").strip()
EMERGENCY_CONTACT_NUMBER = os.getenv("EMERGENCY_CONTACT_NUMBER", "").strip()
TWILIO_TRIAL_VOICE_TEMPLATE_URL = (
    "https://webhooks.twilio.com/v1/Voice/Template/voice_text_to_speech"
)
TERMINAL_CALL_STATUSES = {"completed", "canceled", "busy", "failed", "no-answer"}
_emergency_call_lock = threading.Lock()
_emergency_call_sids: set[str] = set()

MBERT_RELATIVE_PATH = Path("mBert_language_detect_model/best_model")
HMM_RELATIVE_PATH = Path("pinyin2hanzi_hmm_model")
HMM_MODEL_FILENAME = "pinyin2hanzi_hmm_model.joblib"
HMM_INFERENCE_FILENAME = "hmm_inference.py"
INTENT_RELATIVE_PATH = Path("xlmr")
INTENT_LABELS = [
    "description", "ask_info", "unrelated", "booking",
    "cancel_booking", "check_booking", "reschedule_booking",
    "check_availability_query",
]


def deterministic_intention_override(user_input: str) -> Optional[Dict[str, Any]]:
    """Return high-precision conversational intents before XLM-R inference."""
    text = unicodedata.normalize("NFKC", str(user_input or "")).lower().strip()
    clinic_hours_pattern = re.compile(
        r"(?:\b(?:bila|pukul\s+berapa|jam\s+berapa)\b[^?!.]{0,55}\b(?:buka|tutup)\b|"
        r"\b(?:waktu|masa|jam)\s+(?:operasi|buka|tutup)\b|"
        r"\boperating\s+hours?\b|\bopening\s+hours?\b|"
        r"\bwhen\s+(?:does|is|are)?\s*(?:the\s+)?(?:clinic|klinik)?\s*(?:open|close)\b|"
        r"营业时间|營業時間|开放时间|開放時間|几点开门|幾點開門|"
        r"几点营业|幾點營業|什么时候开|什麼時候開)",
        flags=re.IGNORECASE,
    )
    if clinic_hours_pattern.search(text):
        return {
            "intention": "ask_info",
            "confidence": 1.0,
            "uncertain": False,
            "probabilities": {"ask_info": 1.0},
            "source": "deterministic_clinic_hours_rule",
        }
    capability_pattern = re.compile(
        r"(?:\b(?:what\s+can\s+(?:you|this\s+chatbot)\s+do|"
        r"what\s+are\s+your\s+(?:functions|features)|"
        r"chatbot\s+(?:functions?|features?|capabilit(?:y|ies)))\b|"
        r"\b(?:awak|kamu|chatbot\s+ini)\s+boleh\s+(?:buat|bantu)\s+apa\b|"
        r"\bapa\s+(?:yang\s+)?boleh\s+(?:awak|kamu|chatbot\s+ini)\s+(?:buat|bantu)\b|"
        r"你(?:可以|能)做什么|你有什么功能|聊天机器人(?:可以|能)做什么|机器人有什么功能)",
        flags=re.IGNORECASE,
    )
    doctor_directory_pattern = re.compile(
        r"(?:\b(?:how\s+many\s+doctors?|which\s+doctors?|"
        r"who\s+are\s+the\s+doctors?|doctor\s+list)\b|"
        r"\b(?:berapa\s+(?:orang\s+)?doktor|doktor\s+ada\s+berapa|"
        r"siapa\s+(?:sahaja\s+)?doktor|senarai\s+doktor)\b|"
        r"诊所(?:有)?多少(?:个|位)?医生|診所(?:有)?多少(?:個|位)?醫生|"
        r"(?:有|有哪些|都有)(?:什么|什麼)?(?:医生|醫生)|医生名单|醫生名單)",
        flags=re.IGNORECASE,
    )
    if capability_pattern.search(text) or doctor_directory_pattern.search(text):
        return {
            "intention": "ask_info",
            "confidence": 1.0,
            "uncertain": False,
            "probabilities": {"ask_info": 1.0},
            "source": "deterministic_clinic_information_rule",
        }
    return None

LABEL_ORDER = ["malay", "chinese", "pinyin", "other"]
DEFAULT_ID2LABEL = {0: "malay", 1: "chinese", 2: "pinyin", 3: "other"}


def first_existing_path(candidates: List[Path]) -> Path:
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def resolve_model_path(env_name: str, relative_path: Path) -> Path:
    explicit = os.getenv(env_name)
    if explicit:
        return Path(explicit).expanduser().resolve()

    candidates = [
        DEFAULT_DRIVE_ROOT / relative_path,
        APP_ROOT / "models" / relative_path,
        APP_ROOT.parent / "models" / relative_path,
    ]
    return first_existing_path(candidates)


def resolve_database_path() -> Path:
    """Locate the live Klinik Chong database without creating a copy."""
    explicit = os.getenv("KLINIK_CHONG_DB_PATH")
    if explicit:
        return Path(explicit).expanduser().resolve()

    candidates = [
        DEFAULT_DATABASE_PATH,
        APP_ROOT.parent / "klinik_chong_database" / "klinik_chong.db",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0]


def open_database_read_only() -> sqlite3.Connection:
    """Open SQLite in read-only/query-only mode for Developer View."""
    database_path = resolve_database_path()
    if not database_path.is_file():
        raise HTTPException(
            status_code=503,
            detail=(
                "Klinik Chong database was not found. Run the chatbot server in "
                "the mounted FYP2 folder or set KLINIK_CHONG_DB_PATH."
            ),
        )

    connection = sqlite3.connect(
        f"{database_path.as_uri()}?mode=ro",
        uri=True,
        timeout=5,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON;")
    connection.execute("PRAGMA foreign_keys = ON;")
    return connection


def quote_sql_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def database_table_names(connection: sqlite3.Connection) -> List[str]:
    rows = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name;
        """
    ).fetchall()
    return [str(row["name"]) for row in rows]


def json_safe_database_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return f"<BLOB: {len(value)} bytes>"
    return str(value)


def normalize_label(label: Any) -> str:
    raw = str(label).strip().lower()
    aliases = {
        "0": "malay",
        "1": "chinese",
        "2": "pinyin",
        "3": "other",
        "label_0": "malay",
        "label_1": "chinese",
        "label_2": "pinyin",
        "label_3": "other",
        "hanzi": "chinese",
        "zh": "chinese",
        "ms": "malay",
    }
    return aliases.get(raw, raw if raw in LABEL_ORDER else "other")


class LanguageDetectRequest(BaseModel):
    tokens: List[str] = Field(min_length=1, max_length=256)
    sentence: str = ""


class PinyinToHanziRequest(BaseModel):
    inputs: List[str] = Field(min_length=1, max_length=128)


class IntentionClassifyRequest(BaseModel):
    user_input: str = Field(min_length=1, max_length=1000)


class RagQueryRequest(BaseModel):
    user_input: str = Field(min_length=1, max_length=1000)
    response_language: str = "chinese"
    past_queries: str = ""
    intent_hint: str = "ask_info"
    conversation_state: Dict[str, Any] = Field(default_factory=dict)
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)


class ConversationStateRequest(BaseModel):
    user_input: str = Field(min_length=1, max_length=1000)
    conversation_state: Dict[str, Any] = Field(default_factory=dict)
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)


class EmotionAdjustRequest(BaseModel):
    user_input: str = Field(min_length=1, max_length=1000)
    base_response: str = Field(min_length=1, max_length=6000)
    response_language: str = "malay"
    conversation_state: Dict[str, Any] = Field(default_factory=dict)
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    safety_critical: bool = False


class ClinicalExtractRequest(BaseModel):
    user_input: str = Field(min_length=1, max_length=1000)


class ClinicalNoteRequest(BaseModel):
    symptoms: List[str] = Field(default_factory=list)
    duration: Optional[str] = None
    severity: Optional[int] = Field(default=None, ge=0, le=10)
    descriptions: List[str] = Field(default_factory=list)
    rating: Optional[int] = Field(default=None, ge=1, le=5)


class ClinicalRatingRequest(BaseModel):
    appointment_id: str = Field(min_length=1, max_length=32)
    rating: int = Field(ge=1, le=5)


class EmergencyDetectRequest(BaseModel):
    user_input: str = Field(min_length=1, max_length=1000)


class EmergencyCallRequest(BaseModel):
    trigger: str = Field(default="explicit_ambulance_request", max_length=64)
    severity: Optional[int] = Field(default=None, ge=0, le=10)


def detect_emergency_call_request(user_input: str) -> Dict[str, Any]:
    """Detect ambulance requests or suicide/self-harm risk before intent routing."""
    text = unicodedata.normalize("NFKC", str(user_input or "")).strip().lower()

    suicide_patterns = [
        (
            "chinese_suicide_risk",
            r"(?:自杀|自殺|轻生|輕生|不想(?:再)?活(?:了|下去)?|活不下去|结束(?:我)?(?:的)?生命|結束(?:我)?(?:的)?生命|想去死|死掉算了|割腕|跳楼|跳樓)",
        ),
        (
            "malay_suicide_risk",
            r"\b(?:bunuh\s+diri|nak\s+mati|mahu\s+mati|mau\s+mati|tak\s+nak\s+hidup|tidak\s+mahu\s+hidup|tamatkan\s+hidup|cederakan\s+diri|kelar\s+pergelangan|terjun\s+(?:dari\s+)?bangunan)\b",
        ),
        (
            "english_suicide_risk",
            r"\b(?:suicid(?:e|al)|kill\s+myself|end\s+my\s+life|want\s+to\s+die|wanna\s+die|self[ -]?harm)\b",
        ),
        (
            "pinyin_suicide_risk",
            r"\b(?:zi\s*sha|bu\s*xiang\s*huo|xiang\s*si|jie\s*shu\s*sheng\s*ming)\b",
        ),
    ]
    for signal, pattern in suicide_patterns:
        if re.search(pattern, text, flags=re.I):
            return {"is_emergency": True, "signal": signal, "reason": "suicide_self_harm_risk"}

    negated_patterns = [
        r"(?:不要|不用|不需要|无需|別|别).{0,12}(?:叫|呼叫|拨打|撥打|call)?\s*(?:救护车|救護車|999)",
        r"\b(?:do\s*not|don't|dont|no\s+need|tak\s+perlu|tidak\s+perlu|jangan|tak\s+payah)\b.{0,24}\b(?:call|panggil|telefon|hubungi)?\s*(?:ambulance|ambulans|999)\b",
    ]
    if any(re.search(pattern, text, flags=re.I) for pattern in negated_patterns):
        return {"is_emergency": False, "signal": None, "reason": "negated_request"}

    positive_patterns = [
        (
            "chinese_ambulance_request",
            r"(?:我要|我想|需要|请|請|帮我|幫我|快|赶快|趕快|立刻|马上|馬上).{0,14}(?:叫|呼叫|打|拨打|撥打|联系|聯繫).{0,8}(?:救护车|救護車|999)",
        ),
        (
            "malay_english_ambulance_request",
            r"\b(?:call|phone|ring|contact|panggil|telefon|hubungi)\b.{0,18}\b(?:an?\s+)?(?:ambulance|ambulans|999)\b",
        ),
        (
            "malay_english_need_ambulance",
            r"\b(?:nak|mahu|mau|perlu|tolong|please|need|want)\b.{0,28}\b(?:ambulance|ambulans)\b",
        ),
        (
            "pinyin_ambulance_request",
            r"\b(?:jiao|hu\s*jiao|da|bo\s*da|call)\b.{0,18}\bjiu\s*hu\s*che\b",
        ),
    ]
    for signal, pattern in positive_patterns:
        if re.search(pattern, text, flags=re.I):
            return {"is_emergency": True, "signal": signal, "reason": "explicit_call_request"}
    return {"is_emergency": False, "signal": None, "reason": "no_explicit_call_request"}


def get_twilio_client():
    if TwilioClient is None:
        raise HTTPException(status_code=503, detail="Twilio is not installed. Run pip install twilio.")

    auth_token = str(os.getenv("TWILIO_AUTH_TOKEN", "") or "").strip()
    if not auth_token:
        try:
            from google.colab import userdata

            auth_token = str(userdata.get("TWILIO_AUTH_TOKEN") or "").strip()
        except Exception:
            auth_token = ""
    if not auth_token:
        raise HTTPException(
            status_code=503,
            detail="TWILIO_AUTH_TOKEN is missing from Colab Secrets or environment variables.",
        )
    missing = [
        name
        for name, value in (
            ("TWILIO_ACCOUNT_SID", TWILIO_ACCOUNT_SID),
            ("TWILIO_FROM_NUMBER", TWILIO_FROM_NUMBER),
            ("EMERGENCY_CONTACT_NUMBER", EMERGENCY_CONTACT_NUMBER),
        )
        if not value
    ]
    if missing:
        raise HTTPException(
            status_code=503,
            detail=f"Missing Twilio configuration: {', '.join(missing)}.",
        )
    return TwilioClient(TWILIO_ACCOUNT_SID, auth_token)


class ModelRuntime:
    def __init__(self, mock_mode: bool = False) -> None:
        self.mock_mode = mock_mode
        if torch is None:
            self.device = "cpu"
            self.dtype = "unavailable"
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.dtype = (
                torch.bfloat16
                if self.device.type == "cuda" and torch.cuda.is_bf16_supported()
                else torch.float16
                if self.device.type == "cuda"
                else torch.float32
            )

        self.mbert_path = resolve_model_path("MBERT_MODEL_PATH", MBERT_RELATIVE_PATH)
        self.hmm_path = resolve_model_path("PINYIN2HANZI_HMM_PATH", HMM_RELATIVE_PATH)
        self.intent_path = resolve_model_path("XLMR_INTENT_MODEL_PATH", INTENT_RELATIVE_PATH)

        self.mbert_tokenizer = None
        self.mbert_model = None
        self.hmm_module = None
        self.hmm_bundle = None
        self.intent_tokenizer = None
        self.intent_model = None

        self.mbert_error: Optional[str] = None
        self.hmm_error: Optional[str] = None
        self.intent_error: Optional[str] = None

        self._mbert_lock = threading.Lock()
        self._hmm_lock = threading.Lock()
        self._intent_lock = threading.Lock()

    def model_info(self) -> Dict[str, Any]:
        runtime_mode = "mock" if self.mock_mode else "real"
        return {
            "runtime_mode": runtime_mode,
            "mock_mode": self.mock_mode,
            "device": str(self.device),
            "dtype": str(self.dtype).replace("torch.", ""),
            "mbert": {
                "model_name": (
                    "mock_mBERT_for_UI_test_only"
                    if self.mock_mode
                    else "mBert_language_detect_model/best_model"
                ),
                "path": str(self.mbert_path),
                "exists": self.mbert_path.is_dir(),
                # `loaded` now means the REAL model is actually in memory.
                "loaded": self.mbert_model is not None,
                "mock_available": self.mock_mode,
                "error": self.mbert_error,
            },
            "pinyin2hanzi_hmm": {
                "model_name": (
                    "mock_Pinyin2Hanzi_HMM_for_UI_test_only"
                    if self.mock_mode
                    else "pinyin2hanzi_hmm_model"
                ),
                "path": str(self.hmm_path),
                "model_file": str(self.hmm_path / HMM_MODEL_FILENAME),
                "inference_file": str(self.hmm_path / HMM_INFERENCE_FILENAME),
                "exists": (self.hmm_path / HMM_MODEL_FILENAME).is_file() and (self.hmm_path / HMM_INFERENCE_FILENAME).is_file(),
                # Do not report a mock implementation as a loaded HMM.
                "loaded": self.hmm_bundle is not None,
                "mock_available": self.mock_mode,
                "error": self.hmm_error,
            },
            "xlmr_intention": {
                "model_name": "xlm-roberta-base-intention-classifier",
                "path": str(self.intent_path),
                "exists": self.intent_path.is_dir(),
                "loaded": self.intent_model is not None,
                "error": self.intent_error,
            },
        }

    def load_intention_model(self) -> None:
        if self.intent_model is not None:
            return
        with self._intent_lock:
            if self.intent_model is not None:
                return
            if torch is None:
                raise RuntimeError("PyTorch is required for XLM-R intention inference.")
            if not self.intent_path.is_dir():
                self.intent_error = f"XLM-R intention model not found: {self.intent_path}"
                raise FileNotFoundError(self.intent_error)
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                self.intent_tokenizer = AutoTokenizer.from_pretrained(
                    self.intent_path, local_files_only=True, use_fast=True
                )
                self.intent_model = AutoModelForSequenceClassification.from_pretrained(
                    self.intent_path, local_files_only=True
                ).to(self.device)
                self.intent_model.eval()
                self.intent_error = None
            except Exception as exc:
                self.intent_error = f"{type(exc).__name__}: {exc}"
                self.intent_tokenizer = None
                self.intent_model = None
                raise

    def predict_intention(self, user_input: str) -> Dict[str, Any]:
        self.load_intention_model()
        assert self.intent_tokenizer is not None and self.intent_model is not None
        encoded = self.intent_tokenizer(
            unicodedata.normalize("NFKC", user_input).strip(),
            return_tensors="pt", truncation=True, max_length=64
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = self.intent_model(**encoded).logits[0]
            probabilities = torch.softmax(logits.float(), dim=-1)
        predicted_id = int(torch.argmax(probabilities).item())
        raw_label = self.intent_model.config.id2label.get(predicted_id, INTENT_LABELS[predicted_id])
        label = str(raw_label).strip().lower()
        if label not in INTENT_LABELS:
            label = INTENT_LABELS[predicted_id]
        confidence = float(probabilities[predicted_id].item())
        scores = {
            self.intent_model.config.id2label.get(index, INTENT_LABELS[index]): float(value)
            for index, value in enumerate(probabilities.detach().cpu().tolist())
        }
        return {
            "intention": label,
            "confidence": confidence,
            "uncertain": confidence < 0.60,
            "scores": scores,
            "model": "XLM-R",
        }

    def load_mbert(self) -> None:
        if self.mock_mode or self.mbert_model is not None:
            return

        with self._mbert_lock:
            if torch is None:
                raise RuntimeError("PyTorch is not installed. Start with --mock for Booking Layer testing.")
            if self.mbert_model is not None:
                return
            if not self.mbert_path.is_dir():
                self.mbert_error = f"mBERT model folder not found: {self.mbert_path}"
                raise FileNotFoundError(self.mbert_error)

            try:
                from transformers import AutoModelForTokenClassification, AutoTokenizer

                self.mbert_tokenizer = AutoTokenizer.from_pretrained(
                    self.mbert_path,
                    use_fast=True,
                    local_files_only=True,
                )
                self.mbert_model = AutoModelForTokenClassification.from_pretrained(
                    self.mbert_path,
                    torch_dtype=self.dtype,
                    local_files_only=True,
                ).to(self.device)
                self.mbert_model.eval()
                self.mbert_error = None
            except Exception as exc:
                self.mbert_error = f"{type(exc).__name__}: {exc}"
                self.mbert_model = None
                self.mbert_tokenizer = None
                raise

    def load_hmm(self) -> None:
        if self.mock_mode or self.hmm_bundle is not None:
            return

        with self._hmm_lock:
            if self.hmm_bundle is not None:
                return

            model_file = self.hmm_path / HMM_MODEL_FILENAME
            inference_file = self.hmm_path / HMM_INFERENCE_FILENAME

            if not self.hmm_path.is_dir():
                self.hmm_error = f"Pinyin2Hanzi HMM folder not found: {self.hmm_path}"
                raise FileNotFoundError(self.hmm_error)
            if not model_file.is_file():
                self.hmm_error = f"Pinyin2Hanzi HMM model file not found: {model_file}"
                raise FileNotFoundError(self.hmm_error)
            if not inference_file.is_file():
                self.hmm_error = f"Pinyin2Hanzi HMM inference file not found: {inference_file}"
                raise FileNotFoundError(self.hmm_error)

            try:
                spec = importlib.util.spec_from_file_location(
                    "klinik_chong_pinyin2hanzi_hmm_inference",
                    inference_file,
                )
                if spec is None or spec.loader is None:
                    raise ImportError(f"Unable to import HMM inference module: {inference_file}")

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                bundle = module.load_model(model_file)

                self.hmm_module = module
                self.hmm_bundle = bundle
                self.hmm_error = None
            except Exception as exc:
                self.hmm_error = f"{type(exc).__name__}: {exc}"
                self.hmm_module = None
                self.hmm_bundle = None
                raise

    def _id2label(self) -> Dict[int, str]:
        if self.mbert_model is None:
            return DEFAULT_ID2LABEL
        configured = getattr(self.mbert_model.config, "id2label", None) or {}
        mapping: Dict[int, str] = {}
        for key, value in configured.items():
            try:
                mapping[int(key)] = normalize_label(value)
            except (TypeError, ValueError):
                continue
        return mapping or DEFAULT_ID2LABEL

    def predict_languages(self, tokens: List[str], sentence: str = "") -> List[Dict[str, Any]]:
        clean_tokens = [unicodedata.normalize("NFKC", str(token)).strip() for token in tokens]
        if self.mock_mode:
            return self._mock_language_predictions(clean_tokens, sentence)

        self.load_mbert()
        assert self.mbert_tokenizer is not None
        assert self.mbert_model is not None

        encoded = self.mbert_tokenizer(
            clean_tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        word_ids = encoded.word_ids(batch_index=0)
        model_inputs = {
            key: value.to(self.device)
            for key, value in encoded.items()
            if key in {"input_ids", "attention_mask", "token_type_ids"}
        }

        with torch.inference_mode():
            logits = self.mbert_model(**model_inputs).logits[0]
            probabilities = torch.softmax(logits.float(), dim=-1).cpu()

        # Match the training/evaluation reconstruction: only the FIRST subword
        # of each original word is used for the word-level language label.
        id2label = self._id2label()
        first_subword_probabilities: List[Optional[torch.Tensor]] = [
            None for _ in clean_tokens
        ]
        for subtoken_index, word_index in enumerate(word_ids):
            if word_index is None or word_index >= len(first_subword_probabilities):
                continue
            if first_subword_probabilities[word_index] is None:
                first_subword_probabilities[word_index] = probabilities[subtoken_index]

        predictions: List[Dict[str, Any]] = []
        for token, token_probabilities in zip(clean_tokens, first_subword_probabilities):
            if token_probabilities is None:
                word_probabilities = torch.zeros(len(LABEL_ORDER), dtype=torch.float32)
                word_probabilities[3] = 1.0
            else:
                word_probabilities = token_probabilities

            predicted_id = int(torch.argmax(word_probabilities).item())
            predicted_language = normalize_label(id2label.get(predicted_id, predicted_id))
            confidence = float(word_probabilities[predicted_id].item())

            probability_map: Dict[str, float] = {label: 0.0 for label in LABEL_ORDER}
            for class_id, probability in enumerate(word_probabilities.tolist()):
                label = normalize_label(id2label.get(class_id, class_id))
                if label in probability_map:
                    probability_map[label] += float(probability)

            predictions.append(
                {
                    "token": token,
                    "predicted_id": predicted_id,
                    "language": predicted_language,
                    "confidence": confidence,
                    "probabilities": probability_map,
                }
            )

        return predictions

    def _hmm_pinyin_vocabulary(self) -> set[str]:
        if self.hmm_bundle is None:
            return set()
        model = self.hmm_bundle.get("model", {})
        vocabulary = model.get("pinyin_vocabulary")
        if vocabulary:
            return {str(item).strip().lower() for item in vocabulary if str(item).strip()}
        candidates = model.get("pinyin_candidates", {})
        return {str(item).strip().lower() for item in candidates.keys() if str(item).strip()}

    @staticmethod
    def _prefer_pinyin_segmentation(
        current: Optional[List[str]],
        candidate: List[str],
    ) -> List[str]:
        if current is None:
            return candidate
        if len(candidate) < len(current):
            return candidate
        if len(candidate) > len(current):
            return current
        for candidate_part, current_part in zip(candidate, current):
            if len(candidate_part) > len(current_part):
                return candidate
            if len(candidate_part) < len(current_part):
                return current
        return current

    def _segment_connected_pinyin(
        self,
        word: str,
        vocabulary: set[str],
    ) -> Optional[List[str]]:
        clean = "".join(
            char for char in unicodedata.normalize("NFKC", word).lower()
            if "a" <= char <= "z"
        )
        if not clean:
            return None
        if clean in vocabulary:
            return [clean]

        best_from: List[Optional[List[str]]] = [None] * (len(clean) + 1)
        best_from[len(clean)] = []

        for start in range(len(clean) - 1, -1, -1):
            best: Optional[List[str]] = None
            for end in range(start + 1, len(clean) + 1):
                syllable = clean[start:end]
                if syllable not in vocabulary:
                    continue
                suffix = best_from[end]
                if suffix is None:
                    continue
                candidate = [syllable, *suffix]
                best = self._prefer_pinyin_segmentation(best, candidate)
            best_from[start] = best

        return best_from[0]

    def _prepare_hmm_input(self, value: str) -> Dict[str, Any]:
        vocabulary = self._hmm_pinyin_vocabulary()
        raw_parts = [
            part for part in unicodedata.normalize("NFKC", value).strip().lower().split()
            if part
        ]

        if not raw_parts:
            return {
                "success": False,
                "segments": [],
                "segmented_input": "",
                "status": "empty_input",
                "failed_part": None,
            }

        segments: List[str] = []
        used_segmentation = False
        for part in raw_parts:
            if part in vocabulary:
                segments.append(part)
                continue

            segmented = self._segment_connected_pinyin(part, vocabulary)
            if not segmented:
                return {
                    "success": False,
                    "segments": [],
                    "segmented_input": "",
                    "status": "segmentation_failed",
                    "failed_part": part,
                }

            used_segmentation = True
            segments.extend(segmented)

        return {
            "success": True,
            "segments": segments,
            "segmented_input": " ".join(segments),
            "status": "segmented" if used_segmentation else "already_segmented",
            "failed_part": None,
        }

    def convert_pinyin(self, inputs: List[str]) -> List[Dict[str, Any]]:
        clean_inputs = [
            unicodedata.normalize("NFKC", str(value)).strip().lower()
            for value in inputs
        ]
        if self.mock_mode:
            return self._mock_pinyin_conversions(clean_inputs)

        self.load_hmm()
        assert self.hmm_module is not None
        assert self.hmm_bundle is not None

        prepared = [self._prepare_hmm_input(value) for value in clean_inputs]
        conversions: List[Dict[str, Any]] = []

        for source, item in zip(clean_inputs, prepared):
            conversions.append(
                {
                    "input": source,
                    "segmented_input": item["segmented_input"],
                    "segmentation_status": item["status"],
                    "segmentation_source": "HMM_pinyin_vocabulary_fallback",
                    "output": "",
                    "status": item["status"] if not item["success"] else "pending",
                    "unknown_pinyin": [item["failed_part"]] if item.get("failed_part") else [],
                    "best_log_probability": None,
                    "model_name": "pinyin2hanzi_hmm_model",
                }
            )

        # Preserve sequence context, but never decode across an item whose
        # connected Pinyin could not be segmented.
        valid_runs: List[List[int]] = []
        current_run: List[int] = []
        for index, item in enumerate(prepared):
            if item["success"]:
                current_run.append(index)
            else:
                if current_run:
                    valid_runs.append(current_run)
                    current_run = []
        if current_run:
            valid_runs.append(current_run)

        for run in valid_runs:
            flat_syllables: List[str] = []
            item_lengths: List[int] = []
            for index in run:
                segments = prepared[index]["segments"]
                item_lengths.append(len(segments))
                flat_syllables.extend(segments)

            result = self.hmm_module.predict(flat_syllables, self.hmm_bundle)
            status = str(result.get("status", "unknown"))
            predicted_characters = list(result.get("predicted_characters") or [])
            unknown_pinyin = list(result.get("unknown_pinyin") or [])
            best_log_probability = result.get("best_log_probability")

            cursor = 0
            for index, length in zip(run, item_lengths):
                output = ""
                if status == "success" and length > 0:
                    output = "".join(predicted_characters[cursor:cursor + length])
                cursor += length

                conversions[index].update(
                    {
                        "output": unicodedata.normalize("NFKC", output).strip(),
                        "status": status,
                        "unknown_pinyin": unknown_pinyin,
                        "best_log_probability": best_log_probability,
                    }
                )

        return conversions

    @staticmethod
    def _mock_language_predictions(tokens: List[str], sentence: str) -> List[Dict[str, Any]]:
        pinyin = {"wo", "yao", "yisheng", "yi", "sheng", "ni", "men", "de", "hen", "lei"}
        malay = {"saya", "mahu", "boleh", "batuk", "ini", "doktor", "ke", "kan"}
        predictions = []
        for token in tokens:
            lower = token.lower()
            if any("\u4e00" <= char <= "\u9fff" for char in token):
                language, confidence = "chinese", 0.999
            elif lower in pinyin:
                language, confidence = "pinyin", 0.995
            elif lower in malay:
                language, confidence = "malay", 0.995
            else:
                language, confidence = "other", 0.990
            probabilities = {label: 0.001 for label in LABEL_ORDER}
            probabilities[language] = confidence
            predictions.append(
                {
                    "token": token,
                    "predicted_id": LABEL_ORDER.index(language),
                    "language": language,
                    "confidence": confidence,
                    "probabilities": probabilities,
                }
            )
        return predictions

    @staticmethod
    def _mock_pinyin_conversions(inputs: List[str]) -> List[Dict[str, Any]]:
        mapping = {
            "wo": "我",
            "yao": "要",
            "yisheng": "医生",
            "yi": "医",
            "sheng": "生",
            "hen": "很",
            "lei": "累",
            "ni": "你",
            "men": "们",
            "de": "的",
        }
        return [
            {
                "input": value,
                "output": mapping.get(value.lower(), value),
                "model_name": "mock_Pinyin2Hanzi_HMM_for_UI_test_only",
            }
            for value in inputs
        ]



def read_v17_frontend_config_value(key: str) -> str:
    """Read a simple quoted value from model_config_ver22.js."""
    config_path = APP_ROOT / "model_config_ver22.js"
    if not config_path.is_file():
        return ""
    try:
        config_text = config_path.read_text(encoding="utf-8")
    except OSError:
        return ""
    match = re.search(
        rf"\b{re.escape(key)}\s*:\s*[\"']([^\"']*)[\"']",
        config_text,
    )
    return match.group(1).strip() if match else ""


def booking_api_base() -> str:
    explicit = os.getenv("KLINIK_CHONG_BOOKING_API_BASE", "").strip()
    configured = read_v17_frontend_config_value("bookingApiBase")
    return (explicit or configured).rstrip("/")


def developer_api_key() -> str:
    explicit = os.getenv("KLINIK_CHONG_DEVELOPER_API_KEY", "").strip()
    configured = read_v17_frontend_config_value("developerApiKey")
    return explicit or configured


def proxy_booking_api_json(
    remote_path: str,
    *,
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
    developer: bool = False,
) -> Dict[str, Any]:
    """Server-side bridge to the Colab booking API.

    The browser talks only to localhost, so custom developer headers no longer
    trigger cross-origin preflight from the Developer Monitor.
    """
    base = booking_api_base()
    if not base:
        raise HTTPException(
            status_code=503,
            detail="bookingApiBase is empty in model_config_ver22.js.",
        )

    url = f"{base}{remote_path}"
    headers = {
        "Accept": "application/json",
        "User-Agent": "KlinikChong-V22-LocalProxy/1.0",
    }
    data = None
    if developer:
        key = developer_api_key()
        if not key:
            raise HTTPException(
                status_code=503,
                detail="developerApiKey is empty in model_config_ver22.js.",
            )
        headers["X-Developer-Key"] = key
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")

    request = urllib_request.Request(
        url,
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urllib_request.urlopen(request, timeout=20) as response:
            raw = response.read().decode("utf-8", errors="replace")
            if not raw:
                return {}
            try:
                return json.loads(raw)
            except json.JSONDecodeError as exc:
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "Booking API returned a non-JSON response. "
                        f"Configured API: {base}"
                    ),
                ) from exc
    except urllib_error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            body = {}
        detail = body.get("detail") or body.get("message") or f"Booking API returned HTTP {exc.code}."
        raise HTTPException(status_code=exc.code, detail=detail) from exc
    except urllib_error.URLError as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                f"Unable to reach booking API: {exc.reason}. "
                f"Configured API: {base}"
            ),
        ) from exc


class DoctorLeaveProxyRequest(BaseModel):
    d_id: str
    leave_date: str
    leave_type: Literal["ANNUAL", "MEDICAL", "EMERGENCY", "OTHER"] = "OTHER"
    reason: Optional[str] = None

# V22 RAG and clinical runtimes are lazy-loaded so the V22 model server
# can start even before the OpenAI/BGE components are first used.
_rag_module = None
_clinical_module = None
_rag_lock = threading.Lock()
_clinical_lock = threading.Lock()


def load_rag_runtime():
    global _rag_module
    if _rag_module is not None:
        return _rag_module
    with _rag_lock:
        if _rag_module is not None:
            return _rag_module
        candidates = [
            APP_ROOT / "klinik_chong_rag_runtime_ver22.py",
            APP_ROOT / "klinik_chong_rag_runtime.py",
            Path("/content/drive/MyDrive/FYP2/chatbot_interface/klinik_chong_rag_runtime_ver22.py"),
            Path("/content/drive/MyDrive/FYP2/klinik_chong_database/klinik_chong_rag_runtime.py"),
        ]
        runtime_path = next((x for x in candidates if x.is_file()), None)
        if runtime_path is None:
            raise RuntimeError("A Klinik Chong RAG runtime was not found.")
        spec = importlib.util.spec_from_file_location("klinik_chong_rag_runtime_v22", runtime_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Unable to load RAG runtime.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _rag_module = module
        return module


def load_clinical_runtime():
    global _clinical_module
    if _clinical_module is not None:
        return _clinical_module
    with _clinical_lock:
        if _clinical_module is not None:
            return _clinical_module
        candidates = [
            APP_ROOT / "clinical_information_runtime.py",
            Path("/content/drive/MyDrive/FYP2/clinical_information_runtime.py"),
        ]
        runtime_path = next((x for x in candidates if x.is_file()), None)
        if runtime_path is None:
            raise RuntimeError("clinical_information_runtime.py was not found.")
        spec = importlib.util.spec_from_file_location("clinical_information_runtime_v17", runtime_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Unable to load clinical information runtime.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _clinical_module = module
        return module


MOCK_MODE = os.getenv("KLINIK_V22_MOCK_MODE", os.getenv("KLINIK_V20_MOCK_MODE", "0")) == "1"
runtime = ModelRuntime(mock_mode=MOCK_MODE)

app = FastAPI(
    title="Klinik Chong Chatbot V22 Model API",
    version="19.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health(load: bool = Query(default=False)) -> Dict[str, Any]:
    if load:
        errors: List[str] = []
        try:
            runtime.load_mbert()
        except Exception as exc:
            errors.append(f"mBERT: {exc}")
        try:
            runtime.load_hmm()
        except Exception as exc:
            errors.append(f"Pinyin2Hanzi HMM: {exc}")
        try:
            runtime.load_intention_model()
        except Exception as exc:
            errors.append(f"XLM-R intention: {exc}")
        if errors:
            return {
                "status": "degraded",
                "runtime_mode": "mock" if runtime.mock_mode else "real",
                "mock_mode": runtime.mock_mode,
                "models": runtime.model_info(),
                "errors": errors,
            }

    info = runtime.model_info()
    models_exist = (
        info["mbert"]["exists"]
        and info["pinyin2hanzi_hmm"]["exists"]
        and info["xlmr_intention"]["exists"]
    )
    return {
        "status": "mock" if runtime.mock_mode else ("ok" if models_exist else "degraded"),
        "runtime_mode": "mock" if runtime.mock_mode else "real",
        "mock_mode": runtime.mock_mode,
        "warning": (
            "UI TEST ONLY: real mBERT and Pinyin2Hanzi HMM inference is bypassed."
            if runtime.mock_mode else None
        ),
        "models": info,
    }


@app.post("/api/emergency/detect")
def emergency_detect(request: EmergencyDetectRequest) -> Dict[str, Any]:
    return detect_emergency_call_request(request.user_input)


@app.post("/api/emergency/calls")
def create_emergency_call(request: EmergencyCallRequest) -> Dict[str, Any]:
    """Place a one-way Twilio notification call to Klinik Chong's contact."""
    client = get_twilio_client()
    with _emergency_call_lock:
        for call_sid in list(_emergency_call_sids):
            try:
                existing = client.calls(call_sid).fetch()
                if str(existing.status) not in TERMINAL_CALL_STATUSES:
                    raise HTTPException(
                        status_code=409,
                        detail="An emergency notification call is already active.",
                    )
            except HTTPException:
                raise
            except Exception:
                _emergency_call_sids.discard(call_sid)

        try:
            call = client.calls.create(
                to=EMERGENCY_CONTACT_NUMBER,
                from_=TWILIO_FROM_NUMBER,
                url=TWILIO_TRIAL_VOICE_TEMPLATE_URL,
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Twilio call failed: {exc}") from exc
        _emergency_call_sids.add(str(call.sid))

    return {
        "call_sid": str(call.sid),
        "status": str(call.status),
        "contact_number": EMERGENCY_CONTACT_NUMBER,
        "severity": request.severity,
        "trigger": request.trigger,
    }


@app.get("/api/emergency/calls/{call_sid}")
def emergency_call_status(call_sid: str) -> Dict[str, Any]:
    if not re.fullmatch(r"CA[0-9a-fA-F]{32}", call_sid) or call_sid not in _emergency_call_sids:
        raise HTTPException(status_code=404, detail="Emergency call was not found in this session.")
    try:
        call = get_twilio_client().calls(call_sid).fetch()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Unable to read Twilio call status: {exc}") from exc
    return {"call_sid": call_sid, "status": str(call.status)}


@app.post("/api/emergency/calls/{call_sid}/cancel")
def cancel_emergency_call(call_sid: str) -> Dict[str, Any]:
    if not re.fullmatch(r"CA[0-9a-fA-F]{32}", call_sid) or call_sid not in _emergency_call_sids:
        raise HTTPException(status_code=404, detail="Emergency call was not found in this session.")
    try:
        call_resource = get_twilio_client().calls(call_sid)
        current = call_resource.fetch()
        current_status = str(current.status)
        if current_status in TERMINAL_CALL_STATUSES:
            return {"call_sid": call_sid, "status": current_status}
        target_status = "canceled" if current_status in {"queued", "ringing"} else "completed"
        updated = call_resource.update(status=target_status)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Unable to cancel Twilio call: {exc}") from exc
    return {"call_sid": call_sid, "status": str(updated.status), "requested_status": target_status}


@app.post("/api/language-detect")
def language_detect(request: LanguageDetectRequest) -> Dict[str, Any]:
    try:
        predictions = runtime.predict_languages(request.tokens, request.sentence)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"mBERT inference failed: {exc}") from exc

    return {
        "runtime_mode": "mock" if runtime.mock_mode else "real",
        "model_name": (
            "mock_mBERT_for_UI_test_only"
            if runtime.mock_mode
            else "mBert_language_detect_model/best_model"
        ),
        "model_path": str(runtime.mbert_path),
        "labels": DEFAULT_ID2LABEL,
        "predictions": predictions,
    }


@app.post("/api/intention-classify")
def intention_classify(request: IntentionClassifyRequest) -> Dict[str, Any]:
    deterministic = deterministic_intention_override(request.user_input)
    if deterministic is not None:
        return deterministic
    try:
        return runtime.predict_intention(request.user_input)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"XLM-R intention classifier unavailable: {exc}",
        ) from exc


@app.post("/api/pinyin-to-hanzi")
def pinyin_to_hanzi(request: PinyinToHanziRequest) -> Dict[str, Any]:
    try:
        conversions = runtime.convert_pinyin(request.inputs)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Pinyin2Hanzi HMM inference failed: {exc}") from exc

    return {
        "runtime_mode": "mock" if runtime.mock_mode else "real",
        "model_name": (
            "mock_Pinyin2Hanzi_HMM_for_UI_test_only"
            if runtime.mock_mode
            else "pinyin2hanzi_hmm_model"
        ),
        "model_path": str(runtime.hmm_path),
        "conversions": conversions,
    }



@app.post("/api/rag-query")
def rag_query(request: RagQueryRequest) -> Dict[str, Any]:
    try:
        module = load_rag_runtime()
        return module.run_rag_pipeline(
            user_input=request.user_input,
            response_language=request.response_language,
            past_queries=request.past_queries,
            intent_hint=request.intent_hint,
            conversation_state=request.conversation_state,
            conversation_history=request.conversation_history,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"RAG pipeline failed: {exc}") from exc


@app.post("/api/conversation-state")
def conversation_state(request: ConversationStateRequest) -> Dict[str, Any]:
    try:
        module = load_rag_runtime()
        return module.extract_conversation_state(
            user_input=request.user_input,
            conversation_state=request.conversation_state,
            conversation_history=request.conversation_history,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Conversation state extraction failed: {exc}") from exc


@app.post("/api/emotion-adjust")
def emotion_adjust(request: EmotionAdjustRequest) -> Dict[str, Any]:
    try:
        module = load_rag_runtime()
        return module.generate_emotion_aware_reply(
            user_input=request.user_input,
            base_response=request.base_response,
            response_language=request.response_language,
            conversation_state=request.conversation_state,
            conversation_history=request.conversation_history,
            safety_critical=request.safety_critical,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Emotion-aware reply generation failed: {exc}") from exc


@app.post("/api/clinical/extract")
def clinical_extract(request: ClinicalExtractRequest) -> Dict[str, Any]:
    try:
        module = load_clinical_runtime()
        return module.extract_clinical_information(request.user_input)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Clinical extraction failed: {exc}") from exc


@app.post("/api/clinical/note")
def clinical_note(request: ClinicalNoteRequest) -> Dict[str, Any]:
    try:
        module = load_clinical_runtime()
        note = module.generate_clinical_note(
            symptoms=request.symptoms,
            duration=request.duration,
            severity=request.severity,
            descriptions=request.descriptions,
            rating=request.rating,
        )
        return {"clinical_note": note}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Clinical note generation failed: {exc}") from exc


@app.post("/api/clinical/rating")
def clinical_rating(request: ClinicalRatingRequest) -> Dict[str, Any]:
    database_path = resolve_database_path()
    if not database_path.is_file():
        raise HTTPException(status_code=503, detail="Klinik Chong database was not found.")
    connection = sqlite3.connect(str(database_path), timeout=5)
    try:
        row = connection.execute(
            "SELECT clinical_note FROM appointment WHERE appointment_id = ?",
            (request.appointment_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Appointment was not found.")
        note = str(row[0] or "").strip()
        note = re.sub(r"\n?rate to clinic:\s*[1-5]\s+stars\s*$", "", note, flags=re.I).strip()
        note = f"{note}\nrate to clinic: {request.rating} stars" if note else f"rate to clinic: {request.rating} stars"
        connection.execute(
            "UPDATE appointment SET clinical_note = ?, updated_at = CURRENT_TIMESTAMP WHERE appointment_id = ?",
            (note, request.appointment_id),
        )
        connection.commit()
        return {
            "status": "ok",
            "appointment_id": request.appointment_id,
            "rating": request.rating,
            "clinical_note": note,
        }
    finally:
        connection.close()


@app.get("/api/booking-proxy/health")
def booking_proxy_health() -> Dict[str, Any]:
    return proxy_booking_api_json("/health")


@app.get("/api/booking-proxy/doctors")
def booking_proxy_doctors(
    expertise: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    remote = "/doctors"
    if expertise is not None and expertise.strip():
        remote += f"?expertise={url_quote(expertise.strip(), safe='')}"
    return proxy_booking_api_json(remote)


@app.get("/api/booking-proxy/available-slots")
def booking_proxy_available_slots(
    d_id: str,
    appointment_date: str,
) -> Dict[str, Any]:
    remote = (
        "/available-slots"
        f"?d_id={url_quote(d_id, safe='')}"
        f"&appointment_date={url_quote(appointment_date, safe='')}"
    )
    return proxy_booking_api_json(remote)


@app.get("/api/booking-proxy/appointments/latest")
def booking_proxy_latest_appointment(patient_ic: str) -> Dict[str, Any]:
    safe_ic = url_quote(patient_ic.strip(), safe="")
    return proxy_booking_api_json(
        f"/appointments/latest?patient_ic={safe_ic}"
    )


@app.post("/api/booking-proxy/patients")
def booking_proxy_patients(payload: Dict[str, Any]) -> Dict[str, Any]:
    return proxy_booking_api_json(
        "/patients",
        method="POST",
        payload=payload,
    )


@app.post("/api/booking-proxy/appointments")
def booking_proxy_appointments(payload: Dict[str, Any]) -> Dict[str, Any]:
    return proxy_booking_api_json(
        "/appointments",
        method="POST",
        payload=payload,
    )


@app.patch("/api/booking-proxy/appointments/{appointment_id}/cancel")
def booking_proxy_cancel_appointment(
    appointment_id: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    safe_id = url_quote(appointment_id, safe="")
    return proxy_booking_api_json(
        f"/appointments/{safe_id}/cancel",
        method="PATCH",
        payload=payload,
    )


@app.patch("/api/booking-proxy/appointments/{appointment_id}/reschedule")
def booking_proxy_reschedule_appointment(
    appointment_id: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    safe_id = url_quote(appointment_id, safe="")
    return proxy_booking_api_json(
        f"/appointments/{safe_id}/reschedule",
        method="PATCH",
        payload=payload,
    )


@app.post("/api/booking-proxy/appointments/complete-rebooking")
def booking_proxy_complete_rebooking(payload: Dict[str, Any]) -> Dict[str, Any]:
    return proxy_booking_api_json(
        "/appointments/complete-rebooking",
        method="POST",
        payload=payload,
    )


@app.get("/api/booking-proxy/developer/database/schema")
def booking_proxy_developer_schema() -> Dict[str, Any]:
    return proxy_booking_api_json(
        "/developer/database/schema",
        developer=True,
    )


@app.get("/api/booking-proxy/developer/database/tables/{table_name}/records")
def booking_proxy_developer_records(
    table_name: str,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> Dict[str, Any]:
    safe_table = url_quote(table_name, safe="")
    # Table names come from the already-validated schema UI; keep query values numeric.
    return proxy_booking_api_json(
        f"/developer/database/tables/{safe_table}/records?limit={limit}&offset={offset}",
        developer=True,
    )


@app.post("/api/booking-proxy/developer/database/doctor-leave")
def booking_proxy_doctor_leave(request: DoctorLeaveProxyRequest) -> Dict[str, Any]:
    return proxy_booking_api_json(
        "/developer/database/doctor-leave",
        method="POST",
        payload=request.model_dump(),
        developer=True,
    )


@app.get("/api/booking-proxy/developer/appointments")
def booking_proxy_developer_appointments() -> Dict[str, Any]:
    return proxy_booking_api_json(
        "/developer/appointments",
        developer=True,
    )


@app.post("/api/booking-proxy/developer/appointments/{appointment_id}/complete")
def booking_proxy_complete_developer_appointment(appointment_id: str) -> Dict[str, Any]:
    safe_id = url_quote(appointment_id.strip().upper(), safe="")
    return proxy_booking_api_json(
        f"/developer/appointments/{safe_id}/complete",
        method="POST",
        payload={},
        developer=True,
    )


@app.get("/api/developer/database/schema")
@app.get("/developer/database/schema", include_in_schema=False)
def developer_database_schema() -> Dict[str, Any]:
    """Return live SQLite schema metadata for the read-only Developer View."""
    database_path = resolve_database_path()
    with closing(open_database_read_only()) as connection:
        table_names = database_table_names(connection)
        tables: List[Dict[str, Any]] = []
        relationships: List[Dict[str, Any]] = []

        for table_name in table_names:
            quoted_table = quote_sql_identifier(table_name)
            column_rows = connection.execute(
                f"PRAGMA table_info({quoted_table});"
            ).fetchall()
            foreign_key_rows = connection.execute(
                f"PRAGMA foreign_key_list({quoted_table});"
            ).fetchall()
            row_count = connection.execute(
                f"SELECT COUNT(*) AS total FROM {quoted_table};"
            ).fetchone()["total"]

            columns = [
                {
                    "position": row["cid"],
                    "name": row["name"],
                    "type": row["type"] or "ANY",
                    "not_null": bool(row["notnull"]),
                    "default_value": row["dflt_value"],
                    "primary_key_position": row["pk"],
                }
                for row in column_rows
            ]
            foreign_keys = [
                {
                    "id": row["id"],
                    "sequence": row["seq"],
                    "from_column": row["from"],
                    "to_table": row["table"],
                    "to_column": row["to"],
                    "on_update": row["on_update"],
                    "on_delete": row["on_delete"],
                }
                for row in foreign_key_rows
            ]
            for foreign_key in foreign_keys:
                relationships.append(
                    {
                        "from_table": table_name,
                        "from_column": foreign_key["from_column"],
                        "to_table": foreign_key["to_table"],
                        "to_column": foreign_key["to_column"],
                        "on_update": foreign_key["on_update"],
                        "on_delete": foreign_key["on_delete"],
                    }
                )

            tables.append(
                {
                    "name": table_name,
                    "row_count": row_count,
                    "columns": columns,
                    "foreign_keys": foreign_keys,
                }
            )

    return {
        "status": "connected",
        "read_only": True,
        "database_name": database_path.name,
        "database_path": str(database_path),
        "table_count": len(tables),
        "tables": tables,
        "relationships": relationships,
    }


@app.get("/api/developer/database/tables/{table_name}/records")
@app.get(
    "/developer/database/tables/{table_name}/records",
    include_in_schema=False,
)
def developer_database_records(
    table_name: str,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> Dict[str, Any]:
    """Read records from one verified table; writes are never accepted here."""
    with closing(open_database_read_only()) as connection:
        table_names = database_table_names(connection)
        if table_name not in table_names:
            raise HTTPException(status_code=404, detail="Database table was not found.")

        quoted_table = quote_sql_identifier(table_name)
        total_records = connection.execute(
            f"SELECT COUNT(*) AS total FROM {quoted_table};"
        ).fetchone()["total"]
        column_rows = connection.execute(
            f"PRAGMA table_info({quoted_table});"
        ).fetchall()
        rows = connection.execute(
            f"SELECT * FROM {quoted_table} LIMIT ? OFFSET ?;",
            (limit, offset),
        ).fetchall()

        columns = [str(row["name"]) for row in column_rows]
        records = [
            {
                column: json_safe_database_value(row[column])
                for column in columns
            }
            for row in rows
        ]

    return {
        "status": "ok",
        "read_only": True,
        "table": table_name,
        "columns": columns,
        "total_records": total_records,
        "limit": limit,
        "offset": offset,
        "records": records,
    }


@app.get("/")
def frontend() -> FileResponse:
    return FileResponse(APP_ROOT / "index_ver22.html")


# Keep this mount after all API routes.
app.mount("/", StaticFiles(directory=APP_ROOT, html=True), name="frontend-static")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Klinik Chong Chatbot V22.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument(
        "--booking-api-base",
        default="",
        help=(
            "Override bookingApiBase for this run, useful after Colab creates "
            "a new trycloudflare URL."
        ),
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use explicit mock model outputs for UI testing only.",
    )
    args = parser.parse_args()

    if args.booking_api_base.strip():
        os.environ["KLINIK_CHONG_BOOKING_API_BASE"] = args.booking_api_base.strip()

    print(f"Booking API proxy target: {booking_api_base() or '[not configured]'}")

    if args.mock:
        runtime.mock_mode = True
        print("\n" + "!" * 72)
        print("V22 MOCK MODE — UI TEST ONLY")
        print("Real mBERT and Pinyin2Hanzi HMM inference is BYPASSED.")
        print("Unmapped Pinyin may be returned unchanged by design.")
        print("For normal FYP testing, restart WITHOUT --mock.")
        print("!" * 72 + "\n")
    else:
        runtime.mock_mode = False
        print("\nV22 REAL MODEL MODE — mBERT + Pinyin2Hanzi HMM enabled.\n")

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
