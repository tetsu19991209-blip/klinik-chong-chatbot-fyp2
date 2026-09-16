"""Klinik Chong ask_info RAG runtime.

Loads the existing ChromaDB and SQLite database without re-running ingestion.
Designed for development/evaluation notebooks in Google Colab.
"""

import json
import os
import re
import sqlite3
from difflib import SequenceMatcher
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import chromadb
import torch
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

try:
    from dotenv import load_dotenv
except ImportError:  # Colab Secrets or host environment variables still work.
    load_dotenv = None

try:
    from google.colab import userdata
except Exception:
    userdata = None


# -----------------------------------------------------------------------------
# Paths / constants
# -----------------------------------------------------------------------------
if load_dotenv is not None:
    load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
FYP2_ROOT = Path(
    os.getenv("KLINIK_CHONG_FYP2_ROOT", "/content/drive/MyDrive/FYP2")
).expanduser()
DATABASE_DIR = Path(
    os.getenv(
        "KLINIK_CHONG_DATABASE_DIR",
        str(PROJECT_ROOT / "data"),
    )
).expanduser()
CHROMA_DB_PATH = Path(
    os.getenv("KLINIK_CHONG_CHROMA_PATH", str(DATABASE_DIR / "chroma_db"))
).expanduser()
DB_PATH = Path(
    os.getenv("KLINIK_CHONG_DB_PATH", str(DATABASE_DIR / "klinik_chong.db"))
).expanduser()
COLLECTION_NAME = "klinik_chong_handbook"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
MALAYSIA_TIMEZONE = ZoneInfo("Asia/Kuala_Lumpur")
MIN_RAG_SIMILARITY = 0.50

MAIN_SECTION_DESCRIPTIONS = {
    "1": "Selamat Datang ke Klinik Chong — introduction, supported languages, general disclaimer",
    "2": "Mengenai klinik kami — services, location, contact information, operating hours",
    "3": "Hak dan tanggungjawab pesakit — privacy, accurate information, punctuality, cancellation responsibility",
    "4": "Lawatan Anda ke Klinik — new/follow-up visit process, documents, before/during/after consultation",
    "5": "Panduan temu janji — booking, required information, slots, doctor availability, cancellation/rescheduling, walk-in",
    "6": "Chatbot Klinik Chong — chatbot functions, languages, collected data, limitations, clinical summary",
    "7": "Maklumat kesihatan umum — fever, cough, flu, vomiting, respiratory symptoms, when to seek care",
}


# -----------------------------------------------------------------------------
# Runtime initialization
# -----------------------------------------------------------------------------
def _get_openai_key():
    import os
    key = str(os.environ.get("OPENAI_API_KEY") or "").strip()
    if key:
        return key

    if userdata is not None:
        try:
            key = str(userdata.get("OPENAI_API_KEY") or "").strip()
        except Exception:
            key = ""
        if key:
            return key

    raise RuntimeError(
        "OPENAI_API_KEY not found. Add it to Colab Secrets and expose it to the server environment."
    )


def _validate_paths():
    if not CHROMA_DB_PATH.exists():
        raise FileNotFoundError(f"ChromaDB folder not found: {CHROMA_DB_PATH}")
    if not DB_PATH.exists():
        raise FileNotFoundError(f"SQLite database not found: {DB_PATH}")


_validate_paths()

device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
collection = chroma_client.get_collection(
    name=COLLECTION_NAME,
    embedding_function=None,
)

model = ChatOpenAI(
    api_key=_get_openai_key(),
    model="gpt-4.1-mini",
    temperature=0,
)


# -----------------------------------------------------------------------------
# Handbook structure helpers
# -----------------------------------------------------------------------------
def _load_handbook_catalog():
    data = collection.get(include=["metadatas"])
    rows = []
    for metadata in data.get("metadatas", []):
        if not metadata:
            continue
        rows.append({
            "chapter_number": str(metadata.get("chapter_number", "")),
            "section_number": str(metadata.get("section_number", "")),
            "section_title": str(metadata.get("section_title", "")),
        })
    return rows


HANDBOOK_CATALOG = _load_handbook_catalog()

main_sections_text = "\n".join(
    f"{number}. {description}"
    for number, description in MAIN_SECTION_DESCRIPTIONS.items()
)


def build_subsections_text(selected_main_sections):
    selected = {str(x).strip() for x in selected_main_sections}
    seen = set()
    lines = []

    def section_sort_key(section_number):
        try:
            return tuple(int(p) for p in str(section_number).split("."))
        except Exception:
            return (999, 999)

    for row in sorted(HANDBOOK_CATALOG, key=lambda r: section_sort_key(r["section_number"])):
        if row["chapter_number"] not in selected:
            continue
        section_number = row["section_number"]
        if section_number in seen:
            continue
        seen.add(section_number)
        lines.append(f'{section_number}. {row["section_title"]}')

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# ChromaDB retrieval
# -----------------------------------------------------------------------------
def create_query_embedding(query):
    return embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def retrieve_chunks(query, top_k=3, where=None):
    query_embedding = create_query_embedding(query)

    query_arguments = {
        "query_embeddings": query_embedding.tolist(),
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where is not None:
        query_arguments["where"] = where

    results = collection.query(**query_arguments)
    retrieved_chunks = []

    for rank in range(len(results["ids"][0])):
        distance = float(results["distances"][0][rank])
        metadata = results["metadatas"][0][rank]

        retrieved_chunks.append({
            "rank": rank + 1,
            "chunk_id": results["ids"][0][rank],
            "section_number": metadata["section_number"],
            "section_title": metadata["section_title"],
            "knowledge_domain": metadata.get("knowledge_domain"),
            "distance": distance,
            "similarity": 1.0 - distance,
            "document": results["documents"][0][rank],
            "metadata": metadata,
        })

    return retrieved_chunks


def parse_section_numbers(section_output):
    return [x.strip() for x in section_output.split(",") if x.strip()]


def retrieve_selected_subsections(compiled_query, selected_subsections):
    subsection_numbers = parse_section_numbers(selected_subsections)
    retrieved_results = []

    for section_number in subsection_numbers:
        results = retrieve_chunks(
            compiled_query,
            top_k=1,
            where={"section_number": section_number},
        )
        retrieved_results.extend(results)

    return retrieved_results


# -----------------------------------------------------------------------------
# SQLite helpers
# -----------------------------------------------------------------------------
def get_db_connection():
    return sqlite3.connect(str(DB_PATH))


def build_doctor_lookup():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT d_id, d_name
        FROM doctor
        WHERE is_active = 1
    """)
    rows = cursor.fetchall()
    conn.close()
    return {name: d_id for d_id, name in rows}


doctor_lookup = build_doctor_lookup()


DOCTOR_MATCH_THRESHOLD = 0.80


def normalize_doctor_name(value):
    text = re.sub(r"[^a-z0-9 ]+", " ", str(value or "").lower())
    text = re.sub(r"\b(?:dr|doctor|doktor)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def doctor_similarity(mention, full_name):
    mention_normalized = normalize_doctor_name(mention)
    full_normalized = normalize_doctor_name(full_name)
    if not mention_normalized:
        return 0.0
    aliases = {full_normalized, full_normalized.replace(" ", "")}
    tokens = full_normalized.split()
    aliases.update(tokens)
    if len(tokens) > 1:
        aliases.add(" ".join(tokens[-2:]))
    mention_compact = mention_normalized.replace(" ", "")
    if mention_normalized in aliases or mention_compact in aliases:
        return 1.0
    return max(
        SequenceMatcher(None, mention_normalized, alias).ratio()
        for alias in aliases if alias
    )


def match_doctor_mentions(doctor_mentions, lookup=None, threshold=DOCTOR_MATCH_THRESHOLD):
    if lookup is None:
        lookup = doctor_lookup
    matches_by_id = {}
    for mention in doctor_mentions or []:
        candidates = []
        for full_name, doctor_id in lookup.items():
            score = doctor_similarity(mention, full_name)
            if score >= threshold:
                candidates.append({
                    "doctor_id": str(doctor_id),
                    "doctor_name": str(full_name),
                    "matched_from": str(mention),
                    "similarity": round(float(score), 4),
                })
        exact = [item for item in candidates if item["similarity"] == 1.0]
        for item in exact or candidates:
            previous = matches_by_id.get(item["doctor_id"])
            if previous is None or item["similarity"] > previous["similarity"]:
                matches_by_id[item["doctor_id"]] = item
    return sorted(matches_by_id.values(), key=lambda item: (-item["similarity"], item["doctor_name"]))


def match_doctor_name(extracted_name, lookup=None):
    """Backward-compatible single-doctor helper."""
    matches = match_doctor_mentions([extracted_name], lookup)
    return matches[0]["doctor_id"] if len(matches) == 1 else None


def malaysia_now():
    """Return the current timezone-aware Malaysia date and time."""
    return datetime.now(MALAYSIA_TIMEZONE)


def doctor_name_from_id(doctor_id, lookup=None):
    if lookup is None:
        lookup = doctor_lookup
    for doctor_name, candidate_id in lookup.items():
        if candidate_id == doctor_id:
            return doctor_name
    return doctor_id or "Unknown doctor"


def resolve_relative_or_explicit_date(text, today=None):
    """Best-effort fallback when LLM parameter extraction is unavailable."""
    if today is None:
        today = malaysia_now().date()
    value = str(text or "").lower()

    iso_match = re.search(r"(?<!\d)(20\d{2})[-/](0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])(?!\d)", value)
    if iso_match:
        try:
            return date(*map(int, iso_match.groups())).isoformat()
        except ValueError:
            return None

    day_first = re.search(r"(?<!\d)(0?[1-9]|[12]\d|3[01])[-/](0?[1-9]|1[0-2])[-/](20\d{2})(?!\d)", value)
    if day_first:
        try:
            day_value, month_value, year_value = map(int, day_first.groups())
            return date(year_value, month_value, day_value).isoformat()
        except ValueError:
            return None

    relative_rules = [
        (7, r"(?:一个星期后|一個星期後|\b(?:seminggu|satu\s+minggu|1\s+minggu)\s+(?:lagi|depan)\b)"),
        (5, r"(?:五天后|五天後|\b(?:lima|5)\s+hari\s+lagi\b)"),
        (4, r"(?:四天后|四天後|\b(?:empat|4)\s+hari\s+lagi\b)"),
        (3, r"(?:三天后|三天後|\b(?:tiga|3)\s+hari(?:\s+lagi)?\b)"),
        (2, r"(?:后天|後天|两天后|兩天後|兩日後|两日后|\blusa\b|\b(?:dua|2)\s+hari\s+lagi\b|\bday\s+after\s+tomorrow\b|\bin\s+(?:two|2)\s+days?\b)"),
        (1, r"(?:明天|\besok\b|\btomorrow\b)"),
        (0, r"(?:今天|今日|\bhari\s+ini\b|\btoday\b)"),
    ]
    for offset, pattern in relative_rules:
        if re.search(pattern, value, flags=re.I):
            return (today + timedelta(days=offset)).isoformat()

    weekday_aliases = {
        0: r"(?:星期一|周一|週一|\bmonday\b|\bisnin\b)",
        1: r"(?:星期二|周二|週二|\btuesday\b|\bselasa\b)",
        2: r"(?:星期三|周三|週三|\bwednesday\b|\brabu\b)",
        3: r"(?:星期四|周四|週四|\bthursday\b|\bkhamis\b)",
        4: r"(?:星期五|周五|週五|\bfriday\b|\bjumaat\b)",
        5: r"(?:星期六|周六|週六|\bsaturday\b|\bsabtu\b)",
        6: r"(?:星期日|星期天|周日|週日|\bsunday\b|\bahad\b)",
    }
    for weekday, pattern in weekday_aliases.items():
        if re.search(pattern, value, flags=re.I):
            days_ahead = (weekday - today.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7
            return (today + timedelta(days=days_ahead)).isoformat()
    return None


SLOT_INTENT_LABELS = {
    "ask_slot_availability", "ask_available_slot", "ask_availability_slot",
    "check_availability_query", "check_slot_availability", "slot_availability",
}
SLOT_TERMS = re.compile(
    r"(?:slot(?:s)?|availability|available\s+(?:time|appointment)|"
    r"appointment\s+(?:time|slot)|masa\s+(?:kosong|lapang)|waktu\s+kosong|"
    r"slot\s+kosong|janji\s+temu.*kosong|boleh\s+book|"
    r"可预约|可預約|可以预约|可以預約|预约位|預約位|有位|"
    r"空位|有空|空档|空檔|号源|號源|休息吗|休息嗎)", re.I,
)
SLOT_POLICY_TERMS = re.compile(
    r"(?:berapa\s+lama.*slot|tempoh.*slot|slot.*duration|"
    r"how.*slot.*determined|如何.*决定.*时段|哪些因素.*时段|"
    r"一个slot.*多久|一個slot.*多久)", re.I,
)
STATIC_POLICY_TERMS = re.compile(
    r"(?:迟到|遲到|预约.*规则|預約.*規則|late|lateness|arrive\s+late|"
    r"lewat|kelewatan|polisi|policy|流程|procedure|process|规定|規定)", re.I,
)


def is_slot_availability_query(text, intention=None):
    """Use the classifier as supporting evidence, never as the only evidence."""
    value = str(text or "")
    content_match = bool(SLOT_TERMS.search(value) and not SLOT_POLICY_TERMS.search(value))
    normalized_intention = str(intention or "").strip().lower()
    if content_match:
        return True
    if normalized_intention not in SLOT_INTENT_LABELS:
        return False
    return bool(
        re.search(r"(?:\bdr\.?\b|\bdoctor\b|\bdoktor\b|医生|醫生)", value, re.I)
        and (resolve_relative_or_explicit_date(value) or re.search(r"\b\d{1,2}(?::|\.)\d{2}\b|\b\d{1,2}\s*(?:am|pm)\b|\d{1,2}\s*点", value, re.I))
    )


def get_available_doctors(target_date):
    conn = get_db_connection()
    cursor = conn.cursor()

    date_obj = datetime.strptime(target_date, "%Y-%m-%d")
    day_of_week = date_obj.isoweekday()

    cursor.execute("""
        SELECT holiday_name
        FROM clinic_holiday
        WHERE holiday_date = ?
        AND is_closed = 1
    """, (target_date,))
    holiday = cursor.fetchone()

    if holiday:
        conn.close()
        return {
            "date": target_date,
            "clinic_closed": True,
            "reason": holiday[0],
            "doctors": [],
        }

    cursor.execute("""
        SELECT
            d.d_id,
            d.d_name,
            d.d_expertise,
            s.start_time,
            s.end_time
        FROM doctor d
        JOIN doctor_weekly_schedule s
            ON d.d_id = s.d_id
        WHERE
            d.is_active = 1
            AND s.day_of_week = ?
            AND s.is_working = 1
    """, (day_of_week,))

    doctors = cursor.fetchall()
    available_doctors = []

    for doctor in doctors:
        d_id = doctor[0]
        cursor.execute("""
            SELECT 1
            FROM doctor_leave
            WHERE d_id = ?
            AND leave_date = ?
            AND status = 'APPROVED'
        """, (d_id, target_date))

        if cursor.fetchone():
            continue

        available_doctors.append({
            "d_id": doctor[0],
            "d_name": doctor[1],
            "expertise": doctor[2],
            "start_time": doctor[3],
            "end_time": doctor[4],
        })

    conn.close()
    return {
        "date": target_date,
        "clinic_closed": False,
        "doctors": available_doctors,
    }


def get_active_doctors():
    """Return the complete active doctor directory, independent of today's roster."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT d_id, d_name, d_expertise
        FROM doctor
        WHERE is_active = 1
        ORDER BY d_id
    """)
    doctors = [
        {"d_id": row[0], "d_name": row[1], "expertise": row[2]}
        for row in cursor.fetchall()
    ]
    conn.close()
    return doctors


def doctor_directory_query_mode(user_input, compiled_query="", conversation_history=""):
    """Identify total-doctor and doctor-name questions, including short follow-ups."""
    current = str(user_input or "").strip()
    compiled = str(compiled_query or "").strip()
    history = str(conversation_history or "")
    combined = f"{current}\n{compiled}"

    count_pattern = re.compile(
        r"(?:\bhow\s+many\s+doctors?\b|\bberapa\s+(?:orang\s+)?doktor\b|"
        r"\bdoktor\s+ada\s+berapa\b|诊所(?:有)?多少(?:个|位)?医生|"
        r"診所(?:有)?多少(?:個|位)?醫生|多少(?:个|位)?(?:医生|醫生))",
        re.I,
    )
    list_pattern = re.compile(
        r"(?:\bwhich\s+doctors?\b|\bwho\s+are\s+the\s+doctors?\b|"
        r"\bdoctor\s+list\b|\bsenarai\s+doktor\b|\bsiapa\s+(?:sahaja\s+)?doktor\b|"
        r"(?:有哪些|都有谁|都有誰|有哪几位|有哪幾位)(?:医生|醫生)|"
        r"(?:医生|醫生)(?:有谁|有誰|名单|名單))",
        re.I,
    )
    if count_pattern.search(combined):
        return "count"
    if list_pattern.search(combined):
        return "list"

    follow_up = bool(re.fullmatch(
        r"(?:有谁(?:呢)?|有誰(?:呢)?|都有谁(?:呢)?|都有誰(?:呢)?|"
        r"谁(?:呢)?|誰(?:呢)?|which\s+ones?|who\s+are\s+they|siapa(?:\s+sahaja)?)[?？.!。 ]*",
        current,
        re.I,
    ))
    history_mentions_doctors = bool(re.search(
        r"(?:医生|醫生|\bdoctors?\b|\bdoktor\b)", history, re.I
    ))
    return "list" if follow_up and history_mentions_doctors else None


def format_active_doctor_response(doctors, response_language, mode):
    chinese = str(response_language or "").lower() in {"zh", "chinese", "中文", "simplified chinese"}
    count = len(doctors)
    if mode == "count":
        return (
            f"Klinik Chong 共有 {count} 位在职医生。" if chinese
            else f"Klinik Chong mempunyai {count} orang doktor aktif."
        )

    lines = []
    for index, doctor in enumerate(doctors, start=1):
        name = _doctor_display_name(doctor.get("d_name") or doctor.get("d_id"))
        expertise = str(doctor.get("expertise") or "").strip()
        lines.append(f"{index}. {name}" + (f" — {expertise}" if expertise else ""))
    heading = (
        f"Klinik Chong 共有 {count} 位在职医生：" if chinese
        else f"Klinik Chong mempunyai {count} orang doktor aktif:"
    )
    return heading + "\n" + "\n".join(lines)


def get_available_slots(target_date, doctor_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    date_obj = datetime.strptime(target_date, "%Y-%m-%d")
    day_of_week = date_obj.isoweekday()
    malaysia_current = malaysia_now()
    if date_obj.date() < malaysia_current.date():
        conn.close()
        return {
            "date": target_date,
            "doctor_id": doctor_id,
            "available_slots": [],
            "reason": "The selected date is in the past.",
            "reason_code": "PAST_DATE",
        }

    cursor.execute("""
        SELECT
            appointment_start,
            last_slot_start,
            break_start,
            break_end,
            slot_duration,
            is_open
        FROM clinic_schedule
        WHERE day_of_week = ?
    """, (day_of_week,))

    clinic_schedule = cursor.fetchone()
    if not clinic_schedule:
        conn.close()
        return {
            "date": target_date,
            "doctor_id": doctor_id,
            "available_slots": [],
            "reason": "No clinic schedule found.",
            "reason_code": "CLINIC_SCHEDULE_NOT_FOUND",
        }

    appointment_start, last_slot_start, break_start, break_end, slot_duration, is_open = clinic_schedule

    if not is_open:
        conn.close()
        return {
            "date": target_date,
            "doctor_id": doctor_id,
            "available_slots": [],
            "reason": "Clinic is closed.",
            "reason_code": "CLINIC_CLOSED",
        }

    cursor.execute("""
        SELECT holiday_name
        FROM clinic_holiday
        WHERE holiday_date = ?
        AND is_closed = 1
    """, (target_date,))
    holiday = cursor.fetchone()

    if holiday:
        conn.close()
        return {
            "date": target_date,
            "doctor_id": doctor_id,
            "available_slots": [],
            "reason": holiday[0],
            "reason_code": "CLINIC_HOLIDAY",
        }

    cursor.execute("""
        SELECT start_time, end_time
        FROM doctor_weekly_schedule
        WHERE d_id = ?
        AND day_of_week = ?
        AND is_working = 1
    """, (doctor_id, day_of_week))
    doctor_schedule = cursor.fetchone()

    if not doctor_schedule:
        conn.close()
        return {
            "date": target_date,
            "doctor_id": doctor_id,
            "available_slots": [],
            "reason": "Doctor is not working on this day.",
            "reason_code": "DOCTOR_REST_DAY",
        }

    doctor_start, doctor_end = doctor_schedule

    cursor.execute("""
        SELECT 1
        FROM doctor_leave
        WHERE d_id = ?
        AND leave_date = ?
        AND status = 'APPROVED'
    """, (doctor_id, target_date))

    if cursor.fetchone():
        conn.close()
        return {
            "date": target_date,
            "doctor_id": doctor_id,
            "available_slots": [],
            "reason": "Doctor is on approved leave.",
            "reason_code": "DOCTOR_ON_LEAVE",
        }

    cursor.execute("""
        SELECT start_time
        FROM appointment
        WHERE d_id = ?
        AND appointment_date = ?
        AND status NOT IN ('CANCELLED', 'CANCELED')
    """, (doctor_id, target_date))
    booked_slots = {row[0] for row in cursor.fetchall()}

    clinic_start_dt = datetime.strptime(appointment_start, "%H:%M")
    clinic_last_dt = datetime.strptime(last_slot_start, "%H:%M")
    doctor_start_dt = datetime.strptime(doctor_start, "%H:%M")
    doctor_end_dt = datetime.strptime(doctor_end, "%H:%M")

    start_dt = max(clinic_start_dt, doctor_start_dt)
    last_dt = min(clinic_last_dt, doctor_end_dt)

    if break_start and break_end:
        break_start_dt = datetime.strptime(break_start, "%H:%M")
        break_end_dt = datetime.strptime(break_end, "%H:%M")
    else:
        break_start_dt = None
        break_end_dt = None

    available_slots = []
    current = start_dt

    while current <= last_dt:
        slot_time = current.strftime("%H:%M")
        in_break = (
            break_start_dt
            and break_end_dt
            and break_start_dt <= current < break_end_dt
        )
        slot_is_future = True
        if date_obj.date() == malaysia_current.date():
            slot_datetime = datetime.combine(
                date_obj.date(), current.time(), tzinfo=MALAYSIA_TIMEZONE
            )
            slot_is_future = slot_datetime > malaysia_current
        if not in_break and slot_time not in booked_slots and slot_is_future:
            available_slots.append(slot_time)
        current += timedelta(minutes=slot_duration)

    conn.close()
    result = {
        "date": target_date,
        "doctor_id": doctor_id,
        "available_slots": available_slots,
    }
    if not available_slots:
        result["reason"] = "All appointment slots are fully booked or have already passed."
        result["reason_code"] = "FULLY_BOOKED_OR_PASSED"
    return result


# -----------------------------------------------------------------------------
# LangChain components
# -----------------------------------------------------------------------------
query_compiler_prompt = ChatPromptTemplate.from_template("""
Rewrite the current message into one clear standalone retrieval query.
Keep the query in the user's response language and do not answer it.

For appointment-slot questions:
- preserve the doctor name exactly when stated;
- preserve an explicit date or relative date expression;
- preserve a requested time when stated;
- make the slot-availability request explicit.

Use history only when the current message depends on earlier context.
Never invent a doctor, date, time, slot or database result.

Language: {response_language}
History: {past_queries}
Current message: {input_query}
""")
query_compiler_chain = query_compiler_prompt | model | StrOutputParser()

main_section_prompt = ChatPromptTemplate.from_template("""
Select up to 3 handbook main sections that directly answer the query.

Rules:
- Use only the listed Klinik Chong handbook sections.
- Do not force a section merely because a word is loosely related.
- If the query is unrelated to Klinik Chong, appointment services, or health
  information explicitly covered by the listed sections, return exactly NONE.
- Otherwise return section numbers only, separated by commas.

1. Selamat Datang ke Klinik Chong — introduction, supported languages, general disclaimer
2. Mengenai klinik kami — services, location, contact information, operating hours
3. Hak dan tanggungjawab pesakit — privacy, accurate information, punctuality, cancellation responsibility
4. Lawatan Anda ke Klinik — new/follow-up visit process, documents, before/during/after consultation
5. Panduan temu janji — booking, required information, slots, doctor availability, cancellation/rescheduling, walk-in
6. Chatbot Klinik Chong — chatbot functions, languages, collected data, limitations, clinical summary
7. Maklumat kesihatan umum — fever, cough, flu, vomiting, respiratory symptoms, when to seek care

Query:
{compiled_query}

Main Sections:
{main_sections}
""")
main_section_analyzer_chain = main_section_prompt | model | StrOutputParser()

subsection_prompt = ChatPromptTemplate.from_template("""
Select up to 3 subsections that directly answer the query.

Rules:
- Use only the listed Klinik Chong handbook subsections.
- Do not select a subsection based on a weak or indirect match.
- If none of the listed subsections directly supports an answer, return exactly NONE.
- Otherwise return subsection numbers only, separated by commas.

Query:
{compiled_query}

Subsections:
{subsections}
""")
subsection_analyzer_chain = subsection_prompt | model | StrOutputParser()

retrieval_router_prompt = ChatPromptTemplate.from_template("""
Choose the only retrieval source that can directly answer this non-slot query.

Return only one number:
0 = ChromaDB only
1 = SQLite only
2 = Both ChromaDB and SQLite

ChromaDB contains clinic policies, late-arrival rules, booking procedures,
cancellation/rescheduling policies, handbook information, services and health information.
SQLite contains current doctor schedules, rest days, approved leave and clinic holidays.
Slot-availability questions are handled before this router.

This router receives only supported ask_info requests after intent filtering.
Choose the source based on the information required by the query. Unsupported
requests are handled by the unrelated-intent branch before RAG retrieval.

Examples:
- "如果预约迟到了会怎样？" -> 0
- "Apakah polisi kelewatan Klinik Chong?" -> 0
- "Dr Lee今天有上班吗？" -> 1
- "今天是不是诊所假期？" -> 1

Selected handbook subsections: {selected_subsections}

Query:
{compiled_query}
""")
retrieval_router_chain = retrieval_router_prompt | model | StrOutputParser()

sql_parameter_prompt = ChatPromptTemplate.from_template("""
You extract SQLite parameters for Klinik Chong doctor-slot availability.

Malaysia reference date: {current_date}
Active doctors from SQLite:
{doctor_names}

User query: {compiled_query}

Return JSON only:
{{"doctor_mentions": [], "target_date": "YYYY-MM-DD", "requested_time": null}}

Rules:
1. Return every doctor mentioned. A surname, given name, joined spelling, typo or partial name is allowed. Use the active-doctor list to correct likely typos, but never add an unrelated doctor.
2. If a short name can refer to two active doctors, keep the ambiguous short name so both can be retrieved.
3. Convert explicit and casual dates into YYYY-MM-DD. A day/month without a year means the next occurrence on or after the reference date.
4. Resolve today/hari ini/今天, tomorrow/esok/明天, lusa/后天, N days later, this weekday, next weekday/week and one month later from the Malaysia reference date.
5. If no date is supplied, use the reference date. If no doctor is supplied, return an empty doctor_mentions list.
6. Convert casual times to HH:MM. In clinic context, bare 1-5 o'clock means afternoon. Respect am/pm and preserve a nonstandard time such as 4.25 as 16:25; do not round it.
7. Do not answer the user and do not wrap the JSON in Markdown.
""")
sql_parameter_chain = sql_parameter_prompt | model | StrOutputParser()

context_compiler_prompt = ChatPromptTemplate.from_template("""
Compile only the retrieved information that directly answers the query.

Rules:
- Use only the retrieved Klinik Chong handbook or SQLite information below.
- Do not use general knowledge or add medical, cooking, lifestyle or other advice.
- Remove duplicated and irrelevant details.
- If the retrieved information does not directly answer the query, return exactly
  NO_RELEVANT_CONTEXT.
- Do not answer the user directly.

Query:
{compiled_query}

Retrieved information:
{retrieved_context}
""")
context_compiler_chain = context_compiler_prompt | model | StrOutputParser()

emotion_response_prompt = ChatPromptTemplate.from_template("""
Detect the user's emotion as one of:
neutral, happy, sad, angry, afraid, confuse.

Emotion selection rules:
- Use neutral for ordinary questions, factual enquiries, greetings and messages
  that do not clearly express another emotion.
- Use happy only when the user clearly expresses happiness, gratitude,
  satisfaction or positive excitement.
- Use sad only when sadness, disappointment, grief or emotional pain is clear.
- Use angry only when anger, frustration, hostility or strong irritation is clear.
- Use afraid only when fear, worry, anxiety or panic is clear.
- Use confuse only when the user explicitly says that something is unclear,
  difficult to understand or asks for steps. Do not label a normal information
  question as confuse merely because it contains a question mark.

Use the detected emotion only to adjust the tone. Never mention the emotion
label in the response. Treat the supplied Context as the only authoritative
source. Never invent clinic information, medical advice, appointment data or
available actions. Do not provide consultation or advice unrelated to Klinik
Chong's medical and clinic services. Do not answer beyond the knowledge-based
RAG or SQLite context. If Context is empty, irrelevant, insufficient, or equals
NO_RELEVANT_CONTEXT, state only that no relevant information was found in the
Klinik Chong knowledge base. Do not add general knowledge, suggestions,
alternatives, clarification questions, or offers of further help.

Preserve all facts in valid Context and answer the user's actual question
directly. A question
about where Klinik Chong is located asks for the clinic address; it does not
ask for the user's live location. If the Context contains a clinic address,
state that address and never replace it with a claim that the user's location
cannot be accessed.

Tone rules:
- neutral: polite, clear and natural.
- happy: warm and positively engaged without exaggeration.
- sad: acknowledge the difficulty briefly, then answer with empathy.
- angry: begin with one brief acknowledgement of the user's frustration, remain
  calm and invite respectful, constructive communication without scolding.
- afraid: reassure gently without making false promises, then give a clear and
  practical answer.
- confuse: organise the answer into short numbered steps when this improves
  understanding.

For every non-neutral emotion, the response must visibly reflect the matching
tone instead of repeating a mechanical neutral answer unchanged.

Known user state may be used only when it is relevant. A known user name may
be used naturally, but not in every reply.
Respond in {response_language}. Keep the answer concise.

Return exactly in this format:
Emotion: <emotion>
Response: <final response>

User:
{user_input}

Known user state:
{conversation_state}

Recent conversation:
{conversation_history}

Context:
{compiled_context}
""")
emotion_response_chain = emotion_response_prompt | model | StrOutputParser()

conversation_state_prompt = ChatPromptTemplate.from_template("""
Extract only user information that is explicitly stated in the current
message. Use the existing state and recent conversation only to preserve known
values, not to guess new ones.

Existing conversation state:
{conversation_state}

Recent conversation:
{conversation_history}

Current message:
{user_input}

Rules:
- Extract the user's name only when the user clearly identifies themself.
- Questions asking whether the chatbot remembers the user's name, including
  "你知道我叫什么名字吗", "我的名字是？", "what is my name" and
  "nama saya siapa", must return user_name null.
- Statements about a condition or need are not names. For example, "I am very
  tired", "I am sick", "我是病人" and "saya penat" must return user_name null.
- Never treat a doctor's, family member's or another mentioned person's name as
  the user's name.
- Keep an existing value when the current message does not replace it.
- Never infer an appointment date, time, symptom or severity from an assistant
  message alone.
- Return valid JSON only, without Markdown.

Return exactly these keys:
{{
  "user_name": null,
  "preferred_language": null,
  "doctor_name": null,
  "appointment_date": null,
  "appointment_time": null,
  "symptoms": [],
  "duration": null,
  "severity": null
}}
""")
conversation_state_chain = conversation_state_prompt | model | StrOutputParser()

emotion_adjustment_prompt = ChatPromptTemplate.from_template("""
Detect the user's emotion as one of:
neutral, happy, sad, angry, afraid, confuse.

Rewrite the supplied base response only to make its tone natural and suitable
for the detected emotion. Do not mention the emotion label in the response.

Emotion selection rules:
- neutral for ordinary questions and messages without a clear emotional cue;
- happy for clear happiness, gratitude or satisfaction;
- sad for clear sadness, disappointment or emotional pain;
- angry for clear anger, frustration or hostility;
- afraid for clear fear, worry, anxiety or panic;
- confuse only for explicit confusion or a request for step-by-step help.

Tone rules:
- neutral: polite, clear and natural;
- happy: warm and positively engaged;
- sad: briefly acknowledge the difficulty and respond with empathy;
- angry: begin with one brief acknowledgement of the user's frustration, then
  calmly invite respectful, constructive communication before giving the
  available help;
- afraid: gentle reassurance without false promises, followed by a practical
  answer;
- confuse: use short numbered steps when helpful.

For every non-neutral emotion, the rewritten response must visibly reflect the
corresponding tone. Do not simply return a mechanical base response unchanged
unless it already contains the required emotional acknowledgement.

Mandatory fidelity rules:
- The base response is the authoritative answer. Change its tone only; do not
  reinterpret the user's question or replace the answer with a different one.
- Preserve every factual statement, doctor/patient name, appointment ID, IC,
  date, time, severity, telephone number, warning, option and confirmed system
  outcome from the base response.
- Do not add unsupported medical advice, promises, clinic services or actions.
- If the base response says that no relevant information was found, preserve
  that message without adding general advice, alternatives, clarification
  questions or offers of further help.
- Do not remove required instructions or change a completed/failed operation.
- When safety mode is true, keep emergency content direct and make only minimal
  tone changes.
- Use a known user name naturally only when relevant, not in every response.
- Respond in {response_language} and keep the response concise.

Return exactly:
Emotion: <emotion>
Response: <rewritten response>

User message:
{user_input}

Base response:
{base_response}

Known user state:
{conversation_state}

Recent conversation:
{conversation_history}

Safety mode:
{safety_mode}
""")
emotion_adjustment_chain = emotion_adjustment_prompt | model | StrOutputParser()


# -----------------------------------------------------------------------------
# SQL parameter extraction
# -----------------------------------------------------------------------------
def _doctor_mentions_from_query(text, lookup):
    mentions = []
    for match in re.finditer(r"\b(?:dr|doctor|doktor)\s*\.?\s*([a-z]+(?:\s+[a-z]+){0,2})", str(text or ""), re.I):
        candidate = match.group(1).strip()
        candidate = re.split(r"\b(?:ada|dan|pada|esok|lusa|minggu|boleh|slot|pukul)\b", candidate, 1, flags=re.I)[0].strip()
        if candidate:
            mentions.append(candidate)
    normalized_text = normalize_doctor_name(text)
    for full_name in lookup:
        tokens = [token for token in normalize_doctor_name(full_name).split() if len(token) >= 3]
        if any(re.search(rf"(?<!\w){re.escape(token)}(?!\w)", normalized_text) for token in tokens):
            mentions.extend(token for token in tokens if re.search(rf"(?<!\w){re.escape(token)}(?!\w)", normalized_text))
    return list(dict.fromkeys(mentions))


def extract_sql_parameters(compiled_query, lookup=None, reference_date=None):
    """LLM-first extraction, followed by fuzzy matching against active SQLite doctors."""
    if lookup is None:
        lookup = doctor_lookup
    current_date = reference_date or malaysia_now().date()
    if isinstance(current_date, datetime):
        current_date = current_date.date()
    if not isinstance(current_date, date):
        current_date = datetime.strptime(str(current_date), "%Y-%m-%d").date()

    raw_output = ""
    parse_error = False
    error_message = None
    try:
        raw_output = sql_parameter_chain.invoke({
            "compiled_query": compiled_query,
            "current_date": current_date.isoformat(),
            "doctor_names": "\n".join(f"- {name}" for name in lookup),
        }).strip()
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_output, flags=re.I)
        data = json.loads(cleaned)
    except Exception as error:
        data = {}
        parse_error = True
        error_message = f"{type(error).__name__}: {error}"

    doctor_mentions = data.get("doctor_mentions") or []
    if isinstance(doctor_mentions, str):
        doctor_mentions = [doctor_mentions]
    doctor_mentions = [str(item).strip() for item in doctor_mentions if str(item).strip()]

    # Preserve ambiguous aliases such as Dr Chong, which maps to two database doctors.
    alias_to_doctors = {}
    for full_name in lookup:
        for token in set(normalize_doctor_name(full_name).split()):
            if len(token) >= 3:
                alias_to_doctors.setdefault(token, set()).add(full_name)
    for alias, matching_names in alias_to_doctors.items():
        if len(matching_names) > 1 and re.search(
            rf"\b(?:dr|doctor|doktor)\s*\.?\s*{re.escape(alias)}\b",
            str(compiled_query or ""), re.I,
        ):
            doctor_mentions.append(alias)
    if not doctor_mentions:
        doctor_mentions = _doctor_mentions_from_query(compiled_query, lookup)
    doctor_mentions = list(dict.fromkeys(doctor_mentions))
    matched_doctors = match_doctor_mentions(doctor_mentions, lookup)

    target_date = data.get("target_date") or resolve_relative_or_explicit_date(compiled_query, current_date) or current_date.isoformat()
    try:
        target_date = datetime.strptime(str(target_date), "%Y-%m-%d").date().isoformat()
    except (TypeError, ValueError):
        target_date = resolve_relative_or_explicit_date(compiled_query, current_date) or current_date.isoformat()
        parse_error = True
        error_message = error_message or "LLM returned an invalid target_date."

    requested_time = data.get("requested_time")
    if requested_time:
        time_match = re.fullmatch(r"([01]?\d|2[0-3]):([0-5]\d)", str(requested_time).strip())
        if time_match:
            requested_time = f"{int(time_match.group(1)):02d}:{time_match.group(2)}"
        else:
            requested_time = None
            parse_error = True
            error_message = error_message or "LLM returned an invalid requested_time."

    return {
        "target_date": target_date,
        "doctor_mentions": doctor_mentions,
        "matched_doctors": matched_doctors,
        "doctor_ids": [item["doctor_id"] for item in matched_doctors],
        "doctor_names": [item["doctor_name"] for item in matched_doctors],
        "requested_time": requested_time,
        "doctor_match_threshold": DOCTOR_MATCH_THRESHOLD,
        "parameter_source": "llm+sqlite_doctor_fuzzy_match",
        "parse_error": parse_error,
        "error": error_message,
        "llm_raw": raw_output,
    }


def resolve_slot_query_parameters(user_input, compiled_query, lookup=None):
    combined_query = f"{user_input}\n{compiled_query}".strip()
    return extract_sql_parameters(combined_query, lookup)


def _doctor_display_name(doctor_name):
    value = str(doctor_name or "").strip()
    return value if re.match(r"^dr\.?\s", value, flags=re.I) else f"Dr. {value}"


def _localized_unavailable_reason(result, response_language):
    reason_code = result.get("reason_code")
    raw_reason = str(result.get("reason") or "").strip()
    chinese = response_language == "chinese"

    reason_map = {
        "PAST_DATE": ("所选日期已经过去", "tarikh yang dipilih telah berlalu"),
        "CLINIC_SCHEDULE_NOT_FOUND": ("当天没有诊所营业安排", "tiada jadual operasi klinik pada hari tersebut"),
        "CLINIC_CLOSED": ("诊所当天休息", "klinik ditutup pada hari tersebut"),
        "DOCTOR_REST_DAY": ("当天是该医生的休息日", "hari tersebut ialah hari rehat doktor"),
        "DOCTOR_ON_LEAVE": ("该医生当天请假", "doktor sedang bercuti pada hari tersebut"),
        "FULLY_BOOKED_OR_PASSED": ("当天的时段已满或剩余时段已经过去", "semua slot telah penuh atau waktu yang tinggal telah berlalu"),
    }
    if reason_code == "CLINIC_HOLIDAY":
        return f"当天是假期（{raw_reason}），诊所休息" if chinese else f"klinik ditutup kerana cuti {raw_reason}"
    pair = reason_map.get(reason_code)
    if pair:
        return pair[0] if chinese else pair[1]
    return raw_reason or ("没有可预约时段" if chinese else "tiada slot janji temu tersedia")


def _time_to_minutes(value):
    hour, minute = map(int, str(value).split(":"))
    return hour * 60 + minute


def retrieve_slots_from_parameters(parameters, lookup=None):
    if lookup is None:
        lookup = doctor_lookup
    target_date = parameters["target_date"]
    requested_time = parameters.get("requested_time")
    doctor_ids = parameters.get("doctor_ids") or list(lookup.values())
    doctor_results = []
    for doctor_id in dict.fromkeys(str(item) for item in doctor_ids):
        result = get_available_slots(target_date, doctor_id)
        result["doctor_name"] = doctor_name_from_id(doctor_id, lookup)
        slots = result.get("available_slots", [])
        result["status"] = "AVAILABLE" if slots else result.get("reason_code", "UNAVAILABLE")
        if requested_time and slots:
            if requested_time in slots:
                result.update({
                    "requested_time": requested_time,
                    "requested_time_available": True,
                    "suggested_slots": [],
                    "available_slots": [requested_time],
                    "status": "AVAILABLE",
                })
            else:
                requested_minutes = _time_to_minutes(requested_time)
                suggestions = [slot for slot in slots if _time_to_minutes(slot) > requested_minutes][:2]
                result.update({
                    "requested_time": requested_time,
                    "requested_time_available": False,
                    "suggested_slots": suggestions,
                    "available_slots": suggestions,
                    "status": "NEAREST_SLOTS" if suggestions else "REQUESTED_TIME_UNAVAILABLE",
                    "reason": "The requested time is unavailable.",
                })
        doctor_results.append(result)
    return {
        "status": "SLOT_RESULTS",
        "date": target_date,
        "requested_specific_doctor": bool(parameters.get("doctor_ids")),
        "requested_time": requested_time,
        "doctors": doctor_results,
    }


def format_slot_response(slot_result, response_language):
    chinese = str(response_language or "").lower() in {"zh", "chinese", "中文"}
    target_date = slot_result.get("date", "")
    lines = []
    for result in slot_result.get("doctors", []):
        name = _doctor_display_name(result.get("doctor_name") or result.get("doctor_id"))
        status = result.get("status")
        slots = result.get("available_slots", [])
        if status == "AVAILABLE" and slots:
            slot_text = "、".join(slots) if chinese else ", ".join(slots)
            lines.append(
                f"{name} 在 {target_date} 可预约的时间是：{slot_text}。" if chinese
                else f"Slot {name} yang tersedia pada {target_date}: {slot_text}."
            )
        elif status == "NEAREST_SLOTS" and slots:
            slot_text = "、".join(slots) if chinese else ", ".join(slots)
            requested = result.get("requested_time", "")
            lines.append(
                f"{name} 在 {target_date} 的 {requested} 不可预约；之后最接近的时间是：{slot_text}。" if chinese
                else f"Slot {requested} untuk {name} pada {target_date} tidak tersedia; slot terdekat selepas itu: {slot_text}."
            )
        else:
            reason = _localized_unavailable_reason(result, response_language)
            lines.append(
                f"{name} 在 {target_date} 没有可预约时段，因为{reason}。" if chinese
                else f"{name} tidak mempunyai slot pada {target_date} kerana {reason}."
            )
    if lines:
        return "\n".join(lines)
    return (
        f"{target_date} 暂时找不到可查询的医生预约资料。" if chinese
        else f"Tiada maklumat slot doktor yang dapat disemak pada {target_date}."
    )


def build_sql_only_slot_response(compiled_query, sql_params, response_language):
    slot_result = retrieve_slots_from_parameters(sql_params, doctor_lookup)
    retrieval = {
        "route": "1",
        "route_name": "SQLite only",
        "branch": "ask_slot_availability_sql_only",
        "sql_params": sql_params,
        "chroma": [],
        "sqlite": {"slot_availability": slot_result},
    }
    return {
        "compiled_query": compiled_query,
        "main_sections": None,
        "subsections": None,
        "retrieval": retrieval,
        "compiled_context": json.dumps(retrieval["sqlite"], ensure_ascii=False),
        "emotion": "neutral",
        "response": format_slot_response(slot_result, response_language),
    }


# -----------------------------------------------------------------------------
# Routed retrieval
# -----------------------------------------------------------------------------
def routed_retrieval(compiled_query, selected_subsections, lookup=None):
    if lookup is None:
        lookup = doctor_lookup

    if is_slot_availability_query(compiled_query):
        sql_params = extract_sql_parameters(compiled_query, lookup)
        slot_result = retrieve_slots_from_parameters(sql_params, lookup)
        return {
            "route": "1",
            "route_name": "SQLite only",
            "branch": "ask_slot_availability_sql_only",
            "sql_params": sql_params,
            "chroma": [],
            "sqlite": {"slot_availability": slot_result},
        }

    # Static handbook policy/procedure questions must never be diverted to SQLite.
    if STATIC_POLICY_TERMS.search(str(compiled_query or "")):
        retrieval_route = "0"
    else:
        retrieval_route = retrieval_router_chain.invoke({
            "compiled_query": compiled_query,
            "selected_subsections": selected_subsections or "",
        }).strip()

    if retrieval_route not in {"0", "1", "2"}:
        retrieval_route = "0"

    chroma_results = []
    sqlite_results = {}
    sql_params = None

    if retrieval_route in {"0", "2"}:
        chroma_results = retrieve_selected_subsections(
            compiled_query=compiled_query,
            selected_subsections=selected_subsections,
        )

    if retrieval_route in {"1", "2"}:
        # This branch is for non-slot dynamic information only.
        sqlite_results["available_doctors_today"] = get_available_doctors(
            malaysia_now().date().isoformat()
        )

    return {
        "route": retrieval_route,
        "branch": "general_ask_info",
        "sql_params": sql_params,
        "chroma": chroma_results,
        "sqlite": sqlite_results,
    }


def format_parallel_results(retrieval_results):
    parts = []

    chroma_results = retrieval_results.get("chroma", [])
    if chroma_results:
        parts.append("ChromaDB:")
        for result in chroma_results:
            parts.append(
                f'Section {result["section_number"]} - {result["section_title"]}\n'
                f'{result["document"]}'
            )

    sqlite_results = retrieval_results.get("sqlite", {})
    if sqlite_results:
        parts.append("SQLite:")
        for key, value in sqlite_results.items():
            parts.append(f"{key}: {value}")

    return "\n\n".join(parts)


def has_relevant_context(retrieval_results, min_similarity=MIN_RAG_SIMILARITY):
    """Accept SQLite facts or sufficiently similar handbook chunks only."""
    if retrieval_results.get("sqlite"):
        return True
    similarities = [
        float(item.get("similarity", 0.0))
        for item in retrieval_results.get("chroma", [])
    ]
    return bool(similarities and max(similarities) >= float(min_similarity))


def no_relevant_information_response(response_language):
    """Return a deterministic scope-safe response without invoking generation."""
    language = str(response_language or "").strip().lower()
    if language in {"zh", "chinese", "中文", "simplified chinese"}:
        return (
            "很抱歉，Klinik Chong 的资料库中没有找到相关资料。"
            "此聊天机器人只提供诊所服务、预约，以及资料库涵盖的基本健康信息。"
        )
    return (
        "Maaf, maklumat berkaitan tidak ditemui dalam pangkalan pengetahuan "
        "Klinik Chong. Chatbot ini hanya menyediakan maklumat tentang "
        "perkhidmatan klinik, janji temu dan maklumat kesihatan asas yang disokong."
    )


def _state_json(value):
    return json.dumps(value or {}, ensure_ascii=False, default=str)


def _history_text(history):
    if isinstance(history, str):
        return history.strip()
    lines = []
    for item in (history or [])[-20:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "user").strip().title()
        content = str(item.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _extract_json_object(output):
    text = str(output or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {}
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _validated_explicit_user_name(user_input, candidate):
    """Reject conditions and roles mistakenly extracted as a person's name."""
    text = str(user_input or "").strip()
    candidate = re.sub(r"\s+", " ", str(candidate or "")).strip(" .,!?:;，。！？：；")
    if not candidate:
        return ""

    identity_question = re.compile(
        r"(?:你(?:还|還)?(?:记得|記得|知道)(?:我是谁|我是誰|我叫什么|我叫什麼|我的名字)|"
        r"我(?:叫什么|叫什麼)(?:名字)?(?:吗|嗎|呢)?|我的名字(?:是)?(?:什么|什麼|谁|誰|[?？])|"
        r"who\s+am\s+i|what(?:'s|\s+is)\s+my\s+name|do\s+you\s+(?:know|remember)\s+my\s+name|"
        r"siapa\s+saya|siapa\s+nama\s+saya|nama\s+saya\s+siapa|"
        r"(?:awak|kamu)\s+(?:tahu|ingat)\s+nama\s+saya)",
        re.I,
    )
    if identity_question.search(text):
        return ""

    strong_patterns = [
        r"(?:my\s+name\s+is|(?:you\s+can\s+)?call\s+me)\s*[:：]?\s*([^,.!?，。！？]{1,60})",
        r"(?:nama\s+(?:saya|sy)|panggil\s+(?:saya|sy)|(?:saya|sy)\s+(?:bernama|nama))\s*[:：]?\s*([^,.!?，。！？]{1,60})",
        r"(?:我的名字是|我叫)\s*[:：]?\s*([^，。！？,.!?]{1,30})",
    ]
    stated = ""
    for pattern in strong_patterns:
        match = re.search(pattern, text, re.I)
        if match:
            stated = match.group(1)
            break

    if not stated:
        weak = re.search(
            r"^(?:hello|hi|hai)?\s*(?:i(?:'m|\s+am)|我是|(?:saya|sy)\s+(?:ialah|adalah))\s+([^,.!?，。！？]{1,60})",
            text,
            re.I,
        )
        if weak:
            stated = weak.group(1)

    stated = re.split(
        r"\s+(?:and|dan|then|but|because|need|want|have|feel|perlukan|mahu|nak|rasa)\b",
        stated,
        1,
        flags=re.I,
    )[0]
    stated = re.sub(
        r"\s*(?:你呢|你叫什么|你叫什麼|awak\s+pula|kamu\s+pula|how\s+about\s+you)\s*$",
        "",
        stated,
        flags=re.I,
    )
    stated = re.sub(r"\s+", " ", stated).strip(" .,!?:;，。！？：；")
    non_name = re.compile(
        r"^(?:very|really|quite|so|too|tired|sick|ill|unwell|sad|angry|afraid|"
        r"worried|dizzy|hungry|in\s+pain|not\s+well|a\s+patient|the\s+patient|"
        r"很累|不舒服|生病|病人|害怕|担心|想|要|需要|什么|什麼|谁|誰|哪|penat|sakit|sedih|marah|"
        r"takut|risau|pesakit|mahu|nak)\b",
        re.I,
    )
    if not stated or non_name.search(stated) or re.search(
        r"(?:[?？]|叫什么|叫什麼|什么名字|什麼名字)", stated, re.I
    ):
        return ""
    if len(stated) > 60 or len(stated.split()) > 5:
        return ""
    return stated


def parse_emotion_response(output, fallback_response=""):
    emotion = "neutral"
    response_lines = []
    response_started = False
    raw_output = str(output or "").strip()

    for raw_line in raw_output.splitlines():
        line = raw_line.strip()
        clean_line = line.replace("**", "").strip()
        emotion_match = re.match(
            r"^(?:emotion|emosi|情绪|情緒)\s*[:：]\s*(.+)$",
            clean_line,
            re.I,
        )
        response_match = re.match(
            r"^(?:response|respons|jawapan|回复|回覆|回答)\s*[:：]\s*(.*)$",
            clean_line,
            re.I,
        )
        if emotion_match:
            candidate = emotion_match.group(1).strip().lower()
            if candidate in {"neutral", "happy", "sad", "angry", "afraid", "confuse"}:
                emotion = candidate
        elif response_match:
            response_started = True
            value = response_match.group(1).strip()
            if value:
                response_lines.append(value)
        elif response_started and line:
            response_lines.append(line)

    response = "\n".join(response_lines).strip()
    # Some models return the requested answer without the literal `Response:`
    # label, or localise the label despite the formatting instruction.  The
    # previous parser discarded that valid answer and fell back to the Malay
    # retrieved context.  Preserve any substantive unlabelled output instead.
    if not response and raw_output:
        unlabelled_lines = []
        for raw_line in raw_output.splitlines():
            clean_line = raw_line.replace("**", "").strip()
            if not clean_line:
                continue
            if re.match(r"^(?:emotion|emosi|情绪|情緒)\s*[:：]", clean_line, re.I):
                continue
            if clean_line.lower() in {"neutral", "happy", "sad", "angry", "afraid", "confuse"}:
                continue
            unlabelled_lines.append(raw_line.strip())
        response = "\n".join(unlabelled_lines).strip()
    if not response:
        response = str(fallback_response or "").strip()
    return {"emotion": emotion, "response": response}


def response_language_instruction(response_language):
    """Turn the API language value into an explicit generation constraint."""
    language = str(response_language or "").strip().lower()
    if language in {"zh", "chinese", "中文", "simplified chinese"}:
        return (
            "Simplified Chinese only (简体中文). Do not answer in Bahasa Melayu "
            "or English, except for official names that must be preserved."
        )
    return (
        "Bahasa Melayu only. Do not answer in Chinese or English, except for "
        "official names that must be preserved."
    )


def enforce_grounded_response(base_response, generated_response, query_text=""):
    """Prevent the tone-generation layer from contradicting retrieved facts."""
    base = str(base_response or "").strip()
    generated = str(generated_response or "").strip()
    query = str(query_text or "")
    if not base or not generated:
        return base or generated

    unavailable_pattern = re.compile(
        r"(?:无法(?:直接)?(?:定位|获取|取得|确定|找到|提供|访问|查看)|"
        r"不能(?:定位|获取|确定|找到|提供)|"
        r"tidak\s+dapat\s+(?:mengesan|mendapatkan|menentukan|mencari|memberikan)|"
        r"(?:cannot|unable\s+to|could\s+not)\s+(?:locate|access|find|determine|provide))",
        re.I,
    )
    if unavailable_pattern.search(generated) and not unavailable_pattern.search(base):
        return base

    location_query = re.search(
        r"(?:诊所|診所|klinik|clinic).{0,20}(?:哪里|哪裡|地址|lokasi|alamat|where)|"
        r"(?:哪里|哪裡|地址|lokasi|alamat|where).{0,20}(?:诊所|診所|klinik|clinic)",
        query,
        re.I,
    )
    base_has_address = re.search(r"\b\d{1,5}\s*,?\s*(?:Jalan|Jln\.?|Lorong)\b", base, re.I)
    if location_query and base_has_address and not re.search(r"\b(?:Jalan|Jln\.?|Lorong)\b", generated, re.I):
        return base

    return generated


def extract_conversation_state(user_input, conversation_state=None, conversation_history=None):
    """Return a per-session state update; no state is stored globally."""
    existing = dict(conversation_state or {})
    raw = conversation_state_chain.invoke({
        "conversation_state": _state_json(existing),
        "conversation_history": _history_text(conversation_history),
        "user_input": user_input,
    })
    extracted = _extract_json_object(raw)
    allowed = {
        "user_name", "preferred_language", "doctor_name", "appointment_date",
        "appointment_time", "symptoms", "duration", "severity",
    }
    updated = dict(existing)
    for key in allowed:
        value = extracted.get(key)
        if key == "user_name":
            value = _validated_explicit_user_name(user_input, value)
        if value is None or value == "" or value == []:
            continue
        updated[key] = value
    return {"conversation_state": updated, "extracted": extracted}


def generate_emotion_aware_reply(
    user_input,
    base_response,
    response_language,
    conversation_state=None,
    conversation_history=None,
    safety_critical=False,
):
    """Adjust tone while keeping the supplied reply facts and actions intact."""
    fallback = str(base_response or "").strip()
    if not fallback:
        return {"emotion": "neutral", "response": ""}
    raw = emotion_adjustment_chain.invoke({
        "user_input": user_input,
        "base_response": fallback,
        "response_language": response_language_instruction(response_language),
        "conversation_state": _state_json(conversation_state),
        "conversation_history": _history_text(conversation_history),
        "safety_mode": "true" if safety_critical else "false",
    })
    parsed = parse_emotion_response(raw, fallback_response=fallback)
    parsed["response"] = enforce_grounded_response(
        fallback,
        parsed["response"],
        query_text=user_input,
    )
    return parsed


# -----------------------------------------------------------------------------
# Combined ask_info RAG pipeline
# -----------------------------------------------------------------------------
def _selected_section_list(raw_selection):
    selection = str(raw_selection or "").strip()
    if not selection or "NONE" in selection.upper():
        return []
    return [item.strip() for item in selection.split(",") if item.strip()]


def _unsupported_result(
    compiled_query,
    response_language,
    conversation_state=None,
    main_sections=None,
    subsections=None,
):
    response = no_relevant_information_response(response_language)
    return {
        "compiled_query": compiled_query,
        "main_sections": main_sections,
        "subsections": subsections,
        "retrieval": {
            "route": None,
            "route_name": "No relevant knowledge-base context",
            "branch": "no_relevant_context",
            "sql_params": None,
            "chroma": [],
            "sqlite": {},
        },
        "compiled_context": "NO_RELEVANT_CONTEXT",
        "emotion": "neutral",
        "response": response,
        "emotion_applied": False,
        "conversation_state": conversation_state or {},
    }


def run_rag_pipeline(
    user_input,
    response_language,
    past_queries="",
    intent_hint="ask_info",
    conversation_state=None,
    conversation_history=None,
):
    history_text = _history_text(conversation_history) or str(past_queries or "").strip()
    compiled_query = query_compiler_chain.invoke({
        "past_queries": history_text,
        "input_query": user_input,
        "response_language": response_language,
    }).strip()

    doctor_directory_mode = doctor_directory_query_mode(
        user_input=user_input,
        compiled_query=compiled_query,
        conversation_history=history_text,
    )
    if doctor_directory_mode:
        active_doctors = get_active_doctors()
        base_response = format_active_doctor_response(
            active_doctors,
            response_language,
            doctor_directory_mode,
        )
        adjusted = generate_emotion_aware_reply(
            user_input=user_input,
            base_response=base_response,
            response_language=response_language,
            conversation_state=conversation_state,
            conversation_history=conversation_history,
        )
        return {
            "compiled_query": compiled_query,
            "main_sections": None,
            "subsections": None,
            "retrieval": {
                "route": "1",
                "route_name": "SQLite only",
                "branch": "active_doctor_directory_sql_only",
                "sqlite": {"active_doctors": active_doctors, "count": len(active_doctors)},
                "chroma": [],
                "sql_params": None,
            },
            "compiled_context": base_response,
            "emotion": adjusted["emotion"],
            "response": adjusted["response"],
            "emotion_applied": True,
            "conversation_state": conversation_state or {},
        }

    combined_availability_query = f"{user_input}\n{compiled_query}".strip()
    if is_slot_availability_query(combined_availability_query, intention=intent_hint):
        sql_params = resolve_slot_query_parameters(
            user_input=user_input,
            compiled_query=compiled_query,
            lookup=doctor_lookup,
        )
        slot_response = build_sql_only_slot_response(
            compiled_query=compiled_query,
            sql_params=sql_params,
            response_language=response_language,
        )
        adjusted = generate_emotion_aware_reply(
            user_input=user_input,
            base_response=slot_response["response"],
            response_language=response_language,
            conversation_state=conversation_state,
            conversation_history=conversation_history,
        )
        slot_response.update(adjusted)
        slot_response["emotion_applied"] = True
        slot_response["conversation_state"] = conversation_state or {}
        return slot_response

    operating_hours_query = bool(re.search(
        r"(?i)(?:营业|營業|开门|開門|关门|關門|休息|几点|幾點|"
        r"waktu\s+operasi|bila\s+(?:buka|tutup|rehat)|"
        r"pukul\s+berapa|masa\s+(?:buka|tutup|rehat)|"
        r"opening\s+hours?|closing\s+time|lunch\s+break)",
        user_input,
    ))
    if operating_hours_query:
        compiled_query = (
            f"{compiled_query}\nKlinik Chong operating hours, opening time, "
            "closing time and lunch/rest break schedule."
        )

    selected_main_sections = main_section_analyzer_chain.invoke({
        "compiled_query": compiled_query,
        "main_sections": main_sections_text,
    })
    if operating_hours_query:
        selected_main_sections = "2"

    main_section_list = _selected_section_list(selected_main_sections)
    if not main_section_list:
        return _unsupported_result(
            compiled_query=compiled_query,
            response_language=response_language,
            conversation_state=conversation_state,
            main_sections=selected_main_sections,
        )
    subsections_text = build_subsections_text(main_section_list)

    selected_subsections = subsection_analyzer_chain.invoke({
        "compiled_query": compiled_query,
        "subsections": subsections_text,
    }).strip()
    if not _selected_section_list(selected_subsections):
        return _unsupported_result(
            compiled_query=compiled_query,
            response_language=response_language,
            conversation_state=conversation_state,
            main_sections=selected_main_sections,
            subsections=selected_subsections,
        )

    retrieval_results = routed_retrieval(
        compiled_query=compiled_query,
        selected_subsections=selected_subsections,
        lookup=doctor_lookup,
    )
    if operating_hours_query:
        retrieval_results["route"] = "0"
        retrieval_results["chroma"] = retrieve_chunks(
            compiled_query,
            top_k=5,
            where={"chapter_number": "2"},
        )
        retrieval_results["sqlite"] = {}

    if not has_relevant_context(retrieval_results):
        return _unsupported_result(
            compiled_query=compiled_query,
            response_language=response_language,
            conversation_state=conversation_state,
            main_sections=selected_main_sections,
            subsections=selected_subsections,
        )

    retrieved_context = format_parallel_results(retrieval_results)

    compiled_context = context_compiler_chain.invoke({
        "compiled_query": compiled_query,
        "retrieved_context": retrieved_context,
    }).strip()

    if not compiled_context or "NO_RELEVANT_CONTEXT" in compiled_context.upper():
        return _unsupported_result(
            compiled_query=compiled_query,
            response_language=response_language,
            conversation_state=conversation_state,
            main_sections=selected_main_sections,
            subsections=selected_subsections,
        )

    raw_final_output = emotion_response_chain.invoke({
        "user_input": user_input,
        "compiled_context": compiled_context,
        "response_language": response_language_instruction(response_language),
        "conversation_state": _state_json(conversation_state),
        "conversation_history": history_text,
    })
    final_output = parse_emotion_response(raw_final_output, fallback_response=compiled_context)
    final_output["response"] = enforce_grounded_response(
        compiled_context,
        final_output["response"],
        query_text=f"{user_input}\n{compiled_query}",
    )

    return {
        "compiled_query": compiled_query,
        "main_sections": selected_main_sections,
        "subsections": selected_subsections,
        "retrieval": retrieval_results,
        "compiled_context": compiled_context,
        "emotion": final_output["emotion"],
        "response": final_output["response"],
        "emotion_applied": True,
        "conversation_state": conversation_state or {},
    }


__all__ = [
    "run_rag_pipeline",
    "extract_conversation_state",
    "generate_emotion_aware_reply",
    "enforce_grounded_response",
    "has_relevant_context",
    "no_relevant_information_response",
    "retrieve_chunks",
    "retrieve_selected_subsections",
    "routed_retrieval",
    "extract_sql_parameters",
    "get_available_doctors",
    "get_active_doctors",
    "doctor_directory_query_mode",
    "get_available_slots",
    "build_doctor_lookup",
    "match_doctor_name",
    "doctor_lookup",
    "collection",
]
