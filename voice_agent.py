"""Conversational Urdu voice-agent helpers for Khaansi AI.

Stack (decided architecture — Qwen's native voice models don't support Urdu):
  STT: faster-whisper (small, CPU int8) — strong Urdu support
  LLM: Qwen via DashScope — symptom parsing + Urdu dialogue and explanations
  TTS: Edge-TTS — ur-PK-AsadNeural / ur-PK-UzmaNeural
"""

import io
import json
import os

import edge_tts

# ----------------------------
# Spoken Urdu strings (agent-side)
# ----------------------------

AGENT_GENDERS = ("female", "male")

AGENT_LINES = {
    "greeting": {
        "female": (
            "السلام علیکم! میں کھانسی اے آئی ہوں۔ میں آپ کی کھانسی کی آواز سن کر "
            "اور چند سوالات کے ذریعے آپ کی ٹی بی کی ابتدائی جانچ کرتی ہوں۔ "
            "سب سے پہلے، نیچے بٹن دبا کر اپنی کھانسی ریکارڈ کروائیں۔"
        ),
        "male": (
            "السلام علیکم! میں کھانسی اے آئی ہوں۔ میں آپ کی کھانسی کی آواز سن کر "
            "اور چند سوالات کے ذریعے آپ کی ٹی بی کی ابتدائی جانچ کرتا ہوں۔ "
            "سب سے پہلے، نیچے بٹن دبا کر اپنی کھانسی ریکارڈ کروائیں۔"
        ),
    },
    "thanks_after_cough": {
        "female": "شکریہ! آپ کی کھانسی ریکارڈ ہو گئی ہے۔ اب میں آپ سے کچھ سوالات پوچھوں گی۔",
        "male": "شکریہ! آپ کی کھانسی ریکارڈ ہو گئی ہے۔ اب میں آپ سے کچھ سوالات پوچھوں گا۔",
    },
    "done_ack": {
        "female": "شکریہ! آپ کے تمام جوابات مل گئے۔ اب میں آپ کی کھانسی اور جوابات کا تجزیہ کر رہی ہوں۔",
        "male": "شکریہ! آپ کے تمام جوابات مل گئے۔ اب میں آپ کی کھانسی اور جوابات کا تجزیہ کر رہا ہوں۔",
    },
    "retry_prefix": {
        "female": "معاف کیجیے گا، مجھے آواز اچھی طرح سمجھ نہیں آئی۔ میں دوبارہ پوچھتی ہوں: ",
        "male": "معاف کیجیے گا، مجھے آواز اچھی طرح سمجھ نہیں آئی۔ میں دوبارہ پوچھتا ہوں: ",
    },
}


def _validate_agent_gender(agent_gender: str) -> str:
    if agent_gender not in AGENT_GENDERS:
        raise ValueError(f"unsupported agent gender: {agent_gender!r}")
    return agent_gender


def agent_line(name: str, agent_gender: str) -> str:
    return AGENT_LINES[name][_validate_agent_gender(agent_gender)]


# ordered must-ask questions: (field_key, urdu_question)
MUST_ASK = [
    ("sex", "بتائیے، آپ مرد ہیں یا عورت؟"),
    ("age", "آپ کی عمر کتنی ہے؟"),
    ("reported_cough_dur", "آپ کی یہ کھانسی کتنے دن، ہفتے یا مہینے سے ہے؟"),
    ("hemoptysis", "کیا آپ کو کھانسی میں خون آتا ہے یا نہیں؟"),
    ("fever", "کیا آپ کو بخار ہے یا نہیں؟"),
    ("night_sweats", "کیا آپ کو رات کو پسینہ آتا ہے یا نہیں؟"),
    ("weight_loss", "کیا آپ کا وزن بغیر کسی وجہ کے کم ہوا ہے یا نہیں؟"),
    ("tb_prior", "کیا آپ کو پہلے کبھی ٹی بی ہوئی ہے یا نہیں؟"),
    ("tb_prior_type", "اگر ٹی بی ہوئی تھی، تو کیا وہ پھیپھڑوں میں تھی یا جسم کے کسی اور حصے میں؟"),
    ("smoke_lweek", "کیا آپ نے پچھلے ہفتے سگریٹ یا تمباکو استعمال کیا ہے یا نہیں؟"),
]

# ----------------------------
# Answer state -> symptom vector
# ----------------------------

def field_answered(answers: dict, field: str) -> bool:
    if field == "tb_prior_type":
        return answers.get("tb_prior_type") is not None or answers.get("tb_prior") is False
    return answers.get(field) is not None


def build_symptom_values(answers: dict) -> dict:
    """Voice-agent answers -> the 16-field symptom dict.
    Nice-to-have fields and unanswered must-asks stay NaN for the imputer."""
    import numpy as np

    def b01(v):
        if v is True:
            return 1.0
        if v is False:
            return 0.0
        return np.nan

    sex = answers.get("sex")
    tb_prior = answers.get("tb_prior")
    tb_type = answers.get("tb_prior_type")
    if tb_prior is False:
        pul = extrap = unk = 0.0
    elif tb_prior is True:
        pul = 1.0 if tb_type == "pulmonary" else 0.0
        extrap = 1.0 if tb_type == "extrapulmonary" else 0.0
        unk = 1.0 if tb_type in (None, "unknown") else 0.0
    else:
        pul = extrap = unk = np.nan

    def num(key):
        v = answers.get(key)
        return np.nan if v is None else float(v)

    return {
        "sex": np.nan if sex is None else (1.0 if sex == "male" else 0.0),
        "age": num("age"),
        "height": np.nan,
        "weight": np.nan,
        "reported_cough_dur": num("reported_cough_dur"),
        "tb_prior": b01(tb_prior),
        "tb_prior_Pul": pul,
        "tb_prior_Extrapul": extrap,
        "tb_prior_Unknown": unk,
        "hemoptysis": b01(answers.get("hemoptysis")),
        "heart_rate": np.nan,
        "temperature": np.nan,
        "weight_loss": b01(answers.get("weight_loss")),
        "smoke_lweek": b01(answers.get("smoke_lweek")),
        "fever": b01(answers.get("fever")),
        "night_sweats": b01(answers.get("night_sweats")),
    }


# ----------------------------
# STT — faster-whisper
# ----------------------------

_whisper_model = None

# "base" misheard numbers often (22 -> 2, hafta -> mahina). "small" is what
# test_whisper.py already uses. Try "medium" if your laptop is fast enough.
WHISPER_MODEL_SIZE = "small"

# Vocabulary hint only (numbers + time units), NOT a full sentence, so Whisper
# is nudged toward them without echoing a plausible answer back.
WHISPER_PROMPT = (
    "ایک، دو، تین، چار، پانچ، دس، بیس، بائیس، پچیس، تیس، پینتیس، چالیس، پچاس۔ "
    "دن، ہفتہ، ہفتے، مہینہ، مہینے، سال۔"
)


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel

        _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    return _whisper_model


def transcribe(audio_bytes: bytes) -> str:
    model = get_whisper_model()
    segments, _ = model.transcribe(
        io.BytesIO(audio_bytes),
        language="ur",
        beam_size=5,                       # more careful decoding
        vad_filter=True,                   # ignore silence / background noise
        condition_on_previous_text=False,  # avoid repeated-text drift
        initial_prompt=WHISPER_PROMPT,     # bias toward numbers and time units
    )
    return " ".join(s.text.strip() for s in segments).strip()


# ----------------------------
# TTS — Edge-TTS
# ----------------------------

_tts_cache: dict = {}


def synthesize(text: str, voice: str = "ur-PK-UzmaNeural") -> bytes:
    """Urdu text -> mp3 bytes. Cached per (voice, text)."""
    key = (voice, text)
    if key in _tts_cache:
        return _tts_cache[key]

    import asyncio

    async def _gen():
        communicate = edge_tts.Communicate(text, voice)
        buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        return buf.getvalue()

    audio = asyncio.run(_gen())
    _tts_cache[key] = audio
    return audio


# ----------------------------
# LLM — Qwen via DashScope
# ----------------------------

QWEN_MODEL = "qwen-plus-character"

SYSTEM_PROMPT = """You are the voice assistant for Khaansi AI, a TB (tuberculosis) screening aid used in Pakistan. You speak ONLY Pakistani Urdu (Urdu script) with the patient. You are warm and use simple everyday words — the patient may be uneducated.

You will receive a JSON object with:
- current_question: the symptom question the app just asked (in Urdu),
- patient_answer: the patient's reply in Urdu (speech-to-text output, may contain spelling errors),
- fields_to_extract: symptom fields the app still needs. Extract from this answer ONLY what it actually addresses.

Respond with ONLY a JSON object, no other text:
{
  "speak": "<ONE short sentence in Pakistani Urdu (Urdu script) warmly acknowledging the patient's answer. Do NOT ask any question.>",
  "extracted": {"<field>": <value or null>}
}

Extraction rules:
- sex: "male", "female", or null.
- age: the exact number of years the patient said, or null. Convert Urdu number words to digits (e.g. بائیس = 22, پچیس = 25, تیس = 30, پینتیس = 35). Never return a single digit unless the patient actually said a single-digit number.
- reported_cough_dur: number of days, or null. ہفتہ/ہفتے = week (7 days), مہینہ/مہینے = month (30 days); do not confuse them. Examples: "ایک ہفتہ" = 7, "دو ہفتے" = 14, "تین ہفتے" = 21, "ایک مہینہ" = 30, "دو مہینے" = 60.
- hemoptysis, fever, night_sweats, weight_loss, tb_prior, smoke_lweek: true, false, or null.
- tb_prior_type: "pulmonary", "extrapulmonary", or "unknown", or null.
- Use null for every field the patient's answer does not clearly address. Never guess or invent values.
- Handle speech-recognition spelling errors (e.g. خانسی = کھانسی, سہت = صحت, ٹیسٹ/ٹیست)."""


def _gender_instruction(agent_gender: str) -> str:
    gender = _validate_agent_gender(agent_gender)
    if gender == "female":
        return (
            "The selected assistant voice is female. In every first-person Urdu phrase, "
            "use feminine grammar consistently, including forms such as کرتی ہوں, پوچھوں گی, "
            "سمجھ سکتی ہوں, and کر رہی ہوں. Never use masculine first-person forms."
        )
    return (
        "The selected assistant voice is male. In every first-person Urdu phrase, "
        "use masculine grammar consistently, including forms such as کرتا ہوں, پوچھوں گا, "
        "سمجھ سکتا ہوں, and کر رہا ہوں. Never use feminine first-person forms."
    )


def llm_available() -> bool:
    from dotenv import load_dotenv

    load_dotenv()
    return bool(os.environ.get("DASHSCOPE_API_KEY"))


def _call(messages, temperature: float = 0.3) -> str:
    import requests
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    base_url = os.environ.get(
        "DASHSCOPE_COMPATIBLE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    resp = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": QWEN_MODEL, "messages": messages, "temperature": temperature},
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"DashScope error {resp.status_code}: {resp.text[:300]}")
    return resp.json()["choices"][0]["message"]["content"]


def _parse_json_block(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


AGE_RANGE = (5, 110)      # an "age 2" is almost surely a mis-heard "22" -> re-ask
COUGH_DAYS_RANGE = (0, 365)


def sanitize_extracted(extracted: dict) -> dict:
    """Drop implausible numbers (set to None) so the app re-asks instead of
    silently scoring on a mis-heard value. Confirmation screen catches the rest."""
    out = dict(extracted)

    def _in_range(key, lo, hi):
        v = out.get(key)
        if v is None:
            return
        try:
            v = float(v)
        except (TypeError, ValueError):
            out[key] = None
            return
        out[key] = v if lo <= v <= hi else None

    _in_range("age", *AGE_RANGE)
    _in_range("reported_cough_dur", *COUGH_DAYS_RANGE)
    return out


def parse_answer(current_question: str, patient_answer: str, fields_to_extract: list,
                 *, agent_gender: str) -> dict:
    """One Qwen call per turn: returns {"speak": str, "extracted": dict}.
    The app (not Qwen) decides which question to ask next."""
    user_msg = json.dumps({
        "current_question": current_question,
        "patient_answer": patient_answer,
        "fields_to_extract": fields_to_extract,
    }, ensure_ascii=False)
    out = _call([
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\n{_gender_instruction(agent_gender)}"},
        {"role": "user", "content": user_msg},
    ])
    data = _parse_json_block(out)
    if "speak" not in data or "extracted" not in data:
        raise ValueError(f"unexpected Qwen output: {out!r}")
    data["extracted"] = sanitize_extracted(data["extracted"])
    return data


def generate_explanation(prob: float, threshold: float, *, agent_gender: str) -> str:
    """Risk score -> plain-language Urdu explanation. Never a diagnosis."""
    flagged = prob > threshold
    if flagged:
        situation = (
            f"- TB risk probability: {prob:.0%}\n"
            f"- Screening threshold: {threshold:.1%}\n"
            "- Result: ELEVATED risk (above threshold)."
        )
        guidance = (
            "1. Acknowledge they finished the screening.\n"
            "2. Tell them honestly but gently that some signs point toward a possible risk.\n"
            "3. Urge them to visit a nearby clinic or hospital soon for a confirmatory test (sputum test or chest X-ray).\n"
            "4. Remind them this is NOT a diagnosis, only an early check.\n"
            "5. End warmly."
        )
    else:
        situation = (
            f"- TB risk probability: {prob:.0%}\n"
            f"- Screening threshold: {threshold:.1%}\n"
            "- Result: NO elevated risk signal (below threshold)."
        )
        guidance = (
            "1. Acknowledge they finished the screening.\n"
            "2. Reassure them that no warning signs were found.\n"
            "3. Advise: if the cough lasts more than two weeks or worsens, see a doctor.\n"
            "4. Remind them this is NOT a diagnosis, only an early check.\n"
            "5. End warmly."
        )
    prompt = f"""You are the voice assistant for Khaansi AI, a TB screening aid used in Pakistan. The patient has just completed a cough-audio analysis and symptom questions. The screening model produced:
{situation}

Write what you will say to the patient, in simple Pakistani Urdu (Urdu script), 4-6 short sentences:
{guidance}

{_gender_instruction(agent_gender)}
Return ONLY the Urdu text to speak."""
    return _call([{"role": "user", "content": prompt}], temperature=0.7)
