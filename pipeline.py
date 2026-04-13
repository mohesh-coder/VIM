"""
pipeline.py — VendorPulse Document Ingestion Pipeline (Person 1)

Two upload modes:
  "baseline" → extract six contract fields → memory.write_baseline()
  "normal"   → extract facts + drift score → memory.append_event()

Uses hindsight.store() and hindsight.append() via memory.py.
"""

import json
import re
import os
from typing import Optional
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


def get_groq() -> Groq:
    return Groq(api_key=os.getenv("GROQ_API_KEY", ""))


MODEL = "qwen/qwen3-32b"


# ── Prompts ───────────────────────────────────────────────────────────────────

BASELINE_EXTRACTOR_SYSTEM = """You are a procurement intelligence analyst reading an original vendor contract.

Extract the baseline terms and return ONLY valid JSON — no markdown, no preamble:
{
  "vendor_name": "full legal vendor company name from contract",
  "agreed_price": "the exact pricing agreed at signing, with units e.g. $0.0075 per SMS",
  "sla_terms": "the SLA commitment verbatim from the contract e.g. 99.95% uptime guaranteed",
  "contract_start": "YYYY-MM-DD",
  "renewal_date": "YYYY-MM-DD",
  "rep_name": "name of the vendor sales or account rep who signed"
}

Rules:
- agreed_price: the single most important field. Include the exact number and unit.
- sla_terms: use a verbatim quote from the contract, not a paraphrase.
- If any field is missing from the contract, use empty string "".
- All dates must be YYYY-MM-DD format.
- Return only the JSON object above, nothing else."""


EVENT_EXTRACTOR_SYSTEM = """You are a procurement intelligence analyst. Read the vendor document and extract structured facts.

Return ONLY valid JSON — no markdown, no preamble:
{
  "doc_date": "YYYY-MM-DD",
  "vendor_name": "full vendor company name",
  "facts": [
    {
      "fact_type": "price_change | sla_violation | term_change | commitment | incident",
      "value": "the specific value, term, or statement",
      "summary": "one sentence plain English description",
      "confidence": 0.95
    }
  ]
}

fact_type rules (use EXACTLY these values):
- "price_change"   → price moved vs what was originally agreed
- "sla_violation"  → uptime or response time breach
- "term_change"    → clause added, removed, or reworded in contract/renewal
- "commitment"     → verbal or written promise by vendor rep
- "incident"       → outage, degradation, or error event

Only include facts with confidence >= 0.7.
Return {"doc_date": null, "vendor_name": null, "facts": []} if nothing relevant found."""


DRIFT_CLASSIFIER_SYSTEM = """You are a procurement risk analyst. Compare a new vendor fact against the baseline contract terms.

Return ONLY valid JSON — no markdown:
{
  "is_drift": true,
  "severity": 3,
  "delta_summary": "one sentence explaining the change vs baseline",
  "action": "what the procurement team should do"
}

Severity scale:
1 = minor variation, within normal range
2 = notable deviation worth tracking
3 = material breach or significant change requiring immediate attention

If is_drift is false, return severity 0 and empty strings for delta_summary and action."""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _call_groq(system: str, user_msg: str) -> str:
    try:
        client = get_groq()
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=2000,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {e}"


def _parse_json(text: str) -> Optional[dict]:
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Extract plain text from uploaded file bytes."""
    fname = filename.lower()
    if fname.endswith(".pdf"):
        try:
            import pdfplumber, io
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages)
        except ImportError:
            pass
    return file_bytes.decode("utf-8", errors="ignore")


def extract_baseline(text: str, filename: str) -> Optional[dict]:
    """Use Groq to extract the six baseline fields from a contract document."""
    prompt = f"Contract document ({filename}):\n\n{text[:4000]}"
    raw    = _call_groq(BASELINE_EXTRACTOR_SYSTEM, prompt)
    return _parse_json(raw)


def extract_events(text: str, filename: str) -> Optional[dict]:
    """Use Groq to extract structured facts from any post-contract document."""
    prompt = f"Document ({filename}):\n\n{text[:4000]}"
    raw    = _call_groq(EVENT_EXTRACTOR_SYSTEM, prompt)
    return _parse_json(raw)


def classify_drift(baseline: dict, fact: dict) -> Optional[dict]:
    """Compare a new fact against the vendor baseline and return drift severity."""
    prompt = (
        f"Baseline contract terms:\n{json.dumps(baseline, indent=2)}\n\n"
        f"New fact from recent document:\n"
        f"fact_type: {fact.get('fact_type')}\n"
        f"value: {fact.get('value')}\n"
        f"summary: {fact.get('summary')}"
    )
    raw = _call_groq(DRIFT_CLASSIFIER_SYSTEM, prompt)
    return _parse_json(raw)


def is_duplicate(event: dict, existing_events: list) -> bool:
    """
    Per §3.5: discard if fact_type + value + date all match an existing event.
    """
    for e in existing_events:
        if (
            e.get("fact_type") == event.get("fact_type") and
            e.get("value")     == event.get("value")     and
            e.get("date")      == event.get("date")
        ):
            return True
    return False


# ── Pipeline entry points ─────────────────────────────────────────────────────

async def process_baseline_doc(
    vendor_id: str,
    file_bytes: bytes,
    filename: str,
) -> dict:
    """
    Process a document in BASELINE mode.

    Extracts the six baseline fields via Groq and writes to Hindsight
    using memory.write_baseline() → hindsight.store("{vendor_id}:baseline", data).

    Returns: {"ok": bool, "baseline": dict | None, "error": str | None}
    """
    from memory import write_baseline

    text     = extract_text(file_bytes, filename)
    baseline = extract_baseline(text, filename)

    if not baseline:
        return {
            "ok":       False,
            "baseline": None,
            "error":    "Could not extract baseline fields from document.",
        }

    for field in ["vendor_name", "agreed_price", "sla_terms",
                  "contract_start", "renewal_date", "rep_name"]:
        baseline.setdefault(field, "")

    stored = await write_baseline(vendor_id, baseline)
    return {"ok": True, "baseline": stored, "error": None}


async def process_normal_doc(
    vendor_id: str,
    file_bytes: bytes,
    filename: str,
    existing_baseline: Optional[dict] = None,
    existing_events:   Optional[list] = None,
) -> dict:
    """
    Process a document in NORMAL (event) mode.

    Extracts facts via Groq, runs drift classification, deduplicates,
    and appends each new fact using memory.append_event()
    → hindsight.append("{vendor_id}:events", event).

    Returns:
        {
          "ok": bool,
          "doc_date": str,
          "events_added": list,
          "events_skipped": list,
          "error": str | None,
        }
    """
    from memory import append_event

    if existing_events is None:
        existing_events = []

    text       = extract_text(file_bytes, filename)
    extraction = extract_events(text, filename)

    if not extraction or not extraction.get("facts"):
        return {
            "ok":             False,
            "doc_date":       None,
            "events_added":   [],
            "events_skipped": [],
            "error":          "No facts extracted from document.",
        }

    doc_date       = extraction.get("doc_date") or ""
    facts          = extraction.get("facts", [])
    events_added   = []
    events_skipped = []

    for fact in facts:
        # Discard low-confidence facts (§3.5)
        if float(fact.get("confidence", 0)) < 0.7:
            events_skipped.append({"fact": fact, "reason": "confidence < 0.7"})
            continue

        # Normalise fact_type to valid values
        ft = fact.get("fact_type", "incident")
        ft_map = {
            "price":       "price_change",
            "sla":         "sla_violation",
            "term_change": "term_change",
            "commitment":  "commitment",
            "incident":    "incident",
            "baseline":    None,
        }
        mapped = ft_map.get(ft, ft)
        if mapped is None:
            events_skipped.append({"fact": fact, "reason": "baseline fact in normal mode"})
            continue
        fact["fact_type"] = mapped

        # Drift classification against baseline
        severity     = 1
        drift_result = None
        if existing_baseline:
            drift_result = classify_drift(existing_baseline, fact)
            if drift_result:
                severity = int(drift_result.get("severity", 1))

        event = {
            "date":       doc_date,
            "fact_type":  fact["fact_type"],
            "value":      str(fact.get("value", "")),
            "summary":    str(fact.get("summary", "")),
            "severity":   severity,
            "source_doc": filename,
        }

        # Duplicate check
        if is_duplicate(event, existing_events):
            events_skipped.append({"event": event, "reason": "duplicate"})
            continue

        stored = await append_event(vendor_id, event)
        existing_events.append(stored)
        events_added.append({"event": stored, "drift": drift_result})

    return {
        "ok":             True,
        "doc_date":       doc_date,
        "events_added":   events_added,
        "events_skipped": events_skipped,
        "error":          None,
    }