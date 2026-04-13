"""
memory.py — VendorPulse Hindsight Memory Layer
Shared between Person 1 (pipeline.py) and Person 2 (agent.py).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OFFICIAL HINDSIGHT API  (from docs.hindsight.vectorize.io)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WRITE — aretain()
  client.aretain(bank_id, content, context, document_id, tags, metadata, timestamp)

READ — arecall()
  client.arecall(
      bank_id,               # str   — required
      query,                 # str   — natural-language search query
      budget     = "mid",    # str   — search depth: "low" | "mid" | "high"
      max_tokens = 4096,     # int   — token cap on returned content
      types      = [...],    # list  — optional memory type filter
      trace      = False,    # bool  — debug trace
      min_score  = None,     # float — relevance threshold 0–1
      limit      = None,     # int   — max results to return
  )

  RecallResult fields (response.results[]):
      .text          str  — the stored memory text
      .type          str  — "world_fact" | "observation" | "experience"
      .context       str  — verbatim value set in aretain(context=...)
      .entities      list — entity names mentioned
      .mentioned_at  str  — ISO-8601 timestamp
      .id            str  — unique memory id

  IMPORTANT: The documented recall response does NOT include a metadata field.
  All data routing must use item.context (exact match) and item.text (JSON parse).
  Do NOT rely on item.metadata — it is not guaranteed to be returned.

REASON — areflect()  [not used here — agent.py / Groq is our reasoning layer]
  client.areflect(bank_id, query, context, budget, max_tokens, response_schema)
  Returns a synthesised prose answer — unsuitable for structured data extraction.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RETRIEVAL STRATEGY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

We use arecall() (not areflect()) for reads because:
  • We need raw structured data for the LLM context block, not prose summaries.
  • agent.py (Groq) is our reasoning layer; Hindsight is our memory store only.

Context-based routing:
  aretain(context="aws:baseline") → arecall filters item.context == "aws:baseline"
  aretain(context="aws:events")   → arecall filters item.context == "aws:events"
  This is an exact string match on a stored verbatim field — fully reliable.

Budget per operation (per docs, default is "mid"):
  Registry reads  → "low"   (single tiny doc, cheapest)
  Baseline reads  → "mid"   (single structured doc, default depth)
  Event reads     → "high"  (many separate memories, need maximum coverage)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KEY LAYOUT IN HINDSIGHT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Registry  context="vendorpulse:registry"  document_id="vendorpulse:registry"
  Baseline  context="{vendor_id}:baseline"  document_id="{vendor_id}:baseline"
  Events    context="{vendor_id}:events"    document_id="{vendor_id}:events_{date}_{fact_type}_{slug}"

  Stable document_id on registry and baseline causes aretain() to overwrite
  the previous entry instead of accumulating duplicates.
"""

import asyncio
import json
import os
import re
from datetime import datetime, timezone
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

try:
    from hindsight_client import Hindsight
    HINDSIGHT_AVAILABLE = True
except ImportError:
    HINDSIGHT_AVAILABLE = False

_client: Optional["Hindsight"] = None

# FIX: was os.getenv("BANK_NAME") with no default → BANK_ID = None when env var
# is missing, which silently passes None to every API call. Default added.
BANK_ID = os.getenv("BANK_NAME", "VIM")

BASELINE_FIELDS = [
    "vendor_name", "agreed_price", "sla_terms",
    "contract_start", "renewal_date", "rep_name",
]
VALID_FACT_TYPES = {
    "price_change", "sla_violation", "term_change",
    "commitment", "incident",
}
REGISTRY_CONTEXT  = "vendorpulse:registry"
REGISTRY_DOCUMENT = "vendorpulse:registry"


# ── Client ─────────────────────────────────────────────────────────────────────

def _check_available():
    if not HINDSIGHT_AVAILABLE:
        raise RuntimeError(
            "hindsight_client not installed. Run: pip install hindsight-client"
        )


def get_client() -> "Hindsight":
    global _client
    _check_available()
    if _client is None:
        _client = Hindsight(
            base_url=os.getenv("HINDSIGHT_URL", "https://api.hindsight.vectorize.io"),
            api_key=os.getenv("HINDSIGHT_API_KEY", ""),
        )
    return _client


# ── Context helpers ────────────────────────────────────────────────────────────

def baseline_context(vendor_id: str) -> str:
    return f"{vendor_id}:baseline"


def events_context(vendor_id: str) -> str:
    return f"{vendor_id}:events"


# ── Normalisers ────────────────────────────────────────────────────────────────

def _normalise_baseline(data: dict) -> dict:
    return {field: str(data.get(field, "")) for field in BASELINE_FIELDS}


def _normalise_event(event: dict) -> dict:
    fact_type = event.get("fact_type", "incident")
    if fact_type not in VALID_FACT_TYPES:
        fact_type = "incident"
    return {
        "date":       str(event.get("date", "")),
        "fact_type":  fact_type,
        "value":      str(event.get("value", "")),
        "summary":    str(event.get("summary", "")),
        "severity":   int(event.get("severity", 1)),
        "source_doc": str(event.get("source_doc", "")),
    }


def _meta(d: dict) -> dict:
    """Hindsight metadata values must all be strings."""
    return {k: str(v) for k, v in d.items()}


def _parse_timestamp(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    try:
        date_str = date_str.rstrip("Z")
        dt = (datetime.fromisoformat(date_str) if "T" in date_str
              else datetime.fromisoformat(f"{date_str}T00:00:00"))
        return dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _extract_json(text: str) -> Optional[dict]:
    """Extract the first complete JSON object from a string."""
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _extract_json_list(text: str) -> Optional[list]:
    """Extract the first JSON array from a string."""
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            pass
    return None


# ── Async helper for sync wrappers ─────────────────────────────────────────────

def _run_async(coro):
    """
    Run a coroutine from a synchronous context (used by sync wrappers).

    asyncio.run() raises RuntimeError when called inside a running event loop
    (e.g. uvicorn's loop, even from a ThreadPoolExecutor thread). We use
    get_event_loop().run_until_complete() on the thread's own loop, or create
    a fresh one if the thread has none.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("loop closed")
        return loop.run_until_complete(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


# ── LOW-LEVEL RECALL WITH CONTEXT FILTER ──────────────────────────────────────

async def _recall_by_context(
    context_value: str,
    query: str,
    budget: str = "mid",
    max_tokens: int = 8192,
) -> list:
    """
    Recall memories from Hindsight and filter by exact context match.

    Uses the documented arecall() API:
        arecall(bank_id, query, budget, max_tokens)

    `budget` is a valid documented parameter ("low" | "mid" | "high").
    Callers choose the depth appropriate for the data volume:
        "low"  → registry (tiny single doc)
        "mid"  → baseline (single structured doc)
        "high" → events   (many memories, need full coverage)

    Context filter: item.context == context_value is an exact match on the
    verbatim string stored by aretain(context=...). This is reliable routing,
    not semantic similarity.

    IMPORTANT: The documented recall response does NOT include metadata.
    Callers must parse all data from item.text only.
    """
    client = get_client()
    try:
        response = await client.arecall(
            bank_id=BANK_ID,
            query=query,
            budget=budget,      # valid documented param — do NOT omit
            max_tokens=max_tokens,
        )
    except Exception as e:
        print(f"[memory] arecall error (context={context_value}): {e}")
        return []

    if not response or not response.results:
        return []

    matched = [
        item for item in response.results
        if getattr(item, "context", None) == context_value
    ]
    print(
        f"[memory] _recall_by_context '{context_value}': "
        f"{len(matched)}/{len(response.results)} matched"
    )
    return matched


# ── VENDOR REGISTRY ────────────────────────────────────────────────────────────

async def get_all_vendors() -> list:
    """
    Return the list of all registered vendor_id strings.

    Stored as a single memory with a stable document_id so aretain() overwrites
    it on every update. The vendor list is parsed from item.text as JSON.
    
    Fallback: If registry JSON parsing fails, detect vendors from baseline contexts.
    """
    items = await _recall_by_context(
        context_value=REGISTRY_CONTEXT,
        query="vendorpulse vendor registry list all vendors",
        budget="low",
        max_tokens=1024,
    )
    for item in items:
        text    = getattr(item, "text", "") or ""
        vendors = _extract_json_list(text)
        if vendors is not None:
            print(f"[memory] get_all_vendors: found {len(vendors)} from registry JSON")
            return [str(v) for v in vendors]
    
    # Fallback: Detect vendors dynamically from baseline contexts
    print("[memory] get_all_vendors: registry JSON empty, detecting from baselines...")
    vendors = await _detect_vendors_from_baselines()
    return vendors


async def _detect_vendors_from_baselines() -> list:
    """
    Fallback: Detect all vendors by searching for baseline contexts.
    Returns a list of unique vendor_id strings.
    """
    try:
        client = get_client()
        response = await client.arecall(
            bank_id=BANK_ID,
            query="vendor baseline contract",
            budget="high",
            max_tokens=32768,
        )
        
        if not response or not response.results:
            return []
        
        # Extract unique vendor IDs from baseline contexts
        vendors = set()
        for item in response.results:
            ctx = getattr(item, "context", None)
            if ctx and ctx.endswith(":baseline"):
                vendor = ctx[:-len(":baseline")]
                if vendor and vendor != "vendorpulse":
                    vendors.add(vendor)
        
        vendor_list = sorted(list(vendors))
        print(f"[memory] detected vendors from baselines: {vendor_list}")
        
        # Update registry with detected vendors (optional)
        if vendor_list:
            await _save_registry(vendor_list)
        
        return vendor_list
    except Exception as e:
        print(f"[memory] _detect_vendors_from_baselines error: {e}")
        return []


async def _save_registry(vendors: list) -> None:
    """Overwrite the registry memory with an updated vendor list."""
    client  = get_client()
    content = (
        f"VENDORPULSE VENDOR REGISTRY\n"
        f"All registered vendors: {', '.join(vendors)}\n\n"
        f"JSON: {json.dumps(vendors)}"
    )
    await client.aretain(
        bank_id=BANK_ID,
        content=content,
        context=REGISTRY_CONTEXT,
        document_id=REGISTRY_DOCUMENT,   # stable → overwrites, never duplicates
        tags=["vendorpulse:registry", "registry"],
        metadata=_meta({"record_type": "registry", "vendors": json.dumps(vendors)}),
        timestamp=None,
    )
    print(f"[memory] registry saved: {vendors}")


async def register_vendor(vendor_id: str) -> None:
    """
    Add vendor_id to the central registry if not already present.
    Called automatically by write_baseline().
    """
    existing = await get_all_vendors()
    if vendor_id not in existing:
        updated = existing + [vendor_id]
        await _save_registry(updated)
        print(f"[memory] register_vendor: added '{vendor_id}', total={len(updated)}")
    else:
        print(f"[memory] register_vendor: '{vendor_id}' already registered")


# ── WRITE BASELINE (Person 1 — pipeline.py) ────────────────────────────────────

async def write_baseline(vendor_id: str, baseline: dict) -> dict:
    """
    Store vendor baseline in Hindsight.

        context     = "{vendor_id}:baseline"
        document_id = "{vendor_id}:baseline"  ← stable, overwrites on re-ingest

    The normalised dict is serialised as both human-readable key: value lines
    AND as a "JSON: {...}" block at the end of the content string.
    get_baseline() parses the JSON block from item.text — this is the primary
    retrieval path since recall() does not return item.metadata per the docs.
    """
    client     = get_client()
    ctx        = baseline_context(vendor_id)
    normalised = _normalise_baseline(baseline)

    content = (
        f"VENDOR BASELINE — {vendor_id.upper()}\n"
        + "\n".join(f"{k}: {v}" for k, v in normalised.items())
        + f"\n\nJSON: {json.dumps(normalised)}"
    )

    await client.aretain(
        bank_id=BANK_ID,
        content=content,
        context=ctx,
        document_id=ctx,          # deterministic — overwrites previous baseline
        tags=[ctx, vendor_id, "baseline"],
        metadata=_meta({"record_type": "baseline", "vendor_id": vendor_id, **normalised}),
        timestamp=None,
    )
    print(f"[memory] write_baseline: stored context={ctx}")

    await register_vendor(vendor_id)
    return normalised


# ── APPEND EVENT (Person 1 — pipeline.py) ──────────────────────────────────────

async def append_event(vendor_id: str, event: dict) -> dict:
    """
    Store one vendor event in Hindsight.

        context     = "{vendor_id}:events"      (shared by all events for this vendor)
        document_id = unique slug per event     (Hindsight-side deduplication)

    Each call stores one event memory. get_events() collects all memories
    that share context="{vendor_id}:events" via a single arecall() call.

    The event dict is embedded as "JSON: {...}" in the content text so
    _parse_event_item() can reconstruct it from item.text without metadata.
    """
    client     = get_client()
    ctx        = events_context(vendor_id)
    normalised = _normalise_event(event)
    sev_label  = {1: "MINOR", 2: "NOTABLE", 3: "BREACH"}.get(normalised["severity"], "MINOR")

    value_slug  = re.sub(r"[^a-z0-9]", "_", normalised["value"].lower())[:30]
    document_id = f"{ctx}_{normalised['date']}_{normalised['fact_type']}_{value_slug}"

    content = (
        f"VENDOR EVENT — {vendor_id.upper()}\n"
        f"date: {normalised['date']}\n"
        f"fact_type: {normalised['fact_type']}\n"
        f"severity: {normalised['severity']} ({sev_label})\n"
        f"summary: {normalised['summary']}\n"
        f"value: {normalised['value']}\n"
        f"source_doc: {normalised['source_doc']}\n"
        f"\nJSON: {json.dumps(normalised)}"
    )

    await client.aretain(
        bank_id=BANK_ID,
        content=content,
        context=ctx,
        document_id=document_id,
        tags=[ctx, vendor_id, "event", normalised["fact_type"]],
        metadata=_meta({"record_type": "event", "vendor_id": vendor_id, **normalised}),
        timestamp=_parse_timestamp(normalised["date"]),
    )
    print(f"[memory] append_event: {normalised['fact_type']} @ {normalised['date']} ctx={ctx}")
    return normalised


# ── BASELINE FIELD EXTRACTION FROM TEXT FRAGMENTS ───────────────────────────

def _extract_baseline_fields_from_text(text: str, baseline_data: dict) -> None:
    """
    Extract baseline fields from text fragments.
    Updates baseline_data dict in-place with extracted values.
    """
    import re
    
    # Extract vendor_name (e.g., "Twilio Inc.", "AWS", "Salesforce")
    vendor_match = re.search(r"([\w\s&]+(?:Inc|Corp|LLC|Ltd|Company|Services)\.?)", text, re.IGNORECASE)
    if vendor_match and "vendor_name" not in baseline_data:
        baseline_data["vendor_name"] = vendor_match.group(1).strip()
    
    # Extract agreed_price (e.g., "$0.0075 per SMS", "€100/month", "$1,000 per unit")
    price_match = re.search(r"[\$€][\d,]+\.?\d*(?:\s*(?:per|/)\s*\w+)?", text)
    if price_match and "agreed_price" not in baseline_data:
        baseline_data["agreed_price"] = price_match.group(0).strip()
    
    # Extract SLA terms (e.g., "99.95% uptime", "99.9% availability")
    sla_match = re.search(r"(\d+\.?\d*%\s*(?:uptime|availability|guaranteed|monthly))", text, re.IGNORECASE)
    if sla_match and "sla_terms" not in baseline_data:
        baseline_data["sla_terms"] = sla_match.group(0).strip()
    
    # Extract contract start date (e.g., "March 1, 2022", "2022-03-01")
    if "contract_start" not in baseline_data:
        # Look for "started" or "contract" context
        start_match = re.search(r"(?:contract.*?)?(?:started on|start date|begins?\s+\w+\s+)?(\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}|\d{4}-\d{2}-\d{2})", text, re.IGNORECASE)
        if start_match:
            baseline_data["contract_start"] = start_match.group(1).strip()
    
    # Extract renewal date (e.g., "June 1, 2024", "2024-06-01")
    # Look for "renewal" or "renews" context specifically
    if "renewal_date" not in baseline_data:
        renewal_context = re.search(r"renewal.*?\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}|\d{4}-\d{2}-\d{2})", text, re.IGNORECASE | re.DOTALL)
        if renewal_context:
            baseline_data["renewal_date"] = renewal_context.group(1).strip()
    
    # Extract rep name (e.g., "Jordan Mills", "John Smith")
    # Look very carefully for actual names after "representative" or "rep"
    if "rep_name" not in baseline_data:
        # Pattern: look for "representative is <Name>" or "rep: <Name>" or "sales representative: <Name>"
        rep_patterns = [
            r"representative\s+(?:for|of)\s+[\w\s]+\s+is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",  # "representative for X is NAME"
            r"(?:representative|rep).*?name\s+is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",  # "representative name is NAME"
            r"Involving:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*(?:sales representative|\(.*?sales)", # "Involving: NAME (sales representative"
            r"(?:representative|rep)(?:\s+is|\s+:\s*|$|\s+)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",  # fallback patterns
        ]
        for pattern in rep_patterns:
            rep_match = re.search(pattern, text, re.IGNORECASE)
            if rep_match:
                name = rep_match.group(1).strip()
                # Filter out invalid matches
                if len(name.split()) >= 2 and name not in ["For", "The", "And", "Or", "Name Is"]:
                    baseline_data["rep_name"] = name
                    break


# ── GET BASELINE (Person 2 — agent.py) ────────────────────────────────────────

async def get_baseline(vendor_id: str) -> Optional[dict]:
    """
    Retrieve vendor baseline from Hindsight.

    Uses arecall() with budget="mid" (single doc, standard depth sufficient).
    Filters results by item.context == "{vendor_id}:baseline" (exact match).
    Parses the baseline dict from either JSON block or text fragments.

    No item.metadata access — not in the documented recall response.
    """
    ctx   = baseline_context(vendor_id)
    items = await _recall_by_context(
        context_value=ctx,
        query=f"vendor baseline contract agreed price SLA renewal rep name for {vendor_id}",
        budget="mid",
        max_tokens=4096,
    )

    # Aggregate data from all matching items (may contain different fields)
    baseline_data = {}
    
    for item in items:
        text = getattr(item, "text", "") or ""
        
        # Try JSON first
        parsed = _extract_json(text)
        if parsed and any(f in parsed for f in ("vendor_name", "agreed_price")):
            print(f"[memory] get_baseline: found JSON for '{vendor_id}'")
            return _normalise_baseline(parsed)
        
        # Parse text fragments for structured data
        _extract_baseline_fields_from_text(text, baseline_data)
    
    # If we extracted any fields, return them
    if baseline_data:
        print(f"[memory] get_baseline: found text fragments for '{vendor_id}'")
        return _normalise_baseline(baseline_data)

    print(f"[memory] get_baseline: not found for '{vendor_id}'")
    return None


# ── GET EVENTS (Person 2 — agent.py) ──────────────────────────────────────────

async def get_events(vendor_id: str) -> list:
    """
    Retrieve all events for a vendor from Hindsight.

    Uses arecall() with budget="high" and large max_tokens because:
      • Events accumulate indefinitely over the vendor relationship lifetime.
      • A lower budget risks Hindsight's semantic search not surfacing older
        or less-queried events.
      • We need full coverage, not top-k relevance.

    Filters by item.context == "{vendor_id}:events", parses each from item.text
    (both JSON blocks and text fragments), deduplicates by (fact_type, value, date),
    and sorts chronologically.

    No item.metadata access — not in the documented recall response.
    """
    ctx   = events_context(vendor_id)
    items = await _recall_by_context(
        context_value=ctx,
        query=(
            f"vendor events price changes SLA violations incidents "
            f"commitments term changes for {vendor_id}"
        ),
        budget="high",
        max_tokens=32768,
    )

    events = []
    seen   = set()

    for item in items:
        # Try JSON first
        ev = _parse_event_item(item, vendor_id)
        if ev:
            key = (ev["fact_type"], ev["value"], ev["date"])
            if key not in seen:
                seen.add(key)
                events.append(ev)
        else:
            # Try parsing as text fragment
            text = getattr(item, "text", "") or ""
            ev_from_text = _parse_event_from_text_fragment(text, vendor_id)
            if ev_from_text:
                key = (ev_from_text["fact_type"], ev_from_text["value"], ev_from_text["date"])
                if key not in seen:
                    seen.add(key)
                    events.append(ev_from_text)

    events.sort(key=lambda e: e.get("date", ""))
    print(f"[memory] get_events: {len(events)} events for '{vendor_id}'")
    return events


def _parse_event_item(item, vendor_id: str) -> Optional[dict]:
    """
    Parse a RecallResult into a normalised event dict.

    Extracts the JSON block embedded in item.text by append_event().
    No item.metadata access — not in the documented recall response.
    """
    text   = getattr(item, "text", "") or ""
    parsed = _extract_json(text)
    if parsed and "fact_type" in parsed:
        normalised = _normalise_event(parsed)
        if normalised["fact_type"] in VALID_FACT_TYPES:
            return normalised
    return None


def _parse_event_from_text_fragment(text: str, vendor_id: str) -> Optional[dict]:
    """
    Parse an event from a text fragment.
    Returns a normalised event dict or None.
    """
    import re
    
    event = {
        "date": "",
        "fact_type": "incident",  # default
        "value": "",
        "summary": "",
        "severity": 1,
        "source_doc": "",
    }
    
    # Extract date (various formats)
    date_match = re.search(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}|\d{4}-\d{2}-\d{2}", text, re.IGNORECASE)
    if date_match:
        event["date"] = date_match.group(0).strip()
    
    # Determine fact_type from content keywords
    text_lower = text.lower()
    if "price" in text_lower or "cost" in text_lower or "fee" in text_lower or "$" in text:
        event["fact_type"] = "price_change"
        # Extract price value
        price_match = re.search(r"[\$€][\d,]+\.?\d*", text)
        if price_match:
            event["value"] = price_match.group(0).strip()
    elif "sla" in text_lower or "uptime" in text_lower or "availability" in text_lower or "violation" in text_lower or "downtime" in text_lower:
        event["fact_type"] = "sla_violation"
        # Extract percentage or duration
        value_match = re.search(r"(\d+\.?\d*%|\d+\s*(?:hours?|minutes?|days?))", text, re.IGNORECASE)
        if value_match:
            event["value"] = value_match.group(0).strip()
    elif "term" in text_lower or "condition" in text_lower or "change" in text_lower:
        event["fact_type"] = "term_change"
    elif "commit" in text_lower or "promise" in text_lower or "agree" in text_lower:
        event["fact_type"] = "commitment"
    elif "incident" in text_lower or "issue" in text_lower or "problem" in text_lower or "outage" in text_lower:
        event["fact_type"] = "incident"
    
    # Severity detection
    if any(keyword in text_lower for keyword in ["critical", "major", "breach", "failed", "down"]):
        event["severity"] = 3
    elif any(keyword in text_lower for keyword in ["notable", "warning", "violation"]):
        event["severity"] = 2
    else:
        event["severity"] = 1
    
    # Set summary to the full text and return
    event["summary"] = text.strip()[:200]  # Limit to 200 chars
    
    # Only return if we have at least date or some meaningful content
    if event["date"] or len(event["summary"]) > 10:
        return _normalise_event(event)
    
    return None


# ── SYNC WRAPPERS (agent.py calls these via ThreadPoolExecutor) ────────────────

def get_baseline_sync(vendor_id: str) -> Optional[dict]:
    try:
        return _run_async(get_baseline(vendor_id))
    except Exception as e:
        print(f"[memory] get_baseline_sync error: {e}")
        return None


def get_events_sync(vendor_id: str) -> list:
    try:
        return _run_async(get_events(vendor_id))
    except Exception as e:
        print(f"[memory] get_events_sync error: {e}")
        return []


def get_all_vendors_sync() -> list:
    try:
        return _run_async(get_all_vendors())
    except Exception as e:
        print(f"[memory] get_all_vendors_sync error: {e}")
        return []


# ── LOCAL FILE FALLBACK (offline dev / testing) ────────────────────────────────

def recall_from_file(
    vendor_id: str, filepath: str = "data/dummy_memories.json"
) -> dict:
    with open(filepath, "r") as f:
        data = json.load(f)
    return {
        "baseline": data.get(f"{vendor_id}:baseline"),
        "events":   data.get(f"{vendor_id}:events", []),
    }