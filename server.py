"""
server.py — VendorPulse Unified FastAPI Server

Endpoints:
  GET  /                  — Agent UI (chat, overview, drift alerts)
  GET  /pipeline          — Pipeline UI (document ingestion)
  GET  /health            — liveness check
  POST /upload            — ingest raw files through extraction pipeline
  GET  /api/vendors       — list all registered vendors
  GET  /api/overview      — baseline + events for one or ALL vendors
  GET  /api/alerts        — severity >= 2 events for one or ALL vendors
  POST /api/chat          — ask a question about a specific vendor
  POST /api/brief         — negotiation brief for a specific vendor
  POST /api/ingest        — write a pre-structured baseline or event directly

Environment variables (.env):
  GROQ_API_KEY       — Groq LLM key
  HINDSIGHT_API_KEY  — Hindsight memory key
  HINDSIGHT_URL      — Hindsight base URL (optional)
  BANK_NAME          — Hindsight bank ID (optional, default "vendorpulse")

No vendor is hardcoded anywhere. All endpoints require vendor_id explicitly,
or fan out across all registered vendors when vendor_id is omitted.
"""

import os
import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

import memory
import pipeline
from agent import ask_vendorpulse, get_negotiation_brief, build_context_block, USE_HINDSIGHT

# UI file paths
_BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
_AGENT_UI    = os.path.join(_BASE_DIR, "static/agent_ui.html")
_PIPELINE_UI = os.path.join(_BASE_DIR, "static/pipeline_ui.html")

# Thread pool — agent functions call get_baseline_sync / get_events_sync
# which use asyncio.run() internally. They must run on plain OS threads
# (no existing event loop attached).
_executor = ThreadPoolExecutor()

app = FastAPI(title="VendorPulse", version="2.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request schemas ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    vendor_id: Optional[str] = None  # optional — if None or "all", uses all vendors
    question:  str


class IngestRequest(BaseModel):
    doc_type:  str          # "baseline" or "event"
    vendor_id: str          # required — no default
    data:      dict = {}


# ── Frontend routes ───────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def agent_ui():
    if os.path.exists(_AGENT_UI):
        return FileResponse(_AGENT_UI)
    return JSONResponse({"message": "VendorPulse running. Place agent_ui.html here."})


@app.get("/pipeline", include_in_schema=False)
async def pipeline_ui_route():
    if os.path.exists(_PIPELINE_UI):
        return FileResponse(_PIPELINE_UI)
    return JSONResponse({"message": "Pipeline UI not found. Place pipeline_ui.html here."})


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    vendors = await memory.get_all_vendors()
    return {
        "status":    "ok",
        "bank":      memory.BANK_ID,
        "hindsight": memory.HINDSIGHT_AVAILABLE,
        "vendors":   vendors,
    }


# ── Verification & Debugging (for troubleshooting Hindsight data) ────────────

@app.get("/api/verify")
async def verify_hindsight():
    """
    Verify Hindsight connection and show what data is available.
    Use when: "Agent not showing data"
    
    Returns: {
        "hindsight_enabled": bool,
        "connection_ok": bool,
        "vendors_found": [list],
        "errors": [list],
        "vendor_status": {vendor_id: {baseline: bool, events_count: int}}
    }
    """
    result = {
        "hindsight_enabled": memory.HINDSIGHT_AVAILABLE,
        "connection_ok": False,
        "vendors_found": [],
        "vendor_status": {},
        "errors": []
    }
    
    try:
        vendors = await memory.get_all_vendors()
        result["vendors_found"] = vendors
        result["connection_ok"] = bool(vendors)
        
        # Check each vendor's data
        for vendor in vendors:
            baseline = await memory.get_baseline(vendor)
            events = await memory.get_events(vendor) 
            result["vendor_status"][vendor] = {
                "has_baseline": bool(baseline),
                "baseline_fields": list(baseline.keys()) if baseline else [],
                "events_count": len(events) if events else 0,
                "event_types": list(set(e.get("fact_type") for e in events)) if events else []
            }
        
        if not vendors:
            result["errors"].append("No vendors found in Hindsight")
    except Exception as e:
        result["errors"].append(str(e))
        result["connection_ok"] = False
    
    return result


@app.get("/api/debug/vendor/{vendor_id}")
async def debug_vendor(vendor_id: str = None):
    """
    Deep dive into what Hindsight has for a specific vendor.
    Use when: "I uploaded data for [vendor] but agent doesn't show it"
    
    Returns exactly what agent sees for this vendor:
    - Full baseline contract data
    - All events with complete details
    - How the agent formats this for LLM
    """
    vendor = vendor_id.strip().lower() if vendor_id else ""
    
    if not vendor:
        raise HTTPException(status_code=400, detail="vendor_id required")
    
    try:
        baseline = await memory.get_baseline(vendor)
        events = await memory.get_events(vendor)
        
        # Build the context block as the agent sees it
        from agent import build_context_block
        loop = asyncio.get_event_loop()
        context_block = await loop.run_in_executor(
            _executor, 
            build_context_block, 
            vendor
        )
        
        return {
            "vendor_id": vendor,
            "baseline": baseline,
            "baseline_status": "LOADED" if baseline else "MISSING",
            "events": events,
            "events_status": f"{len(events)} events" if events else "NO EVENTS",
            "summary": {
                "has_baseline": bool(baseline),
                "has_events": bool(events),
                "total_events": len(events),
                "event_types": list(set(e.get("fact_type") for e in events)) if events else []
            },
            "context_block_preview": context_block[:500] + "..." if len(context_block) > 500 else context_block
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Debug error for {vendor}: {str(e)}"
        )


# ── Vendor registry ───────────────────────────────────────────────────────────

@app.get("/api/vendors")
async def list_vendors():
    """
    Return all vendor IDs registered in Hindsight.

    A vendor is registered automatically the first time its baseline is
    written via POST /upload (mode=baseline) or POST /api/ingest (doc_type=baseline).

    Response: {"vendors": ["twilio", "aws", "salesforce"]}
    """
    vendors = await memory.get_all_vendors()
    return {"vendors": vendors}


# ── Upload — raw file ingestion pipeline ─────────────────────────────────────

@app.post("/upload")
async def upload(
    vendor_id: str          = Form(...),
    mode:      str          = Form(...),   # "baseline" or "normal"
    files: list[UploadFile] = File(...),
):
    """
    Ingest one or more raw documents for a vendor.

    Form fields:
      vendor_id — lowercase vendor shorthand e.g. "twilio", "aws"
      mode      — "baseline" writes frozen contract terms + registers vendor
                  "normal"   appends extracted facts to the event log
      files     — one or more files (.pdf .txt .docx .csv .eml)
    """
    vendor_id = vendor_id.strip().lower()
    if not vendor_id:
        raise HTTPException(status_code=422, detail="vendor_id must not be empty.")
    if mode not in ("baseline", "normal"):
        raise HTTPException(status_code=422, detail="mode must be 'baseline' or 'normal'.")

    results = []

    existing_baseline = None
    existing_events   = []
    if mode == "normal":
        existing_baseline = await memory.get_baseline(vendor_id)
        existing_events   = await memory.get_events(vendor_id)

    for file in files:
        file_bytes = await file.read()
        filename   = file.filename

        if mode == "baseline":
            result = await pipeline.process_baseline_doc(
                vendor_id=vendor_id,
                file_bytes=file_bytes,
                filename=filename,
            )
        else:
            result = await pipeline.process_normal_doc(
                vendor_id=vendor_id,
                file_bytes=file_bytes,
                filename=filename,
                existing_baseline=existing_baseline,
                existing_events=existing_events,
            )

        results.append({
            "filename":       filename,
            "ok":             result.get("ok", False),
            "error":          result.get("error"),
            "baseline":       result.get("baseline"),
            "doc_date":       result.get("doc_date"),
            "events_added":   result.get("events_added"),
            "events_skipped": result.get("events_skipped"),
        })

    return {
        "ok":        all(r["ok"] for r in results),
        "vendor_id": vendor_id,
        "mode":      mode,
        "results":   results,
    }


# ── Overview ──────────────────────────────────────────────────────────────────

@app.get("/api/overview")
async def overview(vendor_id: Optional[str] = Query(default=None)):
    """
    Return baseline + events from Hindsight.

    GET /api/overview?vendor_id=twilio  → data for that vendor only
    GET /api/overview                   → data for ALL registered vendors
    """
    if vendor_id:
        vendor   = vendor_id.strip().lower()
        baseline = await memory.get_baseline(vendor)
        events   = await memory.get_events(vendor)
        return {
            "vendors": [vendor],
            "data": {vendor: {"baseline": baseline, "events": events}},
        }

    vendors = await memory.get_all_vendors()
    if not vendors:
        return {"vendors": [], "data": {}}

    data = {}
    for v in vendors:
        baseline = await memory.get_baseline(v)
        events   = await memory.get_events(v)
        data[v]  = {"baseline": baseline, "events": events}

    return {"vendors": vendors, "data": data}


# ── Drift alerts ──────────────────────────────────────────────────────────────

@app.get("/api/alerts")
async def alerts(
    vendor_id:    Optional[str] = Query(default=None),
    min_severity: int           = Query(default=2),
):
    """
    Return severity >= min_severity events, annotated with baseline ref.

    GET /api/alerts?vendor_id=twilio  → alerts for that vendor only
    GET /api/alerts                   → alerts across ALL registered vendors
    GET /api/alerts?min_severity=3    → only breaches, all vendors
    """
    async def _alerts_for(v: str) -> list:
        baseline     = await memory.get_baseline(v)
        events       = await memory.get_events(v)
        alert_events = [e for e in events if e.get("severity", 1) >= min_severity]

        for ev in alert_events:
            ev["vendor_id"] = v   # tag so caller knows which vendor
            if baseline:
                agreed_price = baseline.get("agreed_price", "")
                sla_terms    = baseline.get("sla_terms", "")
                if ev["fact_type"] == "price_change" and agreed_price:
                    ev.setdefault("baseline_ref", f"Baseline price: {agreed_price}")
                elif ev["fact_type"] in ("sla_violation", "term_change") and sla_terms:
                    ev.setdefault("baseline_ref", f"Baseline SLA: {sla_terms}")

        return alert_events

    if vendor_id:
        result = await _alerts_for(vendor_id.strip().lower())
        result.sort(key=lambda e: e.get("date", ""), reverse=True)
        return result

    vendors    = await memory.get_all_vendors()
    all_alerts = []
    for v in vendors:
        all_alerts.extend(await _alerts_for(v))

    all_alerts.sort(key=lambda e: e.get("date", ""), reverse=True)
    return all_alerts


# ── Chat ──────────────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(body: ChatRequest):
    """
    Ask a free-form question about vendor(s).
    
    vendor_id field:
      - If omitted, None, or "all": queries across ALL vendors
      - Otherwise: queries that specific vendor
    
    Both vendor_id and question can be in the request body.
    """
    vendor_input = (body.vendor_id or "").strip().lower()
    question = body.question.strip() if body.question else ""

    # Determine if multi-vendor or single vendor
    if vendor_input in ("", "all"):
        # Multi-vendor mode: get all vendors and query across all
        vendors = await memory.get_all_vendors()
        vendor_key = "all"
    else:
        vendors = [vendor_input]
        vendor_key = vendor_input

    if not question:
        raise HTTPException(
            status_code=400,
            detail="question is required and cannot be empty."
        )

    if not vendors:
        raise HTTPException(
            status_code=400,
            detail="No vendors found in the system."
        )

    try:
        loop   = asyncio.get_event_loop()
        # Pass vendor_key (either specific vendor or comma-joined list for multi-vendor)
        vendor_context = vendor_key if vendor_key != "all" else ",".join(vendors)
        answer = await loop.run_in_executor(_executor, ask_vendorpulse, vendor_context, question)
        return {
            "vendor_id": vendor_key,
            "vendors": vendors if vendor_key == "all" else [vendor_key],
            "question": question,
            "answer": answer
        }
    except Exception as e:
        print(f"[chat] error for vendors={vendor_key}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate answer: {str(e)}"
        )


# ── Negotiation brief ─────────────────────────────────────────────────────────

@app.post("/api/brief")
async def brief(vendor_id: str = Query(...)):
    """
    Generate a pre-renewal negotiation brief for a specific vendor.
    POST /api/brief?vendor_id=twilio
    """
    vendor = vendor_id.strip().lower() if vendor_id else ""
    if not vendor:
        raise HTTPException(
            status_code=400,
            detail="vendor_id is required and cannot be empty."
        )
    try:
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(_executor, get_negotiation_brief, vendor)
        return {"vendor_id": vendor, "brief": result}
    except Exception as e:
        print(f"[brief] error for vendor={vendor}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate brief: {str(e)}"
        )


# ── Direct ingest ─────────────────────────────────────────────────────────────

@app.post("/api/ingest")
async def ingest(body: IngestRequest):
    """
    Write a pre-structured baseline or event directly to Hindsight.
    Baseline ingestion automatically registers the vendor.

    Body:
      doc_type  — "baseline" or "event"
      vendor_id — required, no default
      data      — the baseline dict or event dict
    """
    doc_type = body.doc_type.strip()
    vendor   = body.vendor_id.strip().lower()
    data     = body.data

    if doc_type not in ("baseline", "event"):
        raise HTTPException(status_code=400, detail="doc_type must be 'baseline' or 'event'.")
    if not vendor:
        raise HTTPException(status_code=400, detail="vendor_id is required.")
    if not data:
        raise HTTPException(status_code=400, detail="data is required.")

    if not USE_HINDSIGHT:
        from memory import _normalise_baseline, _normalise_event
        normalised = _normalise_baseline(data) if doc_type == "baseline" else _normalise_event(data)
        return {
            "status":     "preview",
            "message":    "USE_HINDSIGHT is False — schema validated, not stored.",
            "doc_type":   doc_type,
            "vendor_id":  vendor,
            "normalised": normalised,
        }

    if doc_type == "baseline":
        # write_baseline() calls register_vendor() internally
        stored = await memory.write_baseline(vendor, data)
        return {"status": "stored", "doc_type": "baseline", "vendor_id": vendor, "stored": stored}
    else:
        stored = await memory.append_event(vendor, data)
        return {"status": "stored", "doc_type": "event", "vendor_id": vendor, "stored": stored}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("\n  VendorPulse running at → http://localhost:8000\n")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)