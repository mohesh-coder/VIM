"""
agent.py — VendorPulse Agent Reasoning Layer (Person 2)

Reads memory from Hindsight via the correct key-value path:
  get_baseline(vendor_id) → hindsight.get("{vendor_id}:baseline")
  get_events(vendor_id)   → hindsight.get("{vendor_id}:events")

Functions exposed to server.py:
  build_context_block(vendor_id)  → formatted memory string for LLM
  ask_vendorpulse(vendor_id, q)   → free-form question answer
  get_negotiation_brief(vendor_id)→ structured pre-renewal brief
  get_drift_alerts(vendor_id)     → severity >= 2 events
  get_overview(vendor_id)         → raw baseline + events dict
"""

import os
import re
from groq import Groq
from collections import defaultdict
from dotenv import load_dotenv
from memory import get_baseline_sync, get_events_sync, recall_from_file

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "qwen/qwen3-32b"

# Set to False to use local data/dummy_memories.json instead of Hindsight
USE_HINDSIGHT = True


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Strip chain-of-thought tags and normalise whitespace."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("\\$", "$")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── STEP 1 — Load memory ──────────────────────────────────────────────────────

def _load_memory(vendor_id: str) -> dict:
    """
    Returns {"baseline": {...}, "events": [...]}

    Routes to Hindsight (key-value get) or local file based on USE_HINDSIGHT.
    
    IMPORTANT: This function will print debug info to console showing:
    - Whether Hindsight is being used or local files
    - Whether data was found
    - What fields were retrieved
    """
    if USE_HINDSIGHT:
        # Retrieve from Hindsight
        baseline = get_baseline_sync(vendor_id)
        events = get_events_sync(vendor_id)
        
        # Debug logging
        print(f"\n{'='*70}")
        print(f"[AGENT] _load_memory() for '{vendor_id}'")
        print(f"[AGENT] SOURCE: Hindsight memory (USE_HINDSIGHT={USE_HINDSIGHT})")
        print(f"[AGENT] Baseline found: {bool(baseline)}")
        if baseline:
            print(f"[AGENT]   Fields: {list(baseline.keys())}")
            for k, v in baseline.items():
                print(f"[AGENT]     {k}: {v}")
        print(f"[AGENT] Events found: {len(events) if events else 0}")
        if events:
            for i, ev in enumerate(events[:3]):  # Show first 3
                print(f"[AGENT]   Event {i+1}: {ev.get('fact_type')} - {ev.get('summary')}")
            if len(events) > 3:
                print(f"[AGENT]   ... and {len(events)-3} more events")
        print(f"{'='*70}\n")
        
        return {
            "baseline": baseline,
            "events": events or [],
        }
    
    # Fallback to local file
    print(f"\n[AGENT] Using local file fallback (USE_HINDSIGHT=False)")
    return recall_from_file(vendor_id)


# ── STEP 2 — Build context block ──────────────────────────────────────────────

def _has_memory(vendor_id: str) -> dict:
    """
    Check if memory exists for this vendor.
    Returns: {"has_baseline": bool, "has_events": bool, "baseline": {...}, "events": [...]}
    """
    mem = _load_memory(vendor_id)
    baseline = mem.get("baseline")
    events = mem.get("events", [])
    return {
        "has_baseline": bool(baseline),
        "has_events": bool(events),
        "baseline": baseline,
        "events": events
    }


def build_context_block(vendor_id: str) -> str:
    """
    Formats the vendor memory into a structured text block for the LLM.
    Groups events by fact_type and shows them chronologically.
    
    If memory is unavailable, returns a template indicating the data absence,
    allowing the agent to provide general guidance instead.
    
    If memory is partial, clearly notes what's available and what's not.
    """
    mem      = _load_memory(vendor_id)
    baseline = mem.get("baseline")
    events   = mem.get("events", [])

    vendor_name = vendor_id.upper()
    
    if not baseline and not events:
        # No data available — inform LLM to provide general guidance
        return f"""VENDOR: {vendor_name}

🔍 MEMORY STATUS: UNAVAILABLE

⚠️ No stored memory found for "{vendor_id}". This vendor has NOT been onboarded yet.

Data Known: None
- No baseline contract data
- No event history or incidents
- No pricing records
- No SLA commitments stored

INSTRUCTION FOR AGENT:
You must provide GENERAL PROCUREMENT GUIDANCE based on:
1. Industry benchmarks and best practices for {vendor_name}
2. Common contract clauses and negotiation strategies
3. Typical pricing models in their industry
4. Standard SLA expectations
5. Red flags to watch for with this vendor type

Be transparent: "Based on industry standards for {vendor_name}..." 
Recommend: "Best practice would be to..."
Suggest: "Typical {vendor_name} contracts include..."

Help the user understand what SHOULD be in their contract, even though we don't have
their specific historical data yet."""

    # Partial or full data available
    lines = [
        f"VENDOR: {vendor_name}",
        "",
        "✅ MEMORY STATUS: DATA AVAILABLE (Retrieved from Hindsight)",
    ]
    
    if baseline:
        lines.extend([
            "",
            "┌─────────────────────────────────────────────────────────────┐",
            "│ 📋 BASELINE CONTRACT — Original Agreed Terms                │",
            "└─────────────────────────────────────────────────────────────┘",
        ])
        
        fields = {
            "vendor_name": "Vendor Company:",
            "agreed_price": "Agreed Price:",
            "sla_terms": "SLA Commitment:",
            "contract_start": "Contract Started:",
            "renewal_date": "Renewal Date:",
            "rep_name": "Account Rep:",
        }
        
        for field_key, label in fields.items():
            value = baseline.get(field_key, "")
            if value:
                lines.append(f"  {label:<25} {value}")
            else:
                lines.append(f"  {label:<25} [NOT RECORDED]")
    else:
        lines.extend([
            "",
            "✗ No baseline contract data available",
        ])
    
    if events:
        lines.extend([
            "",
            "┌─────────────────────────────────────────────────────────────┐",
            f"│ 📊 EVENT HISTORY — {len(events)} Changes/Incidents Logged      │",
            "└─────────────────────────────────────────────────────────────┘",
        ])
        
        # Summary by type
        by_type = defaultdict(list)
        for ev in events:
            by_type[ev["fact_type"]].append(ev)
        
        lines.append("")
        lines.append("Summary by type:")
        for fact_type in ["price_change", "sla_violation", "term_change", "commitment", "incident"]:
            if fact_type in by_type:
                count = len(by_type[fact_type])
                lines.append(f"  • {fact_type.replace('_', ' ').title():<20} {count:>3} events")
    else:
        lines.extend([
            "",
            "✗ No event history available (baseline contract only)",
        ])

    if not events:
        return "\n".join(lines)

    # DETAILED EVENT LOG
    lines.extend([
        "",
        "┌─────────────────────────────────────────────────────────────┐",
        "│ 📅 DETAILED CHRONOLOGICAL EVENT LOG                         │",
        "└─────────────────────────────────────────────────────────────┘",
        "",
    ])
    
    sev_icons = {1: "⚪", 2: "🟡", 3: "🔴"}
    sev_label = {1: "MINOR", 2: "NOTABLE", 3: "BREACH"}
    
    grouped = defaultdict(list)
    for ev in sorted(events, key=lambda x: x.get("date", "")):
        grouped[ev["fact_type"]].append(ev)

    for fact_type in ["price_change", "sla_violation", "term_change", "commitment", "incident"]:
        if fact_type not in grouped:
            continue
            
        lines.append("")
        lines.append(f"━━━ {fact_type.upper().replace('_', ' ')} ━━━")
        
        for i, ev in enumerate(grouped[fact_type], 1):
            sev = sev_icons.get(ev.get("severity", 1), "?")
            sev_name = sev_label.get(ev.get("severity", 1), "?")
            
            lines.extend([
                f"",
                f"  [{i}] {sev} {sev_name} | {ev['date']}",
                f"      Summary:  {ev['summary']}",
                f"      Value:    {ev['value']}",
                f"      Source:   {ev['source_doc']}",
            ])

    lines.extend([
        "",
        "┌─────────────────────────────────────────────────────────────┐",
        "│ 💡 ANALYSIS GUIDANCE                                        │",
        "└─────────────────────────────────────────────────────────────┘",
        "",
        "Use the detailed event log above to:",
        "  1. Identify price increases and calculate overpayment amounts",
        "  2. Find SLA violations and quantify business impact",
        "  3. Spot term changes that weaken your position",
        "  4. Locate broken commitments for negotiation leverage",
        "  5. Document incidents for credit/compensation claims",
        "",
        "This is YOUR vendor history. Use it to negotiate from strength.",
    ])
    
    return "\n".join(lines)


# ── STEP 3 — Free-form question ───────────────────────────────────────────────

SYSTEM_PROMPT = """You are VendorPulse, a procurement intelligence agent.
Your job is to help procurement teams understand their vendor relationships deeply.

═══════════════════════════════════════════════════════════════════════════════

TWO MODES OF OPERATION:

1️⃣  DATA-PRESENT MODE (Memory exists for this vendor)
   You have complete vendor history: contracts, pricing, incidents, promises.
   
   YOUR ROLE:
   - Extract specific facts: "Your price went up 8% on March 15, 2024"
   - Identify leverage: "No credit was issued for the June 2023 outage"
   - Show contradictions: "SLA was 99.95% in original — now 99.9% in renewal"
   - Uncover hidden risks: "Rep promised in writing to lock rates — not honored"
   - Quantify impact: "$50K overpayment from 2023-2024"
   - Recommend actions: "Use Q2 outage as basis to demand $15K credit"
   
   TONE: Specific, confident, fact-based. Use dates and numbers heavily.

2️⃣  DATA-ABSENT MODE (No memory yet for this vendor)
   This vendor hasn't been onboarded. You don't have their specific contracts yet.
   
   YOUR ROLE:
   - Explain what SHOULD be in their contract
   - Share industry benchmarks: "Typical SaaS SLAs are 99.95% uptime"
   - Suggest negotiation strategies: "For a $500K spend, demand 48-hour response time SLA"
   - Warn about common traps: "Watch for auto-renewal clauses without price caps"
   - Ask for proof: "Future uploads will validate this vendor's commitments"
   
   TONE: Educational, strategic, forward-looking. Prepare them for negotiations.

═══════════════════════════════════════════════════════════════════════════════

RESPONSE RULES:

Format:
- Clean markdown only. NO preamble. Answer directly.
- Use **bold** for: prices, dates, SLA %, vendor promises, risk amounts
- Use bullet points for lists
- NO LaTeX formulas — write math as plain text: "$100 - $92 = **$8 savings**"

Data Sources (be transparent):
- If data present: "Based on your stored memory: Your price was $0.0075 per SMS..."
- If data absent: "Industry standard for Twilio suggests..."

Structure for renewal calls (if asked):
1. Current state (if data present)
2. What changed (incidents, increases, broken promises)
3. Your leverage
4. Recommended asks
5. Red flags for next contract

Content rules:
- Be specific when you have data (dates, amounts, exact terms)
- Be general when you don't (benchmarks, best practices, templates)
- Always recommend: "Next step: upload your renewal draft here to analyze changes"
- End with one impactful takeaway

═══════════════════════════════════════════════════════════════════════════════

SPECIAL CASES:

Multi-vendor queries:
- When comparing vendors: note which have stored data vs. which need uploads
- Example: "AWS (has 12 events): uptime 99.95% | Salesforce (no data): typical 99.9%"
- Recommend: "Upload Salesforce contract to get comparison depth"

Incomplete data (baseline but no events):
- Use baseline to establish what was promised
- Note: "We don't have incident records yet — upload support tickets to add context"

Incomplete data (events but no baseline):
- Note the pattern without original terms: "Prices increased 3 times, SLA violated twice"
- Add: "Upload original contract so we can calculate the impact delta"

═══════════════════════════════════════════════════════════════════════════════

CORE MISSION: Close the information gap between your vendor's perfect records
and your incomplete files. Use what we have. Ask for what we need."""


def ask_vendorpulse(vendor_id: str, question: str) -> str:
    """
    Answer a question about vendor(s).
    
    vendor_id can be:
      - Single vendor: "aws"
      - Multiple vendors: "aws,salesforce,azure" (comma-separated)
      - All vendors: handled by server.py which passes comma-separated list
    """
    # Parse vendor IDs (support comma-separated or single)
    vendor_list = [v.strip().lower() for v in vendor_id.split(",") if v.strip()]
    
    if not vendor_list:
        return "No vendor specified."
    
    if len(vendor_list) == 1:
        # Single vendor — use normal context block
        context = build_context_block(vendor_list[0])
    else:
        # Multiple vendors — build composite context
        contexts = []
        for v in vendor_list:
            contexts.append(build_context_block(v))
        context = "\n\n" + "=" * 70 + "\n\n".join(contexts)
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Vendor memory:\n\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.3,
        max_tokens=1000,
    )
    return _clean(response.choices[0].message.content)


# ── STEP 4 — Negotiation brief ────────────────────────────────────────────────

BRIEF_PROMPT = """You are VendorPulse, a procurement intelligence agent.
Write a pre-renewal negotiation brief using the vendor memory below.

YOUR MEMORY CONTEXT IS BELOW. If it shows "UNAVAILABLE" or "No data", you are in 
DISCOVERY MODE — help them prepare for negotiations by explaining what to look for.
If it shows specific data, you are in LEVERAGE MODE — use concrete facts to build 
their negotiating position.

STRICT OUTPUT RULES:
- Clean markdown only. No preamble. Start with the brief.
- **Bold** every specific value: price, date, percentage, name.
- One fact per bullet. No LaTeX. No escaped dollar signs.
- If data unavailable: provide framework for what should be in their contract
- If data available: provide specific facts and ask amounts

═══════════════════════════════════════════════════════════════════════════════

IF YOU HAVE DATA (Memory shows baseline + events):

Produce EXACTLY these four sections:

## 1. Pricing history
- Original agreed price: **$X** from **contract start date**
- Every price change: **date** | old **$X** → new **$Y** | **Z% change** | authorized? 
- Total cumulative overpayment: **$AMOUNT**
- Baseline: was/wasn't a price lock in place?

## 2. SLA compliance
- Original SLA commitment: quote it: **"99.95% uptime"**
- Every violation: **date** | **what happened** | credit issued: **$X or none**
- Overall verdict in one sentence: "X violations unaddressed" or "Perfect record"
- Leverage point: "Demand $X credit for downtime"

## 3. Broken commitments
- Every unkept promise: **what** (by **rep name** on **date**) | current status
- Verbal-only promises: 🚩 **FLAG THESE** — "No written record"
- Written commitments: link to specific document/date

## 4. Suggested opening position
Three numbered, specific asks backed by evidence:
1. **$X credit for outages** (based on section 2)
2. **Price locked at $Y** (based on section 1 trend)
3. **Upgrade SLA to 99.99%** (based on section 3 non-delivery)

═══════════════════════════════════════════════════════════════════════════════

IF YOU DON'T HAVE DATA (Memory shows "UNAVAILABLE"):

Produce EXACTLY these four sections:

## 1. Pricing checklist - What to find
- What was the original agreed price? (Look in: initial contract, SOW, quote)
- How has it changed? (Invoice progression, rate change requests, amendments)
- Calculate: Today's spend vs original price = overpayment?
- **Red flag**: Auto-renewal without price caps = unlimited increases possible

## 2. SLA compliance checklist - What to document
- What's the actual SLA? (99.95%? 99.9%? Best-effort?)
- How often did they miss it? (Check: status pages, support tickets, incident reports, credits issued)
- **Red flag**: No SLA = no recourse for downtime

## 3. Commitment verification checklist
- What did they PROMISE? (In writing: contract clauses | Verbally: call transcripts, emails?)
- What did they DELIVER? (Compare: original promise vs actual service received)
- **Red flag**: Everything is verbal = nothing enforceable

## 4. Negotiation prep framework - Build your position
- [ ] Gather evidence: upload 2-3 documents per section above
- [ ] Benchmark their terms: typical **Vendor Type** contracts get **X**
- [ ] Calculate asks: If overpaid **$50K**, demand **$15K credit**
- [ ] Prepare walkaway point: "If no price cut, we move to Competitor"

---

✅ **Next step**: Upload your documents here. Each upload makes this brief more precise.

═══════════════════════════════════════════════════════════════════════════════

Vendor memory:
{context}"""


def get_negotiation_brief(vendor_id: str) -> str:
    mem = _load_memory(vendor_id)
    baseline = mem.get("baseline")
    events = mem.get("events", [])
    context = build_context_block(vendor_id)
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": BRIEF_PROMPT.format(context=context)},
        ],
        temperature=0.2,
        max_tokens=2000,
    )
    return _clean(response.choices[0].message.content)


# ── STEP 5 — Drift alerts ─────────────────────────────────────────────────────

def get_drift_alerts(vendor_id: str, min_severity: int = 2) -> list:
    """Return events with severity >= min_severity, newest first."""
    mem    = _load_memory(vendor_id)
    events = mem.get("events", [])
    alerts = [ev for ev in events if ev.get("severity", 0) >= min_severity]
    return sorted(alerts, key=lambda x: x.get("date", ""), reverse=True)


# ── STEP 5B — Memory status ────────────────────────────────────────────────────

def get_memory_status(vendor_id: str) -> dict:
    """
    Return detailed memory status for a vendor.
    Useful for UI to show what data is available without full context block.
    
    Returns:
    {
        "vendor_id": str,
        "has_baseline": bool,
        "has_events": bool,
        "baseline_fields": {"vendor_name": str, "agreed_price": str, ...},
        "event_count": int,
        "event_summary": {"price_change": int, "sla_violation": int, ...},
        "latest_event_date": str or None,
        "status": "complete" | "partial" | "unavailable"
    }
    """
    mem = _load_memory(vendor_id)
    baseline = mem.get("baseline")
    events = mem.get("events", [])
    
    # Count by type
    by_type = defaultdict(int)
    for ev in events:
        by_type[ev.get("fact_type", "incident")] += 1
    
    # Determine status
    if baseline and events:
        status = "complete"
    elif baseline or events:
        status = "partial"
    else:
        status = "unavailable"
    
    return {
        "vendor_id": vendor_id,
        "has_baseline": bool(baseline),
        "has_events": bool(events),
        "baseline_fields": baseline or {},
        "event_count": len(events),
        "event_summary": dict(by_type),
        "latest_event_date": max((ev.get("date") for ev in events), default=None),
        "status": status
    }


# ── STEP 6 — Raw overview ─────────────────────────────────────────────────────

def get_overview(vendor_id: str) -> dict:
    """Return raw baseline + events dict for the overview endpoint."""
    return _load_memory(vendor_id)


# ── VERIFICATION & DEBUGGING ──────────────────────────────────────────────────

def verify_hindsight_connection() -> dict:
    """
    Verify that Hindsight connection works and return status.
    Use this to debug: "Agent not showing Hindsight data"
    
    Returns:
    {
        "hindsight_enabled": bool,
        "connection_ok": bool,
        "vendors_found": list,
        "errors": list
    }
    """
    from memory import get_all_vendors_sync
    
    print("\n" + "="*70)
    print("[VERIFICATION] Testing Hindsight Connection")
    print("="*70)
    
    result = {
        "hindsight_enabled": USE_HINDSIGHT,
        "connection_ok": False,
        "vendors_found": [],
        "errors": []
    }
    
    if not USE_HINDSIGHT:
        result["errors"].append("USE_HINDSIGHT=False (using local file fallback)")
        print("❌ USE_HINDSIGHT is FALSE — using local files, not Hindsight")
        print("   To use Hindsight, set: USE_HINDSIGHT = True in agent.py")
        return result
    
    print("✓ USE_HINDSIGHT = True")
    
    try:
        vendors = get_all_vendors_sync()
        result["vendors_found"] = vendors
        
        if vendors:
            print(f"✓ Connected to Hindsight — found {len(vendors)} vendor(s)")
            result["connection_ok"] = True
            
            for vendor in vendors:
                print(f"\n  Checking vendor: {vendor}")
                mem = _load_memory(vendor)
                baseline = mem.get("baseline")
                events = mem.get("events", [])
                
                if baseline:
                    print(f"    ✓ Baseline: {len(baseline)} fields")
                    for k, v in baseline.items():
                        print(f"      - {k}: {v}")
                else:
                    print(f"    ✗ No baseline found")
                
                if events:
                    print(f"    ✓ Events: {len(events)} total")
                    by_type = defaultdict(int)
                    for ev in events:
                        by_type[ev["fact_type"]] += 1
                    for fact_type, count in sorted(by_type.items()):
                        print(f"      - {fact_type}: {count}")
                else:
                    print(f"    ✗ No events found")
        else:
            result["errors"].append("No vendors found in Hindsight")
            print("⚠️  Hindsight connected but NO vendors found")
            print("   Next steps:")
            print("   1. Check /pipeline to upload vendor documents")
            print("   2. Upload baseline contract to register vendor")
            print("   3. Upload invoices/incidents as events")
    
    except Exception as e:
        result["errors"].append(str(e))
        print(f"❌ Error connecting to Hindsight: {e}")
        print("   Check HINDSIGHT_API_KEY in .env file")
    
    print("\n" + "="*70)
    return result


def debug_vendor_data(vendor_id: str) -> None:
    """
    Deep dive into what's stored for a specific vendor.
    Use when: "Agent not showing data for [vendor_name]"
    """
    print("\n" + "="*70)
    print(f"[DEBUG] Detailed data check for: {vendor_id}")
    print("="*70)
    
    print(f"\nLoading memory for '{vendor_id}'...")
    mem = _load_memory(vendor_id)
    
    baseline = mem.get("baseline")
    events = mem.get("events", [])
    
    print(f"\n📋 BASELINE DATA:")
    if baseline:
        print(f"  Type: {type(baseline)}")
        print(f"  Fields: {len(baseline)}")
        for k, v in baseline.items():
            print(f"    {k}: {repr(v)[:80]}")
    else:
        print("  [NONE] - No baseline found")
    
    print(f"\n📊 EVENTS DATA:")
    print(f"  Total events: {len(events)}")
    if events:
        print(f"  Type: {type(events)}")
        for i, ev in enumerate(events, 1):
            print(f"\n  Event {i}:")
            for k, v in ev.items():
                print(f"    {k}: {repr(v)[:80]}")
    else:
        print("  [NONE] - No events found")
    
    print(f"\n🧠 CONTEXT BLOCK (as seen by LLM):")
    print("─" * 70)
    context = build_context_block(vendor_id)
    print(context)
    print("─" * 70)
    print("\n" + "="*70)


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # First, verify Hindsight connection
    print("\n🔍 FIRST: Verifying Hindsight connection...")
    verify_hindsight_connection()
    
    print("\n" + "="*70)
    print("PROCEEDING WITH TESTS")
    print("="*70)
    
    SEP = "=" * 60
    print(SEP); print("TEST 1: Context block"); print(SEP)
    print(build_context_block("aws"))

    print(f"\n{SEP}"); print("TEST 2: Question"); print(SEP)
    print(ask_vendorpulse("aws",
        "What is the difference between the old price and new price?"))

    print(f"\n{SEP}"); print("TEST 3: Brief"); print(SEP)
    print(get_negotiation_brief("aws"))

    print(f"\n{SEP}"); print("TEST 4: Drift alerts"); print(SEP)
    badge = {1: "⚪ minor", 2: "🟡 notable", 3: "🔴 BREACH"}
    for a in get_drift_alerts("aws"):
        print(f"{badge.get(a['severity'], '?')} | {a['date']} | {a['summary']}")