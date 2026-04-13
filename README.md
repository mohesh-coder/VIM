# VIM — Vendor Intelligence Management

![VIM Banner](https://github.com/mohesh-coder/VIM/blob/c06819898c3222b791bd74f788a47cb82207f90b/asset/banner.png)

> **Procurement intelligence that treats every vendor document as structured memory.**

VIM catches what contracts forget: silent price hikes, watered-down SLAs at renewal, credits that were promised and never arrived. It ingests raw vendor documents — contracts, invoices, call transcripts, incident reports, renewal drafts — extracts structured facts, scores them against the original baseline, and gives you a queryable timeline of every deviation across every vendor relationship.

---

## The Problem It Solves

Vendor contracts get signed, then forgotten. By the time a renewal comes around, the procurement team has no clear record of what was originally agreed versus what actually happened. VIM fixes this by maintaining a living memory of each vendor relationship from day one.

The Twilio sample dataset illustrates the pattern the system is built to catch: a silent SMS rate hike discovered via invoice ($0.0075 → $0.0081, no notice given), a four-hour outage with an unpaid $1,182 SLA credit, and a renewal document that quietly removes the 99.95% uptime guarantee and replaces it with "commercially reasonable efforts."

---

## Features

**Document Pipeline (`pipeline.py`)** — Two upload modes:

- `baseline` mode extracts the six frozen contract fields: vendor name, agreed price, SLA terms, contract start date, renewal date, and rep name. These become the benchmark everything else is compared against.
- `normal` mode processes any post-contract document — invoice, email thread, call transcript, incident report — extracts structured facts, classifies each one by type (`price_change`, `sla_violation`, `term_change`, `commitment`, `incident`), scores drift severity (1–3) against the baseline, deduplicates, and appends to the vendor's event log.

**Intelligent Agent (`agent.py`)** — Ask free-form questions against accumulated vendor memory:

- "What is the difference between what we agreed and what we're paying?"
- "Has Twilio met its SLA obligations over the past 12 months?"
- "Generate a negotiation brief for the upcoming Twilio renewal."
- Live drift alerts filtered by severity across all vendors.

**Hindsight Memory Layer (`memory.py`)** — All facts are stored in [Hindsight](https://ui.hindsight.vectorize.io/), a key-value memory service designed for LLM applications. One record per vendor baseline (`{vendor_id}:baseline`), one record per event (`{vendor_id}:events`). The storage and reasoning layers are deliberately separate: Hindsight holds facts, Groq interprets them.

**FastAPI Server (`server.py`)** — Unified REST API with two web UIs: a Pipeline UI for document ingestion and an Agent UI for querying, drift alerts, and negotiation briefs.

---

## Architecture and Data Flow

```
Raw documents (PDF, TXT, DOCX, EML)
          │
          ▼
    POST /upload
    (vendor_id + mode)
          │
          ▼
    pipeline.py
    ┌─────────────────────────────────────────────────┐
    │  extract_text()          → plain text            │
    │  extract_baseline()      → 6-field JSON (Groq)  │
    │  extract_events()        → facts list (Groq)    │
    │  classify_drift()        → severity 1-3 (Groq)  │
    │  is_duplicate()          → deduplication check  │
    └─────────────────────────────────────────────────┘
          │
          ▼
    memory.py  →  Hindsight KV store
    ┌────────────────────────────────────────────────────┐
    │  write_baseline()   →  {vendor_id}:baseline        │
    │  append_event()     →  {vendor_id}:events          │
    │  get_baseline()     ←  read frozen contract terms  │
    │  get_events()       ←  read full event timeline    │
    └────────────────────────────────────────────────────┘
          │
          ▼
    agent.py  →  Groq (Qwen 32B)
    ┌────────────────────────────────────────────────────┐
    │  ask_vendorpulse()        → free-form Q&A          │
    │  get_negotiation_brief()  → structured brief       │
    │  build_context_block()    → baseline + events      │
    └────────────────────────────────────────────────────┘
```

**LLM:** Groq (`qwen/qwen3-32b`) — used for baseline extraction, event extraction, drift classification, and agent reasoning. All calls use the same model with purpose-specific system prompts.

**Memory:** Hindsight — two keys per vendor. The `{vendor_id}:baseline` key stores the frozen contract terms. The `{vendor_id}:events` key holds the append-only event log. A vendor registry key tracks all registered vendor IDs.

---

## Project Structure

```
vim/
├── server.py          # FastAPI app — all HTTP endpoints
├── pipeline.py        # Document ingestion and fact extraction
├── agent.py           # LLM reasoning layer (Q&A, briefs, alerts)
├── memory.py          # Hindsight read/write abstraction
├── requirements.txt   # Python dependencies
├── .env               # API keys (not committed)
└── static/
    ├── agent_ui.html      # Chat + drift alerts UI
    └── pipeline_ui.html   # Document upload UI
```

Sample vendor documents (Twilio):

```
docs/
├── contract_2022.txt      # Original MSA — baseline source
├── invoice_2023_01.txt    # First invoice showing undisclosed rate change
├── incident_jun2023.txt   # SLA outage email thread + unpaid credit
├── renewal_feb2024.txt    # Renewal draft with removed SLA protections
└── call_may2024.txt       # Vendor review call transcript
```

---

## Installation and Setup

### Prerequisites

- Python 3.10+
- A [Groq](https://console.groq.com) API key
- A [Hindsight](https://ui.hindsight.vectorize.io/) API key and bank ID

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/mohesh-coder/VIM.git
cd vim

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` includes:

```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
python-multipart>=0.0.9
groq>=0.9.0
hindsight-client>=0.3.0
python-dotenv>=1.0.0
pdfplumber>=0.11.0
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
# Required
GROQ_API_KEY=gsk_...
HINDSIGHT_API_KEY=hsk_...

# Optional — Hindsight defaults
HINDSIGHT_URL=https://api.hindsight.so
BANK_NAME=vendorpulse
```

`BANK_NAME` is the Hindsight bank (namespace) that VIM will read and write to. Use a unique bank name per environment (e.g. `vendorpulse-prod`, `vendorpulse-dev`).

### 4. Place the UI files

```bash
mkdir static
cp agent_ui.html static/
cp pipeline_ui.html static/
```

### 5. Start the server

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

The server will print:

```
VIM running at → http://localhost:8000
```

Open `http://localhost:8000` for the Agent UI and `http://localhost:8000/pipeline` for the document upload UI.

---

## Hindsight Integration

VIM uses [Hindsight](https://ui.hindsight.vectorize.io/) via `hindsight-client` as its memory layer. Hindsight is a key-value store designed for persistent LLM memory — it provides `store()` (write/overwrite) and `append()` (add to a list) operations.

`memory.py` wraps Hindsight with four functions:

| Function | Hindsight operation | Key pattern |
|---|---|---|
| `write_baseline(vendor_id, data)` | `hindsight.store(key, data)` | `{vendor_id}:baseline` |
| `append_event(vendor_id, event)` | `hindsight.append(key, event)` | `{vendor_id}:events` |
| `get_baseline(vendor_id)` | `hindsight.recall(key)` | `{vendor_id}:baseline` |
| `get_events(vendor_id)` | `hindsight.recall(key)` | `{vendor_id}:events` |

If `HINDSIGHT_API_KEY` is not set, `memory.py` falls back to an in-memory dict for local development and testing. The agent and pipeline work identically in both modes.

---

## API Endpoints

### Document ingestion

```
POST /upload
  Form: vendor_id (string), mode ("baseline" | "normal"), files (multipart)

POST /api/ingest
  Body: { "doc_type": "baseline" | "event", "vendor_id": string, "data": {} }
```

### Vendor data

```
GET  /api/vendors                        → list all registered vendor IDs
GET  /api/overview?vendor_id=twilio      → baseline + full event log
GET  /api/overview                       → all vendors
GET  /api/alerts?vendor_id=twilio        → severity >= 2 events
GET  /api/alerts?min_severity=3          → only material breaches, all vendors
```

### Agent

```
POST /api/chat
  Body: { "vendor_id": "twilio", "question": "What SLA credits are outstanding?" }
  vendor_id is optional — omit to query across all vendors

POST /api/brief?vendor_id=twilio         → negotiation brief
```

### Diagnostics

```
GET  /health                             → server + Hindsight status
GET  /api/verify                         → Hindsight connection + vendor inventory
GET  /api/debug/vendor/{vendor_id}       → full raw data as the agent sees it
```

---

## Environment Variables Reference

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | Groq API key — used for all LLM calls |
| `HINDSIGHT_API_KEY` | Yes | Hindsight API key — memory read/write |
| `HINDSIGHT_URL` | No | Hindsight base URL (defaults to `https://api.hindsight.so`) |
| `BANK_NAME` | No | Hindsight bank/namespace (defaults to `VIM`) |

---

## Contributing

Issues and pull requests are welcome. When adding new document types or fact categories, update the system prompts in `pipeline.py` and the relevant enum values in `memory.py`.

---

*Built with Groq · Hindsight · FastAPI · Qwen-32B*
