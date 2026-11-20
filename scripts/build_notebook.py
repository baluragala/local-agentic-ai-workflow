"""Builds notebooks/LW1_S2_Local_Agentic_AI_Workflow.ipynb.

Run:  python scripts/build_notebook.py
Cells tagged "server" need a live Ollama server; the offline test harness
(scripts/test_notebook_offline.py) skips them and injects a fake client.
"""
from pathlib import Path

import nbformat as nbf

CELLS: list = []


def md(text: str):
    CELLS.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text: str, tags=()):
    cell = nbf.v4.new_code_cell(text.strip("\n"))
    if tags:
        cell.metadata["tags"] = list(tags)
    CELLS.append(cell)


# =============================================================================
# TITLE
# =============================================================================
md(r"""
# Building a Local Agentic AI Workflow
### LW1 · S2 — Hands-on demo (Google Colab compatible)

<a href="https://colab.research.google.com/" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

In this session we build a **small, fully transparent agentic workflow** that runs on a **local open-weight model served by Ollama**. No cloud LLM API, no API keys, no agent framework. Everything is plain Python, so you can see every moving part.

**Business scenario:** *ShopEase Returns Desk*. A customer writes in about an order, and the agent must decide what to do. It can look up the order, check the return policy, calculate the refund and open a return ticket. If information is missing, it asks the customer for it.

| Section | Topic | Time |
|---|---|---|
| 1 | Business problem & the agentic loop | 20 min |
| 2 | Setting up a small local model with Ollama | 25 min |
| 3 | Building the agentic workflow, piece by piece | 55 min |
| 4 | End-to-end business workflow demo (incl. failures) | 40 min |
| 5 | Evaluating the agent with multiple test cases | 30 min |
| 6 | Recap, debugging & Q&A | 10 min |

### How to run this notebook
* **Google Colab (recommended):** `Runtime → Change runtime type → T4 GPU` (CPU also works, just slower). Then run cells top to bottom. Section 2 installs and starts Ollama *inside* the Colab VM.
* **Local machine:** install Ollama from <https://ollama.com/download> and make sure it is running (`ollama serve` or the desktop app). The same cells detect that Ollama is already up and skip the installation.

> **Key idea for the whole session:** the **model only proposes** (text, plans, tool-call requests).
> The **application (orchestrator) disposes**. It validates and executes the tools, keeps the state and decides when to stop.
""")

# =============================================================================
# SECTION 1
# =============================================================================
md(r"""
---
# Section 1 — Business Problem and Agentic Workflow  *(20 min)*

## 1.1 The business task

ShopEase is an online store. Its returns desk receives free-text customer messages such as:

> *"Hi, my headphones from order ORD-1002 arrived with a cracked headband. I'd like to return them."*

| | |
|---|---|
| **Input** | One customer message (free text) |
| **Business goal** | Resolve the request correctly and according to policy |
| **Expected outcome** | One of: **return ticket created** with the correct refund · **return declined** with the policy reason · **order status answered** · **customer asked for missing info** · **out-of-scope reply** |
| **Business rules** | Return windows per category (electronics 15 days, apparel/home 30 days, gift cards non-returnable). A 10% restocking fee applies when the customer simply changed their mind. |

### Tools available to the agent

| Tool | Input | Output |
|---|---|---|
| `get_order` | `order_id` | order details (item, category, price, status, delivery date) |
| `check_return_eligibility` | `order_id` | `eligible` true/false + reason |
| `calculate_refund` | `order_id`, `return_reason` | item price, restocking fee, refund amount |
| `create_return_ticket` | `order_id`, `return_reason`, `refund_amount` | ticket id |

## 1.2 Why an *agentic* approach here?

A fixed script works when the path is always the same. Here the path depends on **intermediate results**:

* The order might **not exist** → stop and tell the customer.
* The item might be **outside the return window** or **non-returnable** → decline; no refund calculation.
* The customer might just ask **"where is my order?"** → a single lookup is enough.
* The customer might **forget the order ID** → ask, don't guess.

Agentic workflows suit tasks that need **dynamic decision-making, conditional tool use, iterative actions and adaptation to intermediate results**. For a fixed, always-identical sequence, a plain program is better and cheaper.

## 1.3 Decomposing the task (the "happy path" of a return)

1. Understand the request → *what does the customer want? which order? why?*
2. `get_order` → does the order exist?
3. `check_return_eligibility` → is it within policy?
4. `calculate_refund` → how much do we refund?
5. `create_return_ticket` → record the return.
6. Reply to the customer with the outcome.

Steps 3–5 are **conditional**. Whether they run depends on what earlier steps returned.

## 1.4 The agentic loop we will build

```
      ┌──────────────┐
      │ Receive task │  customer message
      └──────┬───────┘
             ▼
      ┌──────────────┐   LLM (structured output)
      │ Understand + │── intent, order_id, reason, plan
      │    plan      │
      └──────┬───────┘
             ▼
     missing info? ──yes──► ask the customer  ──► END (needs_info)
             │ no
             ▼
   ┌──────────────────────┐
   │ LLM: decide next     │◄──────────────────────────┐
   │ action (tool call or │                           │
   │ final answer)        │                           │
   └─────────┬────────────┘                           │
             │ tool call requested                    │
             ▼                                        │
   ┌──────────────────────┐    ┌──────────────────┐   │
   │ APP: validate call   │──► │ APP: execute tool│───┘ result appended
   └──────────────────────┘    └──────────────────┘     to state/messages
             │ no tool call (final answer)
             ▼
   APP: termination checks ──► END (ticket_created / declined / answered / …)
```

## 1.5 Who does what?

| Component | Role in this demo |
|---|---|
| **Model** (`qwen2.5:1.5b` via Ollama) | Generates text, plans and **tool-call requests**. It never runs code itself. |
| **Instructions** (system prompts) | Tell the model the goal, rules, tool order and answer style. |
| **Tools** (Python functions) | Do the real work: read orders, apply the policy, create tickets. |
| **Workflow / orchestrator** (our Python loop) | Calls the model, validates & executes tool calls, enforces rules, decides when to stop. |
| **State** (`AgentState`) | Memory of the run: request, plan, messages, tool calls, results, status, final answer. |
| **Execution environment** | Colab VM / your laptop: Python process + Ollama server on `localhost:11434`. |

> ⚠️ **Scope.** We are demonstrating *how an agentic workflow works*. This is not a production-grade or high-performance system. A 1.5B-parameter model will sometimes make mistakes, and that's useful: it gives us real failures to debug and evaluate.
""")

# =============================================================================
# SECTION 2
# =============================================================================
md(r"""
---
# Section 2 — Setting Up a Small Local Model with Ollama  *(25 min)*

**Ollama** is a lightweight runtime that downloads open-weight models and serves them over a local HTTP API (`http://localhost:11434`). Our Python application talks to it like any other web service.

```
 Python app  ──HTTP /api/chat──►  Ollama server  ──►  model weights (qwen2.5:1.5b, ~1 GB)
             ◄── JSON response ──
```

### Model choice: `qwen2.5:1.5b`
* **Small:** ~1 GB download, runs on a free Colab CPU/T4 or an ordinary laptop.
* **Instruction-tuned** and trained for **tool calling** (Ollama exposes it via the `tools` parameter).
* Works with Ollama **structured outputs** (`format=<JSON schema>`), which we use for planning.

That is enough to show planning, tool selection and multi-step execution. If you want more reliability, swap in `qwen2.5:3b` or `llama3.2:3b` by changing `MODEL` below. Nothing else changes.
""")

md("## 2.1 Install Python packages")
code(r"""
%pip install -q "ollama>=0.4" pydantic pandas requests
""", tags=["server"])

md("## 2.2 Configuration & environment detection")
code(r"""
import os, sys, json, re, time, shutil, subprocess
from datetime import date

MODEL = "qwen2.5:1.5b"          # try "qwen2.5:3b" or "llama3.2:3b" for a stronger model
OLLAMA_HOST = "http://127.0.0.1:11434"
IN_COLAB = "google.colab" in sys.modules

print(f"Python      : {sys.version.split()[0]}")
print(f"Running in  : {'Google Colab' if IN_COLAB else 'local Jupyter'}")
print(f"Model       : {MODEL}")
""")

md(r"""
## 2.3 Install Ollama (Colab only)
On Colab we install the Linux build of Ollama with the official script. `zstd` is needed to unpack recent releases. Locally, install it once from <https://ollama.com/download> instead.
""")
code(r"""
def sh(cmd: str) -> int:
    # Run a shell command and show the tail of its output.
    print("$", cmd)
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    out = (r.stdout + r.stderr).strip()
    if out:
        print(out[-1500:])
    return r.returncode

if shutil.which("ollama"):
    print("Ollama binary already installed:", shutil.which("ollama"))
elif IN_COLAB:
    sh("apt-get -qq update && apt-get -qq install -y zstd pciutils > /dev/null")
    sh("curl -fsSL https://ollama.com/install.sh | sh")
else:
    print("Ollama not found. Install it from https://ollama.com/download and start it, then re-run.")
""", tags=["server"])

md(r"""
## 2.4 Start the Ollama server in the background
The server is a separate process. Our notebook is its **client**.
""")
code(r"""
import requests

def ollama_is_up(host: str = OLLAMA_HOST) -> bool:
    try:
        return requests.get(f"{host}/api/version", timeout=2).ok
    except requests.RequestException:
        return False

if not ollama_is_up() and shutil.which("ollama"):
    log = open("ollama_server.log", "w")
    server = subprocess.Popen(["ollama", "serve"], stdout=log, stderr=subprocess.STDOUT)
    for _ in range(30):
        if ollama_is_up():
            break
        time.sleep(1)

if ollama_is_up():
    print("✅ Ollama server is up:", requests.get(f"{OLLAMA_HOST}/api/version").json())
else:
    print("❌ Ollama server not reachable. Check ollama_server.log (Colab) or start the Ollama app (local).")
""", tags=["server"])

md("## 2.5 Pull the model\nThe first pull downloads ~1 GB. That takes 1–2 minutes on Colab.")
code(r"""
from ollama import Client

client = Client(host=OLLAMA_HOST)

t0 = time.time()
last = None
for progress in client.pull(MODEL, stream=True):
    if progress.status != last:          # print each new phase once
        print(progress.status)
        last = progress.status
print(f"Pulled {MODEL} in {time.time() - t0:.0f}s")
print("Local models:", [m.model for m in client.list().models])
""", tags=["server"])

md(r"""
## 2.6 First local inference: raw HTTP
Before using any SDK, here is the call as a **plain HTTP request**. The SDK is only a convenience wrapper around it.
""")
code(r"""
payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "In one sentence: what is a product return policy?"}],
    "stream": False,
    "options": {"temperature": 0},
}
r = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=300)
data = r.json()
print(json.dumps({k: data.get(k) for k in ["model", "message", "done_reason", "prompt_eval_count", "eval_count"]}, indent=2))
print(f"total time: {data['total_duration'] / 1e9:.2f}s   (the first call also loads the model into memory)")
""", tags=["server"])

md("## 2.7 The same call through the `ollama` Python client")
code(r"""
resp = client.chat(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a concise customer-support assistant."},
        {"role": "user", "content": "A customer says their parcel is late. Reply in two sentences."},
    ],
    options={"temperature": 0},
)
print(resp.message.content)
print(f"\ntokens in={resp.prompt_eval_count}  out={resp.eval_count}  time={resp.total_duration / 1e9:.2f}s")
""")

md(r"""
## 2.8 Structured output: getting machine-readable answers
An agent needs **parseable** model outputs, not just prose. Ollama can constrain the model to a JSON schema with `format=`. We define the schema with Pydantic.
""")
code(r"""
from typing import Literal, Optional
from pydantic import BaseModel, ValidationError

class TicketTriage(BaseModel):
    sentiment: Literal["positive", "neutral", "negative"]
    urgency: Literal["low", "medium", "high"]
    summary: str

resp = client.chat(
    model=MODEL,
    messages=[{"role": "user", "content": "Classify this message: 'This is the THIRD time my order arrived broken. Fix it today!'"}],
    format=TicketTriage.model_json_schema(),
    options={"temperature": 0},
)
print("raw  :", resp.message.content)
triage = TicketTriage.model_validate_json(resp.message.content)
print("typed:", triage)
""")

md(r"""
## 2.9 Tool calling: the model *requests*, the application *executes*
We describe a tool with a JSON schema and pass it in `tools=`. Look at what comes back. The response contains a **tool-call request**, not a result: **the model cannot run code.**
""")
code(r"""
def get_shipping_cost(weight_kg: float, express: bool = False) -> dict:
    base = 4.0 + 1.5 * weight_kg
    return {"cost_usd": round(base * (2 if express else 1), 2)}

toy_tools = [{
    "type": "function",
    "function": {
        "name": "get_shipping_cost",
        "description": "Calculate the shipping cost of a parcel in USD.",
        "parameters": {
            "type": "object",
            "properties": {
                "weight_kg": {"type": "number", "description": "Parcel weight in kilograms"},
                "express": {"type": "boolean", "description": "True for express delivery"},
            },
            "required": ["weight_kg"],
        },
    },
}]

messages = [{"role": "user", "content": "How much does it cost to ship a 3 kg parcel by express?"}]
resp = client.chat(model=MODEL, messages=messages, tools=toy_tools, options={"temperature": 0})

print("content   :", repr(resp.message.content))
print("tool_calls:", resp.message.tool_calls)
""")

md("Now **we** (the application) execute the requested tool and send the result back so the model can phrase the answer:")
code(r"""
messages.append(resp.message)                         # the assistant's tool-call request
for call in resp.message.tool_calls or []:
    result = get_shipping_cost(**call.function.arguments)  # APPLICATION executes the tool
    print(f"executed {call.function.name}({call.function.arguments}) -> {result}")
    messages.append({"role": "tool", "content": json.dumps(result), "tool_name": call.function.name})

final = client.chat(model=MODEL, messages=messages, tools=toy_tools, options={"temperature": 0})
print("\nfinal answer:", final.message.content)
""")

md(r"""
That round trip, **model requests → app executes → result goes back → model continues**, is the core of every agent. In Section 3 we put it in a loop and add planning, validation, state and termination.

### Small local models: benefits & limitations

| Benefits | Limitations |
|---|---|
| Free, private: data never leaves the machine | Weaker reasoning; can skip steps or stop early |
| No API keys or rate limits | Can hallucinate tool names, arguments or numbers |
| Fast iteration, works offline | Sensitive to prompt wording; occasional malformed output |
| Great for **learning the mechanics** | Not a benchmark of what large models can do |

That is why our **orchestrator** validates calls, enforces business rules and checks completion. Those guardrails matter even more with a small model.
""")

# =============================================================================
# SECTION 3
# =============================================================================
md(r"""
---
# Section 3 — Building the Agentic Workflow  *(55 min)*

We build the agent **incrementally**:

1. Workflow specification (goal, instructions, inputs/outputs, termination)
2. Mock business data
3. Tools as plain Python functions (tested **without** the LLM)
4. Tool schemas (what the model sees)
5. Validation + execution layer (the application's job)
6. Workflow state
7. LLM wrapper
8. Planner (structured output) + missing-info gate
9. One step by hand
10. The full loop
11. A complete traced execution
""")

md(r"""
## 3.1 Workflow specification
Write the contract down **before** writing any agent code.
""")
code(r"""
WORKFLOW_SPEC = {
    "goal": "Resolve a customer's returns-desk message correctly and according to ShopEase policy.",
    "input": "One free-text customer message.",
    "outputs": "A customer-facing reply + a workflow status + (if applicable) a return ticket.",
    "tools": ["get_order", "check_return_eligibility", "calculate_refund", "create_return_ticket"],
    "missing_info_behaviour": "If the order ID (or, for returns, the return reason) is missing, call NO tools and ask the customer for it.",
    "termination": [
        "model replies without requesting a tool (after orchestrator completion checks)",
        "missing information detected -> needs_info",
        "request out of scope -> out_of_scope",
        "max_steps reached -> max_steps (safety stop)",
    ],
    "statuses": ["ticket_created", "declined", "answered", "order_not_found", "needs_info", "out_of_scope", "incomplete", "max_steps"],
}
for k, v in WORKFLOW_SPEC.items():
    print(f"{k:24}: {v}")
""")

md(r"""
## 3.2 Mock business data
A tiny in-memory "database". We **freeze today's date** so that results are reproducible on any day you run the notebook.
""")
code(r"""
TODAY = date(2026, 3, 15)   # frozen "today" for reproducible policy checks

ORDERS = {
    "ORD-1001": {"order_id": "ORD-1001", "customer": "Priya",  "item": "Wireless Mouse",              "category": "electronics", "price": 25.99,  "status": "delivered", "delivered_on": "2026-03-10"},
    "ORD-1002": {"order_id": "ORD-1002", "customer": "Rahul",  "item": "Noise-Cancelling Headphones", "category": "electronics", "price": 149.00, "status": "delivered", "delivered_on": "2026-03-08"},
    "ORD-1003": {"order_id": "ORD-1003", "customer": "Ananya", "item": "Running Shoes",               "category": "apparel",     "price": 80.00,  "status": "delivered", "delivered_on": "2026-01-20"},
    "ORD-1004": {"order_id": "ORD-1004", "customer": "Vikram", "item": "Yoga Mat",                    "category": "home",        "price": 40.00,  "status": "delivered", "delivered_on": "2026-03-01"},
    "ORD-1005": {"order_id": "ORD-1005", "customer": "Meera",  "item": "Gift Card",                   "category": "gift_card",   "price": 50.00,  "status": "delivered", "delivered_on": "2026-03-12"},
    "ORD-1006": {"order_id": "ORD-1006", "customer": "Arjun",  "item": "Desk Lamp",                   "category": "home",        "price": 35.50,  "status": "shipped",   "delivered_on": None, "expected_delivery": "2026-03-18"},
}

RETURN_WINDOW_DAYS = {"electronics": 15, "apparel": 30, "home": 30, "gift_card": 0}   # 0 = non-returnable
RESTOCKING_FEE = {"damaged": 0.0, "defective": 0.0, "wrong_item": 0.0, "changed_mind": 0.10}
RETURN_REASONS = list(RESTOCKING_FEE)

TICKETS: dict = {}   # created return tickets live here

import pandas as pd
pd.DataFrame(ORDERS.values())
""")

md(r"""
## 3.3 Tools: plain Python functions
Tools are **ordinary, deterministic code**. Notice that they contain *no AI at all*. Test them directly first: if a tool is wrong, no prompt will fix the agent.
""")
code(r"""
def normalize_order_id(raw) -> str:
    # Accept 'ORD-1002', 'ord1002', '#1002', 1002 ... -> 'ORD-1002'.
    m = re.search(r"(\d{4})", str(raw))
    return f"ORD-{m.group(1)}" if m else str(raw).strip().upper()


def get_order(order_id: str) -> dict:
    oid = normalize_order_id(order_id)
    order = ORDERS.get(oid)
    if order is None:
        return {"error": f"Order {oid} not found"}
    return dict(order)


def check_return_eligibility(order_id: str) -> dict:
    order = get_order(order_id)
    if "error" in order:
        return order
    oid, category = order["order_id"], order["category"]
    window = RETURN_WINDOW_DAYS.get(category, 0)
    if order["status"] != "delivered":
        return {"order_id": oid, "eligible": False,
                "reason": f"Order is '{order['status']}' and not delivered yet; returns open after delivery."}
    if window == 0:
        return {"order_id": oid, "eligible": False, "reason": f"{category} items are non-returnable."}
    days = (TODAY - date.fromisoformat(order["delivered_on"])).days
    eligible = days <= window
    return {"order_id": oid, "eligible": eligible, "days_since_delivery": days, "return_window_days": window,
            "reason": f"Delivered {days} days ago; the {category} return window is {window} days."}


def calculate_refund(order_id: str, return_reason: str) -> dict:
    order = get_order(order_id)
    if "error" in order:
        return order
    if return_reason not in RESTOCKING_FEE:
        return {"error": f"Unknown return_reason '{return_reason}'. Use one of {RETURN_REASONS}."}
    fee = round(order["price"] * RESTOCKING_FEE[return_reason], 2)
    return {"order_id": order["order_id"], "item_price": order["price"],
            "restocking_fee": fee, "refund_amount": round(order["price"] - fee, 2)}


def create_return_ticket(order_id: str, return_reason: str, refund_amount: float) -> dict:
    oid = normalize_order_id(order_id)
    ticket_id = f"RET-{oid[-4:]}"
    TICKETS[ticket_id] = {"ticket_id": ticket_id, "order_id": oid, "return_reason": return_reason,
                          "refund_amount": float(refund_amount), "status": "open"}
    return {"ticket_id": ticket_id, "order_id": oid, "refund_amount": float(refund_amount),
            "status": "open", "note": "Prepaid return label emailed to the customer."}


TOOLS = {f.__name__: f for f in [get_order, check_return_eligibility, calculate_refund, create_return_ticket]}

# --- unit-test the tools, no LLM involved ---
print(get_order("#1002"))
print(check_return_eligibility("ORD-1002"))
print(check_return_eligibility("ORD-1003"))
print(check_return_eligibility("ORD-1005"))
print(calculate_refund("ORD-1004", "changed_mind"))
print(get_order("ORD-9999"))
""")

md(r"""
## 3.4 Tool schemas: what the model sees
The model never sees our Python code, only these **names, descriptions and parameter schemas**. Good descriptions are the model's main signal for *which* tool to choose and *when*.
""")
code(r"""
def tool_schema(name, description, properties, required):
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required}}}

ORDER_ID = {"type": "string", "description": "Order ID such as ORD-1002"}
REASON = {"type": "string", "enum": RETURN_REASONS, "description": "Why the customer is returning the item"}

TOOL_SCHEMAS = [
    tool_schema("get_order", "Look up an order. Always call this first to confirm the order exists.",
                {"order_id": ORDER_ID}, ["order_id"]),
    tool_schema("check_return_eligibility", "Check whether a delivered order can be returned under the return policy.",
                {"order_id": ORDER_ID}, ["order_id"]),
    tool_schema("calculate_refund", "Calculate the refund amount for an ELIGIBLE return.",
                {"order_id": ORDER_ID, "return_reason": REASON}, ["order_id", "return_reason"]),
    tool_schema("create_return_ticket", "Create the return ticket. Use the exact refund_amount from calculate_refund.",
                {"order_id": ORDER_ID, "return_reason": REASON,
                 "refund_amount": {"type": "number", "description": "Refund in USD from calculate_refund"}},
                ["order_id", "return_reason", "refund_amount"]),
]
SCHEMA_BY_NAME = {s["function"]["name"]: s["function"]["parameters"] for s in TOOL_SCHEMAS}
print(json.dumps(TOOL_SCHEMAS[2], indent=2))
""")

md(r"""
## 3.5 Workflow state
Everything the orchestrator knows about one run lives in **one object**. Every event is logged with the **layer** it belongs to, so that we can always tell *who* did *what*:

| Layer | Meaning |
|---|---|
| `LLM` | the model generated something (plan, tool request, answer) |
| `PLAN` | planning / missing-info decisions |
| `VALIDATE` | the application checked a tool request |
| `TOOL` | the application executed a tool |
| `STATE` | workflow / state management (start, nudges, termination) |
""")
code(r"""
from dataclasses import dataclass, field

ICONS = {"LLM": "🧠", "PLAN": "📋", "VALIDATE": "🛡️", "TOOL": "🔧", "STATE": "📦"}

@dataclass
class AgentState:
    request: str
    plan: dict | None = None
    messages: list = field(default_factory=list)
    tool_calls: list = field(default_factory=list)    # {"name","args","valid","errors","result"}
    trace: list = field(default_factory=list)         # every event, tagged with its layer
    status: str = "running"
    final_answer: str | None = None
    step: int = 0
    verbose: bool = True

    def log(self, layer: str, event: str, detail="") -> None:
        self.trace.append({"step": self.step, "layer": layer, "event": event, "detail": detail})
        if self.verbose:
            text = detail if isinstance(detail, str) else json.dumps(detail, default=str)
            print(f"[step {self.step}] {ICONS.get(layer, '•')} {layer:<8} {event:<22} {text[:220]}")

    def last_result(self, tool_name: str):
        for tc in reversed(self.tool_calls):
            if tc["name"] == tool_name and tc["valid"]:
                return tc["result"]
        return None

    def tools_used(self) -> list:
        return [tc["name"] for tc in self.tool_calls if tc["valid"]]

demo = AgentState("hello")
demo.log("STATE", "received", "hello")
""")

md(r"""
## 3.6 Validation & execution: the application's job
Before running *anything* the model asked for, the orchestrator checks:

1. **Does the tool exist?** (small models invent tool names)
2. **Are the arguments well-formed?** (required present, no extras, enums respected; numbers sent as strings are *coerced*)
3. **Business preconditions:** the workflow order the business requires, enforced in code, not left to the model:
   * `calculate_refund` only after a **successful, eligible** `check_return_eligibility`
   * `create_return_ticket` only after `calculate_refund`, and with **exactly** the calculated amount

A rejected call is **not executed**. The error goes back to the model so it can correct itself.
""")
code(r"""
def coerce_args(name: str, args) -> dict:
    # Fix harmless type slips, e.g. refund_amount sent as the string "149.0".
    if not isinstance(args, dict):
        return args
    props = SCHEMA_BY_NAME.get(name, {}).get("properties", {})
    fixed = dict(args)
    for k, v in args.items():
        if props.get(k, {}).get("type") == "number" and isinstance(v, str):
            try:
                fixed[k] = float(v.replace("$", "").strip())
            except ValueError:
                pass
    return fixed


def validate_tool_call(name: str, args, state: AgentState) -> list:
    if name not in TOOLS:
        return [f"Unknown tool '{name}'. Available tools: {list(TOOLS)}"]
    if not isinstance(args, dict):
        return ["Arguments must be a JSON object."]
    schema, errors = SCHEMA_BY_NAME[name], []
    for req in schema["required"]:
        if args.get(req) in (None, ""):
            errors.append(f"Missing required argument '{req}'.")
    for k, v in args.items():
        spec = schema["properties"].get(k)
        if spec is None:
            errors.append(f"Unexpected argument '{k}'.")
        elif "enum" in spec and v not in spec["enum"]:
            errors.append(f"'{k}' must be one of {spec['enum']}, got '{v}'.")
        elif spec["type"] == "number" and not isinstance(v, (int, float)):
            errors.append(f"'{k}' must be a number, got {v!r}.")
    if errors:
        return errors

    # --- business preconditions (workflow rules enforced by the application) ---
    if name == "calculate_refund":
        elig = state.last_result("check_return_eligibility")
        if not elig or elig.get("eligible") is not True:
            errors.append("Precondition failed: call check_return_eligibility first; refunds are only for eligible returns.")
    if name == "create_return_ticket":
        refund = state.last_result("calculate_refund")
        if not refund or "refund_amount" not in refund:
            errors.append("Precondition failed: call calculate_refund before create_return_ticket.")
        elif abs(float(args["refund_amount"]) - refund["refund_amount"]) > 0.01:
            errors.append(f"refund_amount {args['refund_amount']} does not match calculated {refund['refund_amount']}.")
    return errors


def execute_tool(name: str, args: dict) -> dict:
    try:
        return TOOLS[name](**args)
    except Exception as e:                    # a crashing tool must not crash the agent
        return {"error": f"{type(e).__name__}: {e}"}


# --- try the validator on some bad requests ---
s = AgentState("test", verbose=False)
print(validate_tool_call("issue_refund", {"order_id": "ORD-1002"}, s))
print(validate_tool_call("get_order", {"order": 1002}, s))
print(validate_tool_call("calculate_refund", {"order_id": "ORD-1002", "return_reason": "damaged"}, s))
print(coerce_args("create_return_ticket", {"order_id": "ORD-1002", "return_reason": "damaged", "refund_amount": "$149"}))
""")

md(r"""
## 3.7 LLM wrapper
One function owns all model calls. It returns a normalised `(assistant_message, tool_calls)` pair. Swapping models (or using a scripted fake model for tests, see Section 4) only touches this one place.

It also handles a **classic small-model failure**: the model writes the tool call as JSON *text* in `content` instead of a native `tool_calls` entry. We detect that and recover it.
""")
code(r"""
def _tool_call_from_text(content: str):
    # Recover {"name": ..., "arguments": {...}} written as plain text.
    m = re.search(r"\{.*\}", content or "", re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict) and obj.get("name") in TOOLS:
        return {"name": obj["name"], "arguments": obj.get("arguments") or obj.get("parameters") or {}}
    return None


def ollama_backend(messages, tools=None, format=None):
    resp = client.chat(model=MODEL, messages=messages, tools=tools, format=format,
                       options={"temperature": 0, "num_ctx": 4096})
    content = resp.message.content or ""
    calls = [{"name": tc.function.name, "arguments": dict(tc.function.arguments or {})}
             for tc in (resp.message.tool_calls or [])]
    if tools and not calls:
        recovered = _tool_call_from_text(content)
        if recovered:
            print("   ⚠️  recovered a tool call the model wrote as text:", recovered)
            calls, content = [recovered], ""
    return content, calls


LLM_BACKEND = ollama_backend          # Section 4 swaps this for a scripted model


def call_llm(messages, tools=None, format=None):
    content, calls = LLM_BACKEND(messages, tools=tools, format=format)
    msg = {"role": "assistant", "content": content}
    if calls:
        msg["tool_calls"] = [{"function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]
    return msg, calls


msg, calls = call_llm([{"role": "user", "content": "Say hello to a ShopEase customer in 5 words."}])
print(msg)
""")

md(r"""
## 3.8 Planning: understanding the task + deciding the approach
The first model call is a **planner**. It turns free text into a structured plan: intent, extracted order ID and reason, and the steps it proposes. Structured output (`format=`) guarantees we can parse it.

The plan is the **model's proposal**. The **application** then applies the defined **missing-information behaviour**, plus a guard against a classic hallucination: an order ID that the customer never wrote.
""")
code(r"""
class Plan(BaseModel):
    intent: Literal["return_request", "order_status", "other"]
    order_id: Optional[str] = None
    return_reason: Optional[Literal["damaged", "defective", "wrong_item", "changed_mind"]] = None
    steps: list[str]

PLANNER_PROMPT = '''You are the planning module of the ShopEase returns desk. Read the customer message and reply with JSON only.
- intent: "return_request" if they want to return an item or get a refund; "order_status" if they ask where their order is or its status; otherwise "other".
- order_id: the order ID exactly as the customer wrote it (for example ORD-1002). Use null if the message has no order ID. Never invent one.
- return_reason: damaged, defective, wrong_item or changed_mind, or null if no reason is given.
  damaged = arrived broken/cracked; defective = stopped working/faulty; wrong_item = received a different item or size; changed_mind = no longer wants/needs it.
- steps: the ordered list of tools you would use. Tools: get_order, check_return_eligibility, calculate_refund, create_return_ticket.'''

REQUIRED_INFO = {"return_request": ["order_id", "return_reason"], "order_status": ["order_id"], "other": []}


def make_plan(state: AgentState) -> Plan | None:
    msg, _ = call_llm([{"role": "system", "content": PLANNER_PROMPT},
                       {"role": "user", "content": state.request}],
                      format=Plan.model_json_schema())
    state.log("LLM", "plan generated", msg["content"])
    try:
        plan = Plan.model_validate_json(msg["content"])
    except ValidationError as e:
        state.log("PLAN", "plan INVALID", str(e).splitlines()[0])
        return None
    # Guardrail: the order number must actually appear in the customer's text.
    if plan.order_id:
        digits = re.findall(r"\d{3,}", plan.order_id)
        if not digits or digits[0] not in state.request:
            state.log("PLAN", "hallucinated order_id", f"{plan.order_id!r} not in request -> treated as missing")
            plan.order_id = None
    state.plan = plan.model_dump()
    return plan


def missing_info(plan: Plan) -> list:
    return [f for f in REQUIRED_INFO[plan.intent] if getattr(plan, f) in (None, "")]


s = AgentState("My headphones from ORD-1002 arrived cracked, I want to return them.")
p = make_plan(s)
print("\nplan   :", p)
print("missing:", missing_info(p))
s2 = AgentState("I want to return my shoes, they don't fit.")
p2 = make_plan(s2)
print("\nplan   :", p2)
print("missing:", missing_info(p2))
""")

md(r"""
## 3.9 One step by hand: next-action selection
Before automating, let's run **one** decision step manually so we can see exactly what flows where.
""")
code(r"""
SYSTEM_PROMPT = '''You are the ShopEase returns-desk agent. Resolve the customer's request using the tools.
Rules:
1. Always call get_order first to confirm the order exists. If it is not found, stop and tell the customer.
2. For a return: call check_return_eligibility. If eligible is false, do NOT call more tools; explain the reason to the customer.
3. If eligible is true: call calculate_refund with the return_reason, then create_return_ticket with the exact refund_amount.
4. For an order-status question: call get_order only, then answer.
5. Call one tool at a time. Never invent order details, amounts or ticket IDs; use tool results only.
6. When finished, reply to the customer in 2-4 sentences WITHOUT calling a tool. If a ticket was created, mention the ticket ID and the refund amount.'''


def initial_messages(state: AgentState, plan: Plan) -> list:
    context = (f"[Planner context] intent={plan.intent}; order_id={plan.order_id}; "
               f"return_reason={plan.return_reason}")
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{state.request}\n\n{context}"}]


state = AgentState("My headphones from ORD-1002 arrived cracked, I want to return them.")
plan = make_plan(state)
state.messages = initial_messages(state, plan)

# 1) MODEL decides the next action
msg, calls = call_llm(state.messages, tools=TOOL_SCHEMAS)
print("\nmodel requested:", calls, "| text:", repr(msg["content"]))

# 2) APPLICATION validates + executes, 3) result goes back into the conversation (state)
state.messages.append(msg)
for c in calls:
    args = coerce_args(c["name"], c["arguments"])
    errs = validate_tool_call(c["name"], args, state)
    result = {"error": "; ".join(errs)} if errs else execute_tool(c["name"], args)
    print("tool result    :", result)
    state.messages.append({"role": "tool", "content": json.dumps(result), "tool_name": c["name"]})

# 4) MODEL decides again, now seeing the tool result
msg, calls = call_llm(state.messages, tools=TOOL_SCHEMAS)
print("next request   :", calls, "| text:", repr(msg["content"]))
""")

md(r"""
## 3.10 The full agent loop
Now we automate exactly what we just did by hand, and add:
* **termination conditions** (final answer, `max_steps`, missing info, out of scope)
* a **completion check**: if the model tries to finish while a required step is still open (e.g. eligible but no ticket), the orchestrator nudges it **once**
* a **status derived from tool results**, *not* from the model's wording
""")
code(r"""
NEEDS_INFO_REPLY = ("Happy to help! To process this I need {fields}. "
                    "Could you reply with {it}? (Order IDs look like ORD-1234.)")
OUT_OF_SCOPE_REPLY = ("I'm the ShopEase returns desk, so I can help with returns, refunds and order status. "
                      "For other questions please contact support@shopease.example.")


def completion_gap(state: AgentState) -> str | None:
    # What required step is still open? (None = OK to finish)
    plan = state.plan or {}
    if not state.tool_calls:
        return "You have not called any tool yet. Start with get_order."
    if plan.get("intent") == "return_request":
        elig = state.last_result("check_return_eligibility")
        order = state.last_result("get_order") or {}
        if elig is None and "error" not in order:
            return "You have not checked return eligibility yet. Call check_return_eligibility."
        if elig and elig.get("eligible") and "create_return_ticket" not in state.tools_used():
            return "The return is eligible but no ticket exists. Call calculate_refund (if needed) then create_return_ticket."
    return None


def derive_status(state: AgentState) -> str:
    # The business outcome comes from TOOL RESULTS, never from the model's text.
    order = state.last_result("get_order")
    elig = state.last_result("check_return_eligibility")
    if state.last_result("create_return_ticket"):
        return "ticket_created"
    if order and "error" in order:
        return "order_not_found"
    if elig and elig.get("eligible") is False:
        return "declined"
    if (state.plan or {}).get("intent") == "order_status" and order:
        return "answered"
    return "incomplete"


def run_agent(request: str, max_steps: int = 8, enforce_completion: bool = True, verbose: bool = True) -> AgentState:
    state = AgentState(request, verbose=verbose)
    state.log("STATE", "task received", request)

    # ---- 1. understand + plan --------------------------------------------------
    plan = make_plan(state)
    if plan is None:
        state.status, state.final_answer = "incomplete", "Sorry, I couldn't understand that. Could you rephrase?"
        return state
    state.log("PLAN", "plan accepted", state.plan)

    if plan.intent == "other":
        state.status, state.final_answer = "out_of_scope", OUT_OF_SCOPE_REPLY
        state.log("STATE", "terminate", "out_of_scope -> no tools")
        return state

    missing = missing_info(plan)
    if missing:                                   # defined missing-information behaviour
        names = " and ".join(m.replace("_", " ") for m in missing)
        state.status = "needs_info"
        state.final_answer = NEEDS_INFO_REPLY.format(fields=f"your {names}", it="that" if len(missing) == 1 else "those")
        state.log("STATE", "terminate", f"needs_info -> missing {missing}; no tools called")
        return state

    # ---- 2. act: next-action loop ---------------------------------------------
    state.messages = initial_messages(state, plan)
    nudged = False
    while state.step < max_steps:
        state.step += 1
        msg, calls = call_llm(state.messages, tools=TOOL_SCHEMAS)
        state.messages.append(msg)

        if not calls:                             # model wants to finish
            gap = completion_gap(state) if enforce_completion else None
            if gap and not nudged:
                nudged = True
                state.log("STATE", "completion check", f"NOT done: {gap}")
                state.messages.append({"role": "user", "content": f"[Orchestrator] {gap}"})
                continue
            state.final_answer = msg["content"]
            state.log("LLM", "final answer", msg["content"])
            break

        for c in calls:
            state.log("LLM", "tool requested", f"{c['name']}({c['arguments']})")
            args = coerce_args(c["name"], c["arguments"])
            errors = validate_tool_call(c["name"], args, state)
            if errors:
                result = {"error": " ".join(errors)}
                state.log("VALIDATE", "REJECTED", result["error"])
            else:
                state.log("VALIDATE", "ok", c["name"])
                result = execute_tool(c["name"], args)
                state.log("TOOL", f"{c['name']}", result)
            state.tool_calls.append({"name": c["name"], "args": args, "valid": not errors,
                                     "errors": errors, "result": result})
            state.messages.append({"role": "tool", "content": json.dumps(result), "tool_name": c["name"]})
    else:
        state.log("STATE", "terminate", f"max_steps={max_steps} reached")
        state.status = "max_steps"
        return state

    state.status = derive_status(state)
    state.log("STATE", "terminate", f"status={state.status}")
    return state
""")

md(r"""
## 3.11 Trace one complete execution
Run the full loop on the damaged-headphones request and read the trace **top to bottom**. Every line is tagged with the layer responsible.
""")
code(r"""
def show_summary(state: AgentState) -> None:
    print("\n" + "=" * 90)
    print("REQUEST     :", state.request)
    print("PLAN        :", state.plan)
    print("TOOL PATH   :", " → ".join(tc["name"] + ("" if tc["valid"] else " ✗") for tc in state.tool_calls) or "(none)")
    print("STATUS      :", state.status)
    print("FINAL ANSWER:", state.final_answer)
    print("=" * 90)


def trace_table(state: AgentState) -> pd.DataFrame:
    df = pd.DataFrame(state.trace)
    df["detail"] = df["detail"].apply(lambda d: d if isinstance(d, str) else json.dumps(d, default=str)).str.slice(0, 120)
    return df


TICKETS.clear()
state = run_agent("Hi, my Noise-Cancelling Headphones from order ORD-1002 arrived with a cracked headband. I want to return them.")
show_summary(state)
trace_table(state)
""")

md(r"""
### Four different things happened. Don't conflate them.

| Concept | Where you saw it in the trace |
|---|---|
| **LLM response generation** | `🧠 LLM final answer`: fluent text for the customer |
| **Agentic planning / next-action selection** | `🧠 LLM plan generated`, `🧠 LLM tool requested`: the model *choosing* what to do |
| **Tool execution** | `🔧 TOOL …`: deterministic Python run **by the application** |
| **Workflow / state management** | `📦 STATE`, `🛡️ VALIDATE`, `📋 PLAN`: the orchestrator controlling, checking and stopping |

Inspect the raw state too. It is the "memory" of the run:
""")
code(r"""
print("tools used :", state.tools_used())
print("tickets db :", TICKETS)
print("messages   :", len(state.messages), "messages in the conversation")
for m in state.messages:
    print(f"  {m['role']:9} | {(m.get('content') or '')[:90]!r} {'+ tool_calls' if m.get('tool_calls') else ''}")
""")

# =============================================================================
# SECTION 4
# =============================================================================
md(r"""
---
# Section 4 — End-to-End Business Workflow Demo  *(40 min)*

Same agent, different inputs → **different valid paths**. Watch the tool path change with the input.

| Scenario | Input | Expected path |
|---|---|---|
| A. Straightforward | order-status question | `get_order` → answer |
| B. Multi-tool | damaged item, in window | `get_order → check_return_eligibility → calculate_refund → create_return_ticket` |
| C. Missing info | no order ID | **no tools** → ask customer |
| D. Variation: different path | outside return window | `get_order → check_return_eligibility` → decline |
| E. Variation: fee applies | changed mind, in window | full path, refund minus 10% |
| F. Non-returnable | gift card | decline after eligibility |
""")
code(r"""
TICKETS.clear()
state_a = run_agent("Hi, where is my order ORD-1006? It hasn't arrived yet.")
show_summary(state_a)
""")
code(r"""
state_b = run_agent("Hello, order ORD-1002: the headphones arrived damaged, the headband is cracked. Please refund me.")
show_summary(state_b)
""")
code(r"""
state_c = run_agent("I want to return my running shoes, they don't fit.")
show_summary(state_c)
""")
code(r"""
state_d = run_agent("I'd like to return the running shoes from ORD-1003, I changed my mind.")
show_summary(state_d)
""")
code(r"""
state_e = run_agent("Please return my yoga mat, order ORD-1004. I don't need it anymore.")
show_summary(state_e)
""")
code(r"""
state_f = run_agent("Can I return the gift card from ORD-1005? I changed my mind.")
show_summary(state_f)
""")

md(r"""
### Side-by-side: how the input changed the path
""")
code(r"""
runs = {"A status": state_a, "B damaged": state_b, "C missing": state_c,
        "D late": state_d, "E mind": state_e, "F giftcard": state_f}
pd.DataFrame([{"scenario": k, "intent": (s.plan or {}).get("intent"), "tool path": " → ".join(s.tools_used()) or "—",
               "status": s.status, "llm steps": s.step} for k, s in runs.items()])
""")

md(r"""
## 4.2 Failures and how to debug them

Agents fail at **different layers**. Three demos follow:

1. **Real input problem:** an order that does not exist.
2. **Bad model output (deterministic):** we replace the model with a *scripted* one that hallucinates a tool name and sends bad arguments. Scripting the model makes the failure reproducible, which is a standard debugging technique.
3. **"Confident but wrong":** the model claims success without doing the work.
""")
code(r"""
state_g = run_agent("My order ORD-9999 arrived damaged, I want a refund.")
show_summary(state_g)
""")

md(r"""
### A scripted model for reproducible debugging
`ScriptedLLM` replays a fixed list of model outputs. The orchestrator can't tell the difference, which also proves that **all control logic lives in the application**.
""")
code(r"""
class ScriptedLLM:
    # Replays canned model outputs: str = text reply, dict = tool call(s).
    def __init__(self, outputs):
        self.outputs, self.i = list(outputs), 0

    def __call__(self, messages, tools=None, format=None):
        out = self.outputs[min(self.i, len(self.outputs) - 1)]
        self.i += 1
        if isinstance(out, dict):
            return "", [out]
        return out, []


def with_backend(backend, fn, *args, **kwargs):
    global LLM_BACKEND
    original, LLM_BACKEND = LLM_BACKEND, backend
    try:
        return fn(*args, **kwargs)
    finally:
        LLM_BACKEND = original


PLAN_JSON = json.dumps({"intent": "return_request", "order_id": "ORD-1002", "return_reason": "damaged",
                        "steps": ["get_order", "check_return_eligibility", "calculate_refund", "create_return_ticket"]})

buggy_model = ScriptedLLM([
    PLAN_JSON,
    {"name": "issue_refund", "arguments": {"order_id": "ORD-1002"}},            # hallucinated tool
    {"name": "get_order", "arguments": {"order": 1002}},                          # wrong argument name
    {"name": "get_order", "arguments": {"order_id": "ORD-1002"}},                 # self-corrected
    {"name": "calculate_refund", "arguments": {"order_id": "ORD-1002", "return_reason": "damaged"}},  # skipped a step!
    {"name": "check_return_eligibility", "arguments": {"order_id": "ORD-1002"}},
    {"name": "calculate_refund", "arguments": {"order_id": "ORD-1002", "return_reason": "damaged"}},
    {"name": "create_return_ticket", "arguments": {"order_id": "ORD-1002", "return_reason": "damaged", "refund_amount": 1490}},  # wrong amount
    {"name": "create_return_ticket", "arguments": {"order_id": "ORD-1002", "return_reason": "damaged", "refund_amount": "149.00"}},
    "Your return RET-1002 is open and $149.00 will be refunded.",
])
TICKETS.clear()
state_bug = with_backend(buggy_model, run_agent, "ORD-1002 headphones arrived damaged", max_steps=12)
show_summary(state_bug)
""")

md(r"""
Every bad request was **caught by the application and never executed**. The error text went back to the "model", and the run still ended in a correct business outcome. Now the third failure type: the model **claims** success without calling the tools. Compare the run **with and without** the orchestrator's completion check:
""")
code(r"""
lying_model = lambda: ScriptedLLM([
    PLAN_JSON,
    {"name": "get_order", "arguments": {"order_id": "ORD-1002"}},
    "Done! Your refund of $149 is processed, ticket RET-7777.",     # no eligibility, no ticket
    "Done! Your refund of $149 is processed, ticket RET-7777.",
])
TICKETS.clear()
print("---- WITHOUT completion check ----")
s_off = with_backend(lying_model(), run_agent, "ORD-1002 headphones arrived damaged", enforce_completion=False)
show_summary(s_off)
print("\n---- WITH completion check ----")
s_on = with_backend(lying_model(), run_agent, "ORD-1002 headphones arrived damaged", enforce_completion=True)
show_summary(s_on)
""")

md(r"""
The answer *sounds* perfect, but `status=incomplete`, no ticket exists and `RET-7777` is invented. With the check on, the orchestrator spots the open step and nudges once. This scripted model ignores the nudge, so the status stays `incomplete`, which is **honest**. A real model usually follows the nudge.

### A debugging helper
Read the state and point at the **layer** that went wrong:
""")
code(r"""
def diagnose(state: AgentState) -> list:
    findings = []
    if state.plan is None and state.status not in ("out_of_scope",):
        findings.append("PLAN: planner output missing/invalid → inspect the 'plan generated' trace line.")
    for tc in state.tool_calls:
        if not tc["valid"]:
            findings.append(f"VALIDATE: rejected {tc['name']}({tc['args']}) → {tc['errors']}")
        elif isinstance(tc["result"], dict) and "error" in tc["result"]:
            findings.append(f"TOOL: {tc['name']} returned error → {tc['result']['error']}")
    if state.status in ("incomplete", "max_steps"):
        findings.append(f"STATE: run ended '{state.status}' → required steps missing; tools used: {state.tools_used()}")
    # numbers / ticket IDs in the answer must come from tool results
    evidence = json.dumps([tc["result"] for tc in state.tool_calls if tc["valid"]])
    for tid in re.findall(r"RET-\d+", state.final_answer or ""):
        if tid not in evidence:
            findings.append(f"LLM: final answer mentions {tid}, which no tool produced (hallucination).")
    for amt in re.findall(r"\$\s?(\d+(?:\.\d+)?)", state.final_answer or ""):
        if not re.search(rf"\b{float(amt):g}\b|{float(amt):.2f}|{float(amt):.1f}", evidence):
            findings.append(f"LLM: final answer mentions ${amt}, not found in any tool result.")
    return findings or ["No problems detected."]

for name, s in [("ORD-9999", state_g), ("buggy model", state_bug), ("lying model", s_off)]:
    print(f"\n{name}:")
    for f in diagnose(s):
        print("  -", f)
""")

md(r"""
### 🧪 Try it yourself
Change the request below and predict the tool path **before** you run it. Ideas:
* `"#1001 the mouse is defective"` (unusual ID format; `normalize_order_id` handles it)
* `"Return ORD-1002 please"` (no reason → what should happen?)
* `"ORD-1006 arrived broken, refund please"` (not delivered yet!)
* `"what's your return policy for shoes?"`
""")
code(r"""
TICKETS.clear()
my_state = run_agent("#1001 the mouse is defective, I want to send it back")
show_summary(my_state)
print(diagnose(my_state))
""")

# =============================================================================
# SECTION 5
# =============================================================================
md(r"""
---
# Section 5 — Evaluating the Agent with Multiple Test Cases  *(30 min)*

One good demo proves nothing: an agent must be **evaluated across many inputs**. We evaluate **workflow correctness and business-task completion**, not fluency.

## 5.1 Evaluation criteria and what counts as "valid"

| # | Criterion | Valid when … | Layer |
|---|---|---|---|
| 1 | Task interpretation | planner `intent` == expected intent | model |
| 2 | Planning / action | the missing-info decision matches expectation | model + workflow |
| 3 | Tool selection | no forbidden tool used and no unknown tool requested | model |
| 4 | Tool execution | every executed tool ran without crashing; at most one rejected request | workflow |
| 5 | Step completion | required tools appear **in order** | workflow |
| 6 | Final output | answer is non-empty and mentions the required facts | model |
| 7 | **Business outcome** | final `status` (+ refund amount / ticket) matches expectation | business |
""")
code(r"""
@dataclass
class TestCase:
    id: str
    request: str
    expected_intent: str
    expected_status: str
    required_tools: list = field(default_factory=list)      # must appear in this order
    forbidden_tools: list = field(default_factory=list)
    expected_refund: float | None = None
    must_mention: list = field(default_factory=list)        # each item: tuple of acceptable alternatives

TEST_CASES = [
    TestCase("TC01-status", "Where is my order ORD-1006?", "order_status", "answered",
             ["get_order"], ["calculate_refund", "create_return_ticket"], must_mention=[("ship", "on its way", "transit", "march 18", "2026-03-18")]),
    TestCase("TC02-damaged", "My headphones from ORD-1002 arrived damaged. I want to return them.", "return_request", "ticket_created",
             ["get_order", "check_return_eligibility", "calculate_refund", "create_return_ticket"], [], 149.00, [("RET-1002",), ("149",)]),
    TestCase("TC03-changed-mind", "I changed my mind about the yoga mat, order ORD-1004. Please take it back.", "return_request", "ticket_created",
             ["get_order", "check_return_eligibility", "calculate_refund", "create_return_ticket"], [], 36.00, [("RET-1004",), ("36",)]),
    TestCase("TC04-outside-window", "I want to return the running shoes from ORD-1003, I changed my mind.", "return_request", "declined",
             ["get_order", "check_return_eligibility"], ["calculate_refund", "create_return_ticket"], must_mention=[("30", "window", "expired", "eligible")]),
    TestCase("TC05-non-returnable", "Can I return the gift card from ORD-1005? I changed my mind.", "return_request", "declined",
             ["get_order", "check_return_eligibility"], ["calculate_refund", "create_return_ticket"], must_mention=[("non-returnable", "not returnable", "cannot be returned", "can't be returned", "not eligible")]),
    TestCase("TC06-missing-id", "I want to return my headphones, they stopped working.", "return_request", "needs_info",
             [], list(TOOLS), must_mention=[("order id",)]),
    TestCase("TC07-not-found", "Order ORD-9999 arrived broken, I want a refund.", "return_request", "order_not_found",
             ["get_order"], ["calculate_refund", "create_return_ticket"], must_mention=[("not find", "couldn't find", "could not find", "not found", "unable to find")]),
    TestCase("TC08-out-of-scope", "Can you recommend a good gaming laptop?", "other", "out_of_scope",
             [], list(TOOLS)),
]
print(len(TEST_CASES), "test cases")
""")

md("## 5.2 The evaluator: compares the **actual execution** with the **requirements**")
code(r"""
def is_subsequence(required, actual) -> bool:
    it = iter(actual)
    return all(t in it for t in required)


def evaluate(tc: TestCase, state: AgentState) -> dict:
    used = state.tools_used()
    requested = [c["name"] for c in state.tool_calls]
    answer = (state.final_answer or "").lower()
    ticket = state.last_result("create_return_ticket")
    rejected = [c for c in state.tool_calls if not c["valid"]]
    crashed = [c for c in state.tool_calls if c["valid"] and "Error:" in str(c["result"].get("error", ""))]

    business_ok = state.status == tc.expected_status
    if tc.expected_refund is not None:
        business_ok &= bool(ticket) and abs(ticket["refund_amount"] - tc.expected_refund) < 0.01

    return {
        "interpretation": (state.plan or {}).get("intent") == tc.expected_intent,
        "planning": (state.status == "needs_info") == (tc.expected_status == "needs_info"),
        "tool_selection": not (set(requested) & set(tc.forbidden_tools)) and all(r in TOOLS for r in requested),
        "tool_execution": not crashed and len(rejected) <= 1,
        "step_completion": is_subsequence(tc.required_tools, used),
        "final_output": bool(answer) and all(any(a.lower() in answer for a in alts) for alts in tc.must_mention),
        "business_outcome": business_ok,
    }


def run_suite(cases, verbose=False) -> pd.DataFrame:
    rows = []
    for tc in cases:
        TICKETS.clear()
        t0 = time.time()
        state = run_agent(tc.request, verbose=verbose)
        scores = evaluate(tc, state)
        rows.append({"case": tc.id, **scores, "status": state.status, "expected": tc.expected_status,
                     "path": " → ".join(state.tools_used()) or "—", "secs": round(time.time() - t0, 1),
                     "_state": state})
        mark = "✅" if scores["business_outcome"] else "❌"
        print(f"{mark} {tc.id:<22} status={state.status:<16} path={rows[-1]['path']}")
    return pd.DataFrame(rows)
""")

md("## 5.3 Run the agent against all test cases")
code(r"""
results = run_suite(TEST_CASES)
CRITERIA = ["interpretation", "planning", "tool_selection", "tool_execution", "step_completion", "final_output", "business_outcome"]
results[["case"] + CRITERIA + ["status", "expected", "path"]].replace({True: "✅", False: "❌"})
""")
code(r"""
print("Pass rate per criterion:")
print((results[CRITERIA].mean() * 100).round(0).astype(int).astype(str) + "%")
print(f"\nBusiness task success: {results['business_outcome'].sum()}/{len(results)}")
""")

md(r"""
## 5.4 Drill into failures
For any ❌ row, open its trace and run `diagnose`. Work out **which stage failed**: interpretation (planner), tool selection, execution, completion or the final answer.
""")
code(r"""
failed = results[~results[CRITERIA].all(axis=1)]
if failed.empty:
    print("All criteria passed for all cases in this run 🎉 (re-run or try a smaller/other model to see failures)")
for _, row in failed.iterrows():
    st = row["_state"]
    bad = [c for c in CRITERIA if not row[c]]
    print(f"\n❌ {row['case']}: failed {bad}")
    print("   final answer:", (st.final_answer or "")[:200])
    for f in diagnose(st):
        print("   -", f)
""")

md(r"""
## 5.5 Three different things to evaluate

| Level | Question | Example signal |
|---|---|---|
| **Model response quality** | Is the reply fluent, polite, accurate? | human review / LLM-as-judge / heuristics |
| **Workflow execution** | Did the agent take allowed actions in the right order? | criteria 1–5 above |
| **Business outcome** | Did the business task actually get done correctly? | criterion 7: ticket + correct refund |

A run can **score well on one level and fail another**. The "lying model" from Section 4 produced a fluent, confident answer but completed **no** business task:
""")
code(r"""
tc = TEST_CASES[1]   # TC02-damaged
fluent_but_wrong = with_backend(lying_model(), run_agent, tc.request, enforce_completion=False, verbose=False)
scores = evaluate(tc, fluent_but_wrong)
print("Answer:", fluent_but_wrong.final_answer)
print("Sounds fluent?   ✅ yes")
for k, v in scores.items():
    print(f"  {k:<17} {'✅' if v else '❌'}")
""")

md(r"""
## 5.6 Extending the evaluation dataset
As the workflow evolves (new tools, new policies, bugs found in production), **every new bug becomes a test case**. Add cases and re-run only those:
""")
code(r"""
NEW_CASES = [
    TestCase("TC09-id-format", "#1001 the mouse is defective, I want to send it back.", "return_request", "ticket_created",
             ["get_order", "check_return_eligibility", "calculate_refund", "create_return_ticket"], [], 25.99, [("RET-1001",)]),
    TestCase("TC10-missing-reason", "Return ORD-1002 please.", "return_request", "needs_info", [], list(TOOLS), must_mention=[("reason",)]),
    TestCase("TC11-not-delivered", "ORD-1006 is broken, I want a refund.", "return_request", "declined",
             ["get_order", "check_return_eligibility"], ["calculate_refund", "create_return_ticket"]),
]
new_results = run_suite(NEW_CASES)
new_results[["case"] + CRITERIA + ["status", "expected"]].replace({True: "✅", False: "❌"})
""")

md(r"""
> **Remember:** the goal of this evaluation is to check whether the workflow **satisfies the defined requirements** across representative cases. Small models are non-deterministic across versions/hardware, so in practice you run each case several times and track the pass rate over time.
""")

# =============================================================================
# SECTION 6
# =============================================================================
md(r"""
---
# Section 6 — Recap, Debugging & Q&A  *(10 min)*

## What we built
```
customer message
   → 🧠 planner (structured output)        : understand + plan
   → 📋 missing-info / scope gate          : defined behaviour, no guessing
   → loop:
        🧠 model proposes next action       : tool call or final answer
        🛡️ app validates (schema + business preconditions)
        🔧 app executes tool, result → state/messages
        📦 app checks completion / max_steps
   → status derived from tool results + customer reply
   → ✅ evaluated against 7 criteria over many test cases
```

## Core concepts
* **Planning:** turning a goal into steps / choosing the next action (model proposes).
* **Tool use:** the model *requests*; the application *validates and executes*.
* **Workflow execution:** the loop, rules, preconditions and termination live in code.
* **State:** one object holds the request, plan, messages, tool calls and status.
* **Iterative decision-making:** each tool result changes the next decision.
* **Evaluation:** multiple test cases × criteria at three levels (response, workflow, business).

## Ollama vs model vs application
* **Ollama** = local model *runtime & HTTP interface* (download, load, serve).
* **qwen2.5:1.5b** = the model that *generates language, plans and tool-call requests*.
* **Our Python app** = the *orchestrator* that controls the agentic workflow.

## Common failure points & how to debug them

| Failure point | Symptom | Debug / mitigation |
|---|---|---|
| Task interpretation | wrong `intent`, invented order ID | inspect planner output; structured output + "must appear in text" guard |
| Tool selection | unknown tool, wrong tool, skipped step | validator rejects; clearer tool descriptions; precondition rules |
| Arguments | wrong names/types, wrong amounts | schema validation, coercion, cross-check against earlier results |
| Tool execution | exceptions, "not found" | tools return structured errors; test tools without the LLM |
| Output format | tool call written as text | recovery parser in the LLM wrapper |
| Premature stop / fake success | fluent answer, no ticket | completion check; derive status from tool results |
| Looping | same call repeated | `max_steps` safety stop; read the trace |
| Final answer | hallucinated numbers / IDs | `diagnose()` compares answer to tool evidence |

**Debugging workflow:** reproduce → read the trace layer by layer → find the *first* wrong event → script the model (`ScriptedLLM`) to make it deterministic → fix (prompt, tool description, validation or workflow rule) → **add a test case**.

## 💬 Q&A starters
1. Which checks would you move from the prompt into code, and why?
2. What would change if the model were 70B instead of 1.5B? What would *not* change?
3. How would you add a human-approval step before `create_return_ticket` for refunds > $100?
""")

md(r"""
## 🏋️ Practice exercises (after the session)

1. **New tool.** Add `check_exchange_stock(order_id)` and let customers ask for an *exchange* instead of a refund. Update the schemas, planner intents, `derive_status` and add 2 test cases.
2. **Human-in-the-loop.** Make `validate_tool_call` reject `create_return_ticket` when `refund_amount > 100` unless `state.approved` is True. Add a status `pending_approval`.
3. **Model comparison.** Set `MODEL = "qwen2.5:3b"` (or `llama3.2:3b`), pull it, and re-run `run_suite(TEST_CASES + NEW_CASES)`. Compare the pass rate per criterion.
4. **Robustness.** Run every test case 3 times and report the pass rate per case. Which cases are flaky? Why?
5. **Prompt ablation.** Delete rule 2 from `SYSTEM_PROMPT` and re-run the suite. Which criteria drop, and does the validator still protect the business outcome?

<details><summary>Hint for exercise 2</summary>

```python
# inside validate_tool_call, business preconditions section:
if name == "create_return_ticket" and float(args["refund_amount"]) > 100 and not getattr(state, "approved", False):
    errors.append("Refunds above $100 need human approval. Tell the customer a specialist will confirm within 24h.")
```
</details>
""")

nb = nbf.v4.new_notebook()
nb.cells = CELLS
nb.metadata = {
    "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
    "accelerator": "GPU",
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python"},
}
out = Path(__file__).resolve().parent.parent / "notebooks" / "LW1_S2_Local_Agentic_AI_Workflow.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print(f"wrote {out} ({len(CELLS)} cells)")
