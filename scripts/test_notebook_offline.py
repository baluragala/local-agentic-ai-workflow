"""Offline smoke test: executes the notebook without an Ollama server.

Cells tagged "server" are replaced by one cell that defines MODEL/OLLAMA_HOST
and a FakeClient that behaves like a well-behaved small model and returns real
ollama.ChatResponse objects. This verifies every orchestration / evaluation
code path; it does NOT measure real model quality.

Run:  python scripts/test_notebook_offline.py   (needs: ollama pydantic pandas nbformat nbclient ipykernel)
"""
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "notebooks" / "LW1_S2_Local_Agentic_AI_Workflow.ipynb"
OUT = ROOT / "reference_runs" / "offline_fake_model_run.ipynb"

FAKE = r'''
# === injected by test_notebook_offline.py: fake Ollama client ===
import json, re, requests
from ollama import ChatResponse, Message

class FakeClient:
    def _resp(self, content="", calls=None):
        tcs = [Message.ToolCall(function=Message.ToolCall.Function(name=n, arguments=a)) for n, a in (calls or [])]
        return ChatResponse(model="fake", message=Message(role="assistant", content=content, tool_calls=tcs or None),
                            prompt_eval_count=42, eval_count=17, total_duration=int(3e8), done=True)

    def chat(self, model=None, messages=None, tools=None, format=None, options=None, **kw):
        msgs = [m if isinstance(m, dict) else m.model_dump() for m in messages]
        user = next(m["content"] for m in msgs if m["role"] == "user")
        if isinstance(format, dict) and format.get("title") == "TicketTriage":
            return self._resp(json.dumps({"sentiment": "negative", "urgency": "high", "summary": "Repeated broken deliveries."}))
        if isinstance(format, dict) and format.get("title") == "Plan":
            # Like Ollama's schema grammar + a small model: optional fields are often skipped.
            plan = self._plan(user)
            req = set(format.get("required", plan))
            return self._resp(json.dumps({k: v for k, v in plan.items() if k in req}))
        names = [t["function"]["name"] for t in (tools or [])]
        tool_msgs = [m for m in msgs if m["role"] == "tool"]
        if "get_shipping_cost" in names:
            if not tool_msgs:
                return self._resp("", [("get_shipping_cost", {"weight_kg": 3, "express": True})])
            return self._resp(f"Express shipping for 3 kg costs ${json.loads(tool_msgs[-1]['content'])['cost_usd']}.")
        if "get_order" in names:
            return self._act(user, tool_msgs, msgs)
        return self._resp("Hello! Thanks for shopping with ShopEase.")

    def _plan(self, text):
        t = text.lower()
        m = re.search(r"(ORD-\d{4}|#\d{4})", text)
        reason = ("damaged" if re.search(r"damaged|cracked|broken", t) else
                  "defective" if re.search(r"defective|stopped working|faulty", t) else
                  "wrong_item" if "wrong" in t else
                  "changed_mind" if re.search(r"changed my mind|don't need|dont need|take it back", t) else None)
        if re.search(r"where is|status", t):
            intent = "order_status"
        elif re.search(r"return|refund|send it back|take it back", t):
            intent = "return_request"
        else:
            intent = "other"
        return {"intent": intent, "order_id": m.group(1) if m else None, "return_reason": reason,
                "steps": ["get_order", "check_return_eligibility", "calculate_refund", "create_return_ticket"]}

    lazy = False   # True = stall like qwen2.5:1.5b did: answer without a tool at the start and after get_order

    def _act(self, user, tool_msgs, msgs=None):
        last = (msgs or [{}])[-1]
        nudged = last.get("role") == "user" and str(last.get("content", "")).startswith("[Orchestrator]")
        if self.lazy and not nudged and len(tool_msgs) in (0, 1):
            return self._resp("I found your order. Please provide your order ID and we will proceed.")
        ctx = dict(re.findall(r"(\w+)=([^;\s]+)", user.split("[Planner context]")[-1]))
        res = {}
        for m in tool_msgs:
            res[m["tool_name"]] = json.loads(m["content"])
        oid, reason = ctx["order_id"], ctx["return_reason"]
        if "get_order" not in res:
            return self._resp("", [("get_order", {"order_id": oid})])
        order = res["get_order"]
        if "error" in order:
            return self._resp(f"Sorry, we could not find order {oid}. Please double-check the order ID.")
        if ctx["intent"] == "order_status":
            return self._resp(f"Your {order['item']} ({oid}) has {order['status']} and is expected on {order.get('expected_delivery')}.")
        if "check_return_eligibility" not in res:
            return self._resp("", [("check_return_eligibility", {"order_id": oid})])
        elig = res["check_return_eligibility"]
        if not elig.get("eligible"):
            return self._resp(f"I'm sorry, this item is not eligible for return: {elig['reason']}")
        if "calculate_refund" not in res:
            return self._resp("", [("calculate_refund", {"order_id": oid, "return_reason": reason})])
        if "create_return_ticket" not in res:
            amt = res["calculate_refund"]["refund_amount"]
            return self._resp("", [("create_return_ticket", {"order_id": oid, "return_reason": reason, "refund_amount": amt})])
        t = res["create_return_ticket"]
        return self._resp(f"Your return ticket {t['ticket_id']} is open and ${t['refund_amount']:.2f} will be refunded. A return label has been emailed.")

client = FakeClient()
'''

nb = nbf.read(SRC, as_version=4)
cfg = None
cells = []
injected = False
for c in nb.cells:
    if "server" in c.metadata.get("tags", []):
        if not injected:
            cells.append(nbf.v4.new_code_cell(FAKE))
            injected = True
        continue
    cells.append(c)
# Regression: a model that stops early twice must still be nudged through to a ticket.
cells.append(nbf.v4.new_code_cell(r"""
client.lazy = True
TICKETS.clear()
_s = run_agent("Hello, order ORD-1002: the headphones arrived damaged, the headband is cracked. Please refund me.")
client.lazy = False
assert _s.status == "ticket_created", (_s.status, _s.tools_used())
assert sum(e["event"] == "completion check" for e in _s.trace) == 2
print("REGRESSION lazy-model:", _s.status, _s.tools_used())
"""))
nb.cells = cells

client = NotebookClient(nb, timeout=120, kernel_name="python3", allow_errors=False,
                        resources={"metadata": {"path": str(ROOT)}})
try:
    client.execute()
finally:
    OUT.parent.mkdir(exist_ok=True)
    nbf.write(nb, OUT)
# Regression gate: the business outcome must hold for every test case.
import re as _re
outs = "".join(o.get("text", "") for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []))
scores = _re.findall(r"Business task success: (\d+)/(\d+)", outs)
assert scores and all(a == b for a, b in scores), f"business task success below 100%: {scores}"
print("business task success:", scores)
print(f"OK - executed {sum(c.cell_type == 'code' for c in nb.cells)} code cells; output saved to {OUT}")
