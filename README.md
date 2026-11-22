# LW1 · S2 — Building a Local Agentic AI Workflow

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/baluragala/local-agentic-ai-workflow/blob/main/notebooks/LW1_S2_Local_Agentic_AI_Workflow.ipynb)

A 3-hour, live-coding demo notebook. It builds a transparent agentic workflow with a small open-weight model served locally by **Ollama**: plain Python, no framework, no API keys.

**Scenario:** *ShopEase Returns Desk.* A customer message comes in, and the agent plans, calls tools (`get_order`, `check_return_eligibility`, `calculate_refund`, `create_return_ticket`) and replies. The result is a ticket, a policy decline, an order-status answer, or a request for missing information.

**Model:** `qwen2.5:1.5b` (~1 GB, supports native tool calling and JSON-schema structured outputs). You can swap it for `qwen2.5:3b` or `llama3.2:3b` by changing `MODEL`.

## Files

| Path | Purpose |
|---|---|
| `notebooks/LW1_S2_Local_Agentic_AI_Workflow.ipynb` | **The session notebook** (Sections 1–6, Colab-ready) |
| `scripts/build_notebook.py` | Generates the notebook. Edit this, then re-run it |
| `scripts/test_notebook_offline.py` | Executes the notebook with a fake model client (no Ollama needed) to smoke-test all code paths |
| `reference_runs/offline_fake_model_run.ipynb` | Output of that offline run. Shows the *intended* traces; it is **not** real model output |

## Running

**Google Colab:** upload the notebook (or open it from GitHub/Drive), select `Runtime → Change runtime type → T4 GPU` (CPU works but is slower), and run top to bottom. Section 2 installs Ollama, starts `ollama serve` in the background and pulls the model (~1–2 min).

**Local:** install Ollama from https://ollama.com/download and start it, then `pip install ollama pydantic pandas requests` and open the notebook in Jupyter. The setup cells detect the running server and skip the install.

## Section map (matches the session agenda)

| § | Time | What is built / shown |
|---|---|---|
| 1 | 20m | Business task, tools, why agentic, task decomposition, loop diagram, roles (model vs orchestrator) |
| 2 | 25m | Install/start Ollama, pull model, raw HTTP call, Python client, structured output, first tool-call round trip, benefits/limits |
| 3 | 55m | Spec → mock data → tools (tested without LLM) → schemas → validator/executor → `AgentState` → LLM wrapper → planner + missing-info gate → one step by hand → full loop → traced run (layer-tagged) |
| 4 | 40m | Six scenarios (status, multi-tool, missing info, outside window, fee, non-returnable), unknown order, `ScriptedLLM` failure demos (hallucinated tool, bad args, skipped step, wrong amount, fake success), `diagnose()` |
| 5 | 30m | 7 criteria with validity rules, 8 test cases, `run_suite`, per-criterion pass rates, failure drill-down, response vs workflow vs business evaluation, extending the dataset (+3 cases) |
| 6 | 10m | Recap, failure-point/debugging table, Q&A prompts, 5 practice exercises |

## Instructor notes

* **Pre-session:** run the whole notebook once on Colab with the chosen model. The first model load is slow, and you want the model cached.
* Small-model runs vary. If a Section 4/5 case fails live, **use it**: open the trace, run `diagnose(state)`, and name the failing layer. Section 5.4 is built for exactly that.
* Failure demos in §4.2 use `ScriptedLLM`, so they are **deterministic** and do not depend on the model misbehaving on cue.
* `TODAY` is frozen to 2026-03-15 so that the policy outcomes are identical whenever the notebook is run.

## Regenerate / re-test

```bash
python scripts/build_notebook.py
python scripts/test_notebook_offline.py   # needs: ollama pydantic pandas nbformat nbclient ipykernel
```

## Slides

`slides/` holds the source of the session deck (26 slides, speaker notes in each slide's `<aside>`):
`deck.json` (slide order and sections), `deck.css` (theme) and one HTML file per slide in `slides/slides/`.
**Present it from GitHub Pages:** https://baluragala.github.io/local-agentic-ai-workflow/slides/
(`slides/index.html` is a small player: ←/→ or click to move, `F` full screen, `N` speaker notes, `#12` in the URL jumps to slide 12).
To run it locally: `cd slides && python3 -m http.server`, then open http://localhost:8000.

The same deck as a Claude artifact (can export to PPTX/PDF from its Share menu): https://claude.ai/artifact/PkMst8sWiiw6Vyp9693zFi
