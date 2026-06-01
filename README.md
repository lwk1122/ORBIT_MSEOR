# ORBIT

Optimization and Operations Research Bridge for Intelligent Tooling

[中文说明](README_CN.md)

ORBIT is a database-first LLM + Operations Research assistant for management
science and engineering work. It helps turn private documents, case text,
tables, and problem statements into cited facts, structured decision records,
optimization model components, inspectable LP/MPS artifacts, solver evidence,
and scoped claims.

ORBIT is no longer positioned as an open-ended discussion platform. For MSE/OR
topics, it uses artifact-state-driven modeling workflows: each round should
produce one useful answer or artifact, deterministic model/solver steps advance
when possible, and the workflow stops once the evidence state indicates that
the problem is solved.

## What It Does

- Ingests private text, markdown, and PDF-derived layout blocks into durable
  document and chunk tables.
- Retrieves corpus chunks, reviewed facts, claims, ledger entries, code
  evidence, API evidence, and prior messages with source provenance.
- Represents MSE concepts such as decision problems, objectives, KPIs,
  constraints, stakeholders, assumptions, interventions, datasets, methods,
  effect sizes, uncertainty, boundary conditions, and managerial implications.
- Extracts OR model components: sets, indices, parameters, variables,
  objectives, constraints, units, assumptions, and data requirements.
- Builds a solver-agnostic ORBIT model IR before compiling solver artifacts, so
  model intent can be inspected independently from LP/MPS syntax.
- Uses a provider-free deterministic template registry for high-confidence OR
  families such as product mix, diet, transportation, network flow, cutting
  stock, workforce scheduling, inventory planning, assignment, TSP diagnostics,
  robust/security LPs, capital budgeting, and small MSE/NL4OPT integer models.
- Generates and validates LP/MPS artifacts from reviewed components before
  optional solver execution.
- Applies deterministic specification gates before fast-mode auto-review; if
  objective, variables, constraints, formal text, source refs, or modeling
  technique requirements are missing, the workflow asks for the blocking
  specification instead of guessing.
- Supports pass@k-style MiniMax component candidate tournaments that validate
  and rank candidates before persisting one best modeling path.
- Records solver runs, diagnostics, code evidence, claim support edges, and
  provenance reports.
- Runs offline evaluation for retrieval, grounding, citation quality, claim
  review, component extraction, solver correctness, and latency.

## Workflow Modes

`mse_workflow_mode=modeling_fast`

- Uses deterministic advancement aggressively.
- Formalizes reviewed components, generates LP/MPS artifacts, solves when a
  local backend is available, and creates solver-backed claim candidates.
- Auto-review is gated by deterministic model-specification checks rather than
  by speed alone.
- Stops early when an active `optimization_result` claim is available.

`mse_workflow_mode=modeling_reviewed`

- Keeps explicit review gates before model artifacts become executable.
- Uses the same artifact-state-driven progression, but favors inspectability
  and human review over speed.

Both modes avoid spending rounds on generic discussion when the current artifact
state already answers the task.

## Evidence Markers

| Marker | Meaning | Evidence Grade |
|--------|---------|---------------|
| `[D{id}]` | Private corpus chunk | Source-document evidence |
| `[F{id}]` | Reviewed fact | Evidence |
| `[C{id}]` | Scoped claim supported by evidence | Derived evidence |
| `[W{id}]` | Web search result | Unreviewed external evidence |
| `[L{id}]` | Ledger entry | Structured evidence |
| `[A{id}]` | Model/API consultation | Model perspective |
| `[E{id}]` | Code or solver evidence | Executable evidence |
| `[M{id}]` | Prior message | Context only |

Only IDs injected into the prompt may be cited. The sanitizer strips
hallucinated evidence IDs while allowing prior-message attribution.

## Core Modules

- `src/orbit_or/corpus.py`: private corpus ingestion, chunking, table capture,
  and layout-block handling.
- `src/orbit_or/rag.py`: hybrid retrieval, RRF fusion, reranking, neighbor
  expansion, context compression, and retrieval notices.
- `src/orbit_or/domain_ontology.py`: generic ontology profiles plus the MSE
  profile.
- `src/orbit_or/domain_profiles.py`: MSE role and modeling prompt profile.
- `src/orbit_or/problem_profile.py`: text-driven OR family profiling used by
  evaluation, batch routing, and service problem seeding.
- `src/orbit_or/template_solvers.py`: conservative provider-free deterministic
  solvers for high-confidence ORQ/NL4OPT/IndustryOR/MAMO problem families.
- `src/orbit_or/optimization.py`: OR component payloads, LP/MPS validation,
  solver dispatch, repair checks, and failure labels.
- `src/orbit_or/minimax_client.py`: MiniMax provider adapter. Current provider
  work in this workspace is MiniMax-only; there is no LM Studio fallback path.
- `src/orbit_or/formal_claims.py`: deterministic formal claim review.
- `src/orbit_or/evaluation.py`: offline evaluation metrics and fixture loader.
- `src/orbit_or/web.py`: monitor and review endpoints, including MSE review and
  provenance exports.

## Run Locally

```bash
uv sync
uv run python -c "from orbit_or.db import init_db; init_db()"
uv run python -m orbit_or.server
```

Start the web monitor in another shell:

```bash
ORBIT_WEB_PORT=8080 uv run python -m orbit_or.web
```

Useful provider-free checks:

```bash
uv run python scripts/evaluate_or_mse_fixtures.py
uv run pytest tests/test_template_solvers.py -q
uv run pytest
```

For ORQ iteration details, see `outputs/orq_dataset_analysis/task_plan.md` and
`outputs/orq_dataset_analysis/notes.md` in the local workspace. The latest full
verification observed `1186 passed, 1 failed, 1 warning`; the single failure is
the known environment-path issue in
`tests/test_web.py::test_dashboard_snapshot_handles_empty_database`, where the
test expects `test_orbit.db` but the environment returns `orbit.db`.

See [model adapter requirements](docs/model_adapter_requirements.md) for the
capabilities any future non-MiniMax adapter must support. Under the current
workspace constraint, provider execution remains MiniMax-only.

## Configuration

- `ORBIT_DB_PATH`: override the SQLite database path.
- `ORBIT_WEB_HOST`, `ORBIT_WEB_PORT`: web monitor binding.
- `ORBIT_OUTPUT_TOKENS`: default LLM output token budget.
- Provider keys such as `MINIMAX_API_KEY` remain optional for offline tests and
  deterministic fixture evaluation.
- Do not configure LM Studio as a fallback in this workspace; deterministic
  tests and ORQ fixture scans must degrade without provider access.

## Data And Provenance

The default database is `orbit.db`; tests use `test_orbit.db`. ORBIT stores
topics, messages, corpus documents/chunks, reviewed facts, claims, ledger
records, optimization problems/components/artifacts, solver runs, diagnostics,
and evidence links. The MSE provenance export surfaces the chain from corpus
sources through model components, artifacts, solver runs, claims, and support
edges.

Local OR + LLM fixture datasets live under `docs/ORQ_Dataset/` and are used as
private design/evaluation references without vendoring external repositories.
To run NL4OPT through MiniMax with per-case archives, use
`PYTHONPATH=src MINIMAX_MAX_CONCURRENT=1 all_proxy= ALL_PROXY= uv run python scripts/run_nl4opt_minimax_batch.py --start 1 --limit 10 --k 2`;
archives, per-case SQLite databases, a JSONL manifest, and `predictions.json`
are written under ignored `outputs/nl4opt_minimax/`.
