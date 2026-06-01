# AGENTS.md

## Project Direction

ORBIT is a database-first LLM + Operations Research assistant for management
science and engineering. It is no longer positioned as a generic discussion platform.
The product should help users move from private documents and problem
statements to grounded answers, decision records, inspectable optimization
models, solver evidence, and provenance-rich reports.

Local OR + LLM reference repositories may be cloned under `docs/ref/` for
private design study. That directory is ignored by git. Persistent takeaways
belong in `docs/or_llm_reference_analysis.md` and this file.

## North Star

The system should answer complex MSE/OR questions from a private corpus while
preserving evidence provenance, decision assumptions, numerical data,
conflicting sources, claim validity boundaries, and solver/model diagnostics.

A domain-ready ORBIT system should:

- Ingest private documents into durable document/chunk indexes.
- Retrieve from external corpus chunks and internal reviewed memory.
- Convert evidence into reviewed facts, structured ledger records, formal
  claims, optimization components, model artifacts, and solver evidence.
- Surface contradictions, stale sources, ambiguous data, and invalid model
  assumptions instead of silently hiding them.
- Let modeling roles reason from citations and structured data rather than
  unsupported model memory.
- Provide repeatable evaluation hooks for retrieval, grounding, claim quality,
  component extraction, solver correctness, and end-to-end usefulness.

## Product Model

Management science and engineering topics should prefer:

- `mse_workflow_mode=modeling_fast`
- `mse_workflow_mode=modeling_reviewed`

These modes replace open-ended rounds with artifact-state-driven modeling
rounds. Each round should produce one useful answer or artifact. If deterministic
logic can formalize components, generate LP/MPS, run a solver, create
solver-backed claims, or close the task, it should do so without spending
additional provider calls. When the current artifact/evidence state solves the
problem, stop early.

## Domain Role Profile

Use compact, non-duplicative MSE/OR role perspectives:

- OR Modeler: objectives, constraints, variables, algorithms, and formulations.
- Empirical Analyst: data validity, metrics, identification, uncertainty.
- Systems Engineer: deployability, reliability, workflow, cost, latency.
- Decision Scientist: trade-offs, stakeholder criteria, and decision quality.
- Skeptic: external validity, hidden assumptions, failure modes.

Roles are topic/domain configuration, not hard-coded replacements for all
non-MSE use.

## Architecture Priorities

### Private Corpus Layer

- `CorpusDocument`: source metadata, title, author, doc type, timestamps,
  source path or URL, checksum, access scope, parser version, index status.
- `CorpusChunk`: hierarchical chunks with parent document, section path,
  granularity, neighboring chunk IDs, text, table markdown, page/position,
  embedding, lexical index text, and freshness metadata.
- `CorpusIngestRun`: parser/indexer metadata, errors, counts, timings, and
  model versions.
- APIs for ingesting, listing, reindexing, and retrieving corpus records.

### Retrieval

- Dense retrieval plus BM25/FTS retrieval.
- Reciprocal Rank Fusion for dense/lexical candidates.
- Cross-encoder reranking when available.
- Parent/neighbor context expansion.
- Prompt-budget-aware context compression.
- Freshness, contradiction, and version notices.
- Confidence gates before answer/modeling injection.

### MSE Ontology

The structured layer should represent decision problems, alternatives,
objectives/KPIs, constraints, stakeholders, assumptions, interventions,
datasets/samples, optimization/simulation/forecasting/causal methods, effect
sizes, uncertainty, boundary conditions, and managerial implications.

### Formal Claim Pipeline

- Analyst, Scientist, Engineer, and MSE/OR roles may submit structured formal
  claim candidates.
- Clerk extraction is a fallback sweep, not the primary source of all claims.
- Review rejects fact restatements, vague claims, missing scope, missing
  uncertainty, circular evidence-gap claims, and unsupported generalization.
- Similar claims should be deduplicated as duplicate, qualifies, subsumes,
  conflicts_with, or refutes.
- JTMS/VIKI should update claim status when supporting evidence is superseded,
  contested, refuted, or solver evidence becomes stale.

### OR Modeling And Solver Evidence

- Extract sets, indices, parameters, decision variables, objectives,
  constraints, units, assumptions, and data requirements.
- Review components before using them as model evidence.
- Generate safe LP/MPS artifacts before arbitrary Python code.
- Run optional solver backends through existing sandbox/evidence patterns.
- Classify failures: missing data, ambiguous variable, unit mismatch, invalid
  artifact syntax, infeasible, unbounded, timeout, runtime error, wrong optimal
  value, and no-solution calibration error.
- Store solver outputs as code/solver evidence linked to facts, ledger rows,
  model components, and claims.
- Use ORQA/NL4OPT/Mamo/ORLM-style local fixtures without vendoring reference
  repositories.
- Prefer provider-free deterministic templates for high-confidence benchmark
  families when all coefficients, domains, bounds, and side constraints are
  explicit in the problem text.
- Keep provider execution MiniMax-only in this workspace. Do not add or invoke
  LM Studio fallback logic.

### Evaluation And Observability

Track chunk boundary quality, table fidelity, retrieval Recall@K/MRR/NDCG,
reranker lift, citation accuracy, answer faithfulness, no-answer calibration,
contradiction detection, claim acceptance/rejection reasons, ledger extraction
accuracy, component extraction accuracy, solver correctness, and per-stage
latency. Evaluation must run from scripts/tests without live provider calls
unless explicitly requested.

## Engineering Rules

- Keep changes compatible with existing topics and databases where feasible.
- Prefer additive migrations through `init_db()` and nullable columns.
- Keep retrieval topic-scoped unless corpus sharing is explicitly configured.
- Do not allow pending candidates into ordinary RAG.
- Preserve citation ID sanitization and only allow injected IDs to be cited.
- Avoid provider-specific logic outside broker/config boundaries.
- Keep long-running provider calls optional in tests.
- Treat `docs/model_adapter_requirements.md` as the contract for any future
  local Qwen or other non-MiniMax adapter.
- Add focused tests for every schema, retrieval, citation, review, and solver
  behavior.
- Keep the base conversation machinery available as compatibility plumbing,
  but do not describe the product as a generic discussion platform.

## Implemented Foundation

- Private corpus tables: `CorpusDocument`, `CorpusChunk`, and
  `CorpusIngestRun`.
- Text, markdown, `LayoutBlock`, and optional PDF/table ingestion in
  `orbit_or.corpus`.
- Corpus listing, retrieval, neighbor lookup, lexical reindexing, and
  provider-free ingest APIs.
- Corpus lexical retrieval plus optional embeddings fused through Reciprocal
  Rank Fusion.
- Neighbor expansion, stale/version/conflict notices, confidence gates, context
  compression, and `[D...]` citation support in RAG.
- `domain_profile=mse` and `mse_workflow_mode=modeling_fast|modeling_reviewed`
  for artifact-state-driven MSE topics.
- Generic ontology profile module in `orbit_or.domain_ontology`, including the
  MSE ontology.
- Optimization tables: `OptimizationProblem`, `OptimizationComponent`,
  `OptimizationArtifact`, `SolverRun`, and `ModelDiagnostic`.
- Deterministic OR helpers in `orbit_or.optimization` for component payloads,
  LP/MPS artifacts, solver dispatch, repair checks, and failure labels.
- LP/MPS parsing, validation, repair candidate persistence, semantic gates, and
  optional local `scipy.optimize.linprog` / `scipy.optimize.milp` execution.
- Topic-level MSE review snapshot and provenance export APIs/web views.
- Solver-backed `optimization_result` claim candidates, deterministic review,
  support edges, JTMS propagation, and VIKI repair handlers.
- Offline metrics in `orbit_or.evaluation` and the local mixed gold fixture at
  `tests/fixtures/or_mse_gold.json`.
- Offline fixture scoring command: `scripts/evaluate_or_mse_fixtures.py`.
- Text-driven OR family profiling in `orbit_or.problem_profile`, reused by
  evaluation, batch routing, and MSE service problem seeding.
- Provider-free deterministic template registry in `orbit_or.template_solvers`
  covering high-confidence ORQ/NL4OPT/IndustryOR/MAMO families including
  product mix, transportation, diet, network flow, cutting stock, workforce
  scheduling, inventory planning, assignment, TSP diagnostics, robust/security
  LPs, capital budgeting, and small MSE integer allocation models.
- Local ORQ provider-free tracking under `outputs/orq_dataset_analysis/`.
  Phase 72 scan status: IndustryOR 87/100 correct, NL4OPT 75/245 correct, and
  MAMO 169/863 correct with 18 retained matched diagnostics.

## Near-Term Optional Extensions

- Additional optional local solver adapters such as HiGHS, PuLP, OR-Tools, MIP,
  or PySCIPOpt when dependencies are installed.
- Backend-specific solver fixtures for any newly available local solver.
- Small API/web improvements that expose existing corpus, component, diagnostic,
  claim, and provenance state before adding large UI surfaces.

## Definition Of Done

A gap is not closed until:

- Schema/API behavior is implemented.
- Existing tests still pass.
- New behavior has focused tests.
- README or docs explain how to use it.
- The feature degrades gracefully without live LLM/provider access.
- A future agent can understand the purpose from this file and nearby docs.
