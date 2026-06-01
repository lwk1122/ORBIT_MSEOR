# ORBIT

Optimization and Operations Research Bridge for Intelligent Tooling

ORBIT 是一个数据库优先的 LLM + Operations Research 辅助工具，面向管理科学
与工程问题。它把私有文档、案例文本、表格和问题描述转化为可引用事实、结构
化决策记录、优化模型组件、可检查的 LP/MPS 工件、求解器证据和有边界的结论。

ORBIT 不再定位为开放式讨论平台。对于 MSE/OR topic，它使用由工件状态
驱动的建模工作流：每一轮尽量产出一个有用回答或工件；能确定推进的建模、校验
和求解步骤由确定性逻辑完成；当证据状态显示问题已经解决时提前停止。

## 核心能力

- 将私有 text、markdown 和 PDF layout block 摄取为持久化 document/chunk。
- 从语料 chunk、reviewed facts、claims、ledger、code evidence、API evidence
  和历史消息中检索，并保留来源溯源。
- 表达 MSE 概念：决策问题、备选方案、目标/KPI、约束、利益相关者、假设、干预、
  数据集、方法、效应量、不确定性、边界条件和管理启示。
- 抽取 OR 模型组件：集合、索引、参数、变量、目标函数、约束、单位、假设和数据需求。
- 在 LP/MPS 之前构建 solver-agnostic ORBIT Model IR，使建模意图可以独立于求解器
  语法被检查。
- 使用 provider-free 的确定性模板注册表处理高置信 OR 家族，包括 product mix、diet、
  transportation、network flow、cutting stock、workforce scheduling、inventory
  planning、assignment、TSP 诊断、robust/security LP、capital budgeting，以及小型
  MSE/NL4OPT 整数模型。
- 基于已审核组件生成并校验 LP/MPS，再选择性调用本地求解器。
- 在 fast 模式自动 review 之前执行确定性规格 gate；如果目标、变量、约束、formal
  text、来源引用或建模技巧要求缺失，系统会先指出阻塞规格，而不是猜测求解。
- 支持 pass@k 风格的 MiniMax 组件候选 tournament：先验证和排序多个候选，再持久化
  一个最佳建模路径。
- 记录 solver run、diagnostic、code evidence、claim support edge 和 provenance report。
- 支持离线评估：retrieval、grounding、citation、claim review、component extraction、
  solver correctness 和 latency。

## 工作流模式

`mse_workflow_mode=modeling_fast`

- 更积极地使用确定性推进。
- 自动 formalize 已审核组件，生成 LP/MPS，若可用则调用本地 solver，并创建
  solver-backed claim candidate。
- 自动 review 现在由确定性模型规格检查 gate 控制，不再单纯因为 fast 模式而跳过检查。
- 一旦存在 active 的 `optimization_result` claim，就提前关闭 topic。

`mse_workflow_mode=modeling_reviewed`

- 在模型工件变成 executable 之前保留显式 review gate。
- 同样由工件状态驱动，但更偏向可检查性和人工复核。

两个模式都会避免在问题已经由当前工件状态解决后继续消耗轮次。

## 证据标记

| Marker | 含义 | 证据等级 |
|--------|------|----------|
| `[D{id}]` | 私有语料 chunk | 来源文档证据 |
| `[F{id}]` | 已审核事实 | 证据 |
| `[C{id}]` | 有边界、受证据支持的 claim | 派生证据 |
| `[W{id}]` | Web search result | 未审核外部证据 |
| `[L{id}]` | Ledger entry | 结构化证据 |
| `[A{id}]` | Model/API consultation | 模型视角 |
| `[E{id}]` | Code 或 solver evidence | 可执行证据 |
| `[M{id}]` | 历史消息 | 仅作上下文 |

只有注入到 prompt 的 ID 才允许被引用；citation sanitizer 会剔除幻觉证据 ID。

## 主要模块

- `src/orbit_or/corpus.py`：私有语料摄取、chunking、表格捕获和 layout block 处理。
- `src/orbit_or/rag.py`：hybrid retrieval、RRF、reranking、neighbor expansion、context
  compression 和 retrieval notice。
- `src/orbit_or/domain_ontology.py`：通用 ontology profile 与 MSE profile。
- `src/orbit_or/domain_profiles.py`：MSE 角色与建模 prompt profile。
- `src/orbit_or/problem_profile.py`：面向 evaluation、batch routing 和服务端 problem
  seeding 的文本驱动 OR family profiling。
- `src/orbit_or/template_solvers.py`：保守的 provider-free 确定性求解器，覆盖高置信
  ORQ/NL4OPT/IndustryOR/MAMO 问题家族。
- `src/orbit_or/optimization.py`：OR component payload、LP/MPS 校验、solver dispatch、
  repair gate 和 failure label。
- `src/orbit_or/minimax_client.py`：MiniMax provider adapter。当前 workspace 的 provider
  路径保持 MiniMax-only，不提供 LM Studio fallback。
- `src/orbit_or/formal_claims.py`：确定性 formal claim review。
- `src/orbit_or/evaluation.py`：离线评估指标与 fixture loader。
- `src/orbit_or/web.py`：monitor/review API、MSE review 和 provenance export。

## 本地运行

```bash
uv sync
uv run python -c "from orbit_or.db import init_db; init_db()"
uv run python -m orbit_or.server
```

另开一个 shell 启动 web monitor：

```bash
ORBIT_WEB_PORT=8080 uv run python -m orbit_or.web
```

无需 provider 的检查：

```bash
uv run python scripts/evaluate_or_mse_fixtures.py
uv run pytest tests/test_template_solvers.py -q
uv run pytest
```

ORQ 迭代细节见本地 workspace 中的 `outputs/orq_dataset_analysis/task_plan.md` 和
`outputs/orq_dataset_analysis/notes.md`。最近一次完整验证结果为
`1186 passed, 1 failed, 1 warning`；唯一失败是已知环境路径问题：
`tests/test_web.py::test_dashboard_snapshot_handles_empty_database` 期望 `test_orbit.db`，
当前环境返回 `orbit.db`。

如未来确需接入非 MiniMax adapter，先看
[模型适配要求](docs/model_adapter_requirements.md)。当前 workspace 约束下，provider
执行保持 MiniMax-only。

## 配置

- `ORBIT_DB_PATH`：覆盖 SQLite database path。
- `ORBIT_WEB_HOST`, `ORBIT_WEB_PORT`：web monitor 绑定地址。
- `ORBIT_OUTPUT_TOKENS`：默认 LLM 输出 token budget。
- `MINIMAX_API_KEY` 等 provider key 对离线测试和确定性 fixture evaluation 不是必需的。
- 当前 workspace 不配置 LM Studio fallback；确定性测试和 ORQ fixture scan 必须在没有
  provider access 时优雅降级。

## 数据与溯源

默认数据库为 `orbit.db`，测试数据库为 `test_orbit.db`。ORBIT 会存储 topic、message、
corpus document/chunk、reviewed fact、claim、ledger、optimization problem/component/
artifact、solver run、diagnostic 和 evidence link。MSE provenance export 会展示从语料
来源到模型组件、工件、solver run、claim 和 support edge 的完整链条。

本地 OR + LLM fixture 数据集放在 `docs/ORQ_Dataset/`，用于私有设计研究和离线评估，
不需要 vendoring 外部 reference repository。
如需用 MiniMax 逐题运行 NL4OPT 并归档，可执行
`PYTHONPATH=src MINIMAX_MAX_CONCURRENT=1 all_proxy= ALL_PROXY= uv run python scripts/run_nl4opt_minimax_batch.py --start 1 --limit 10 --k 2`；
逐题 archive、单题 SQLite 数据库、JSONL manifest 和 `predictions.json` 会写入已被
git 忽略的 `outputs/nl4opt_minimax/`。
