"""Tests for the code sandbox integration: validation, DB CRUD, server helpers, RAG rendering."""

import json
import os
from pathlib import Path

import pytest

import orbit_or.code_sandbox as code_sandbox
from orbit_or.code_sandbox import (
    CODE_FOLLOWUP_TURN,
    ENVIRONMENT_DESCRIPTION_PATH,
    FULL_TIMEOUT,
    PhaseRunResult,
    ROLE_CODE_HINTS,
    SMOKE_TIMEOUT,
    ExperimentResult,
    _classify_failure,
    _render_code_evidence,
    _try_parse_grid_stdout,
    run_code_evidence,
    run_code_evidence_grid,
    validate_calc_expression,
    validate_source,
)
from orbit_or.db import (
    get_db,
    get_db_path,
    init_db,
    insert_code_evidence,
    get_code_evidence_for_topic,
    get_code_evidence_for_topic_full,
    get_code_evidence_by_id,
)
from orbit_or.rag import _render_code_evidence_section
from orbit_or.server import (
    BASE_TURN,
    ANALYSIS_PHASE,
    EVIDENCE_PHASE,
    OPENING_PHASE,
    _extract_calc_requests,
    _extract_code_review_request,
    _extract_code_verify_request,
    _extract_code_verify_requests,
    _get_code_tier,
    should_enable_code_exec,
)


@pytest.fixture(autouse=True)
def setup_teardown():
    os.environ["TESTING"] = "1"
    db_path = get_db_path()
    if os.path.exists(db_path):
        os.remove(db_path)
    init_db()
    yield
    if os.path.exists(db_path):
        os.remove(db_path)


# ---------------------------------------------------------------------------
# validate_source tests
# ---------------------------------------------------------------------------


class TestValidateSource:
    def test_valid_simple_script(self):
        code = "import numpy as np\nprint(np.mean([1,2,3]))"
        ok, reason = validate_source(code)
        assert ok, reason

    def test_valid_multi_import(self):
        code = "import numpy, pandas\nimport math\nprint(42)"
        ok, reason = validate_source(code)
        assert ok, reason

    def test_valid_from_import(self):
        code = "from scipy import stats\nprint(stats.norm.pdf(0))"
        ok, reason = validate_source(code)
        assert ok, reason

    def test_valid_re_compile(self):
        """re.compile() should be allowed since re is in ALLOWED_IMPORTS."""
        code = (
            "import re\npattern = re.compile(r'\\d+')\nprint(pattern.findall('abc123'))"
        )
        ok, reason = validate_source(code)
        assert ok, reason

    def test_reject_empty(self):
        ok, _ = validate_source("")
        assert not ok

    def test_reject_trivially_short(self):
        ok, _ = validate_source("x=1")
        assert not ok

    def test_reject_blocked_import(self):
        ok, reason = validate_source(
            "import http\nhttp.client.HTTPConnection('example.com')"
        )
        assert not ok
        assert "allowlist" in reason

    def test_reject_os_any_attribute(self):
        """os.* should be blocked broadly, not just os.system."""
        ok, reason = validate_source("import os\nprint(os.listdir('.'))")
        assert not ok

    def test_reject_eval(self):
        ok, _ = validate_source("x = eval('1+1')\nprint(x)")
        assert not ok

    def test_reject_exec(self):
        ok, _ = validate_source("exec('print(42)')\npass\nmore stuff")
        assert not ok

    def test_reject_getattr(self):
        ok, _ = validate_source("import sys\ngetattr(sys, 'path')\nprint('done')")
        assert not ok

    def test_reject_open(self):
        ok, _ = validate_source("f = open('/etc/passwd')\nprint(f.read())\nf.close()")
        assert not ok

    def test_reject_subprocess(self):
        ok, _ = validate_source(
            "import subprocess\nsubprocess.run(['ls'])\nprint('ok')"
        )
        assert not ok

    def test_reject_import_os(self):
        ok, _ = validate_source("import os\nprint(os.getcwd())")
        assert not ok

    def test_reject_builtins_access(self):
        """__builtins__ should be blocked."""
        ok, _ = validate_source("x = __builtins__\nprint(type(x))\nmore code")
        assert not ok

    def test_reject_builtins_module(self):
        ok, _ = validate_source("import builtins\nbuiltins.eval('1')\nstuff")
        assert not ok

    def test_reject_sys_modules(self):
        """sys.modules escape should be blocked."""
        ok, _ = validate_source("import sys\nos = sys.modules['os']\nos.system('id')")
        assert not ok

    def test_reject_globals(self):
        ok, _ = validate_source("g = globals()\nprint(g['__builtins__'])\nmore")
        assert not ok

    def test_reject_vars(self):
        ok, _ = validate_source("v = vars()\nprint(v)\nextra line here")
        assert not ok

    def test_reject_line_limit(self):
        code = "\n".join(f"x_{i} = {i}" for i in range(501))
        ok, reason = validate_source(code)
        assert not ok
        assert "line limit" in reason

    def test_semicolon_separated_import(self):
        ok, _ = validate_source("x = 1; import http\nprint(x)")
        assert not ok

    def test_comment_not_checked_for_import(self):
        code = "# import http\nprint('hello world')"
        ok, reason = validate_source(code)
        assert ok, reason

    def test_reject_pickle(self):
        ok, _ = validate_source("import pickle\npickle.loads(b'')\nprint('done')")
        assert not ok

    def test_reject_ctypes(self):
        ok, _ = validate_source("import ctypes\nctypes.CDLL('libc.so.6')\nprint('x')")
        assert not ok

    # --- Class hierarchy / introspection escape vectors ---

    def test_reject_subclasses(self):
        code = "for c in ().__class__.__bases__[0].__subclasses__():\n    print(c)\n# padding"
        ok, _ = validate_source(code)
        assert not ok

    def test_reject_bases(self):
        code = "x = ().__class__.__bases__\nprint(x)\nextra line here"
        ok, _ = validate_source(code)
        assert not ok

    def test_reject_mro(self):
        code = "x = int.__mro__\nprint(x)\nsome more code"
        ok, _ = validate_source(code)
        assert not ok

    def test_reject_globals_dunder(self):
        code = "def f(): pass\nprint(f.__globals__)\nextra padding"
        ok, _ = validate_source(code)
        assert not ok

    def test_reject_code_dunder(self):
        code = "def f(): pass\nprint(f.__code__.co_consts)\nmore code"
        ok, _ = validate_source(code)
        assert not ok

    def test_reject_class_dunder(self):
        code = "x = (1).__class__\nprint(x)\npadding here"
        ok, _ = validate_source(code)
        assert not ok

    def test_reject_sys_path(self):
        code = "import sys\nsys.path.insert(0, '/tmp')\nprint('hi')"
        ok, _ = validate_source(code)
        assert not ok

    def test_reject_getframe(self):
        code = "import sys\nf = sys._getframe()\nprint(f)\nextra"
        ok, _ = validate_source(code)
        assert not ok

    def test_reject_f_globals(self):
        code = "import sys\nf = sys._getframe()\nprint(f.f_globals)\nx=1"
        ok, _ = validate_source(code)
        assert not ok

    def test_reject_traceback(self):
        code = "try:\n    1/0\nexcept Exception as e:\n    print(e.__traceback__)"
        ok, _ = validate_source(code)
        assert not ok

    def test_allow_model_eval_method(self):
        code = (
            "import torch\n"
            "import torch.nn as nn\n"
            "model = nn.Linear(2, 1)\n"
            "model.eval()\n"
            "print('ok')\n"
        )
        ok, reason = validate_source(code)
        assert ok, reason


# ---------------------------------------------------------------------------
# _extract_code_verify_request tests
# ---------------------------------------------------------------------------


class TestExtractCodeVerifyRequest:
    def test_simple(self):
        content = (
            "I suggest we verify this. [CODE_VERIFY: The mean of X is 42] Let's see."
        )
        result = _extract_code_verify_request(content)
        assert result == "The mean of X is 42"

    def test_with_whitespace(self):
        content = "[CODE_VERIFY:   spaced hypothesis   ]"
        result = _extract_code_verify_request(content)
        assert result == "spaced hypothesis"

    def test_no_match(self):
        content = "No code verify here."
        result = _extract_code_verify_request(content)
        assert result is None

    def test_hypothesis_without_nested_brackets(self):
        """Hypothesis text should not contain brackets (LLM convention)."""
        content = "[CODE_VERIFY: Compare model A vs model B]"
        result = _extract_code_verify_request(content)
        assert result == "Compare model A vs model B"

    def test_multiple_matches_returns_first_only(self):
        """Each marker is independent — first is extracted, second is ignored."""
        content = "[CODE_VERIFY: first] and [CODE_VERIFY: second]"
        result = _extract_code_verify_request(content)
        assert result == "first"

    def test_empty_hypothesis(self):
        content = "[CODE_VERIFY: ]"
        result = _extract_code_verify_request(content)
        # .+ requires at least one char
        assert result is None or result.strip() == ""


# ---------------------------------------------------------------------------
# should_enable_code_exec tests
# ---------------------------------------------------------------------------


class TestShouldEnableCodeExec:
    def _state(self, phase, round_number=2):
        return {"phase": phase, "round_number": round_number}

    def test_scientist_evidence_phase(self):
        assert should_enable_code_exec(
            self._state(EVIDENCE_PHASE), "scientist", BASE_TURN
        )

    def test_engineer_analysis_phase(self):
        assert should_enable_code_exec(self._state(ANALYSIS_PHASE), "engineer", BASE_TURN)

    def test_analyst_evidence_phase(self):
        assert should_enable_code_exec(
            self._state(EVIDENCE_PHASE), "analyst", BASE_TURN
        )

    def test_contrarian_analysis_phase(self):
        assert should_enable_code_exec(
            self._state(ANALYSIS_PHASE), "contrarian", BASE_TURN
        )

    def test_dreamer_rejected(self):
        assert not should_enable_code_exec(
            self._state(EVIDENCE_PHASE), "dreamer", BASE_TURN
        )

    def test_critic_now_has_review_tier(self):
        """Critic now has code review tier, so should_enable_code_exec returns True."""
        assert should_enable_code_exec(self._state(ANALYSIS_PHASE), "critic", BASE_TURN)

    def test_opening_phase_rejected(self):
        assert not should_enable_code_exec(
            self._state(OPENING_PHASE, 1), "scientist", BASE_TURN
        )

    def test_non_base_turn_rejected(self):
        assert not should_enable_code_exec(
            self._state(EVIDENCE_PHASE), "scientist", "tron_remediation"
        )

    def test_cat_rejected(self):
        assert not should_enable_code_exec(self._state(ANALYSIS_PHASE), "cat", BASE_TURN)

    def test_skynet_rejected(self):
        assert not should_enable_code_exec(
            self._state(ANALYSIS_PHASE), "skynet", BASE_TURN
        )


# ---------------------------------------------------------------------------
# DB CRUD tests
# ---------------------------------------------------------------------------


class TestCodeEvidenceDB:
    def test_insert_and_get_by_id(self):
        # Need a topic first
        with get_db() as conn:
            cursor = conn.execute(
                "INSERT INTO Topic (summary, detail, status) VALUES (?, ?, 'Started')",
                ("test topic", "detail"),
            )
            topic_id = cursor.lastrowid

        eid = insert_code_evidence(
            origin_topic_id=topic_id,
            origin_subtopic_id=None,
            hypothesis="Mean of [1,2,3] is 2.0",
            source_code="import numpy as np\nprint(np.mean([1,2,3]))",
            stdout="2.0\n",
            stderr="",
            exit_code=0,
            execution_time_s=0.5,
            iterations=1,
            success=True,
            requesting_role="scientist",
            summary="Code verified mean is 2.0",
        )
        assert eid > 0

        row = get_code_evidence_by_id(eid)
        assert row is not None
        assert row["hypothesis"] == "Mean of [1,2,3] is 2.0"
        assert row["exit_code"] == 0
        assert row["success"] == 1  # SQLite stores bool as int
        assert row["requesting_role"] == "scientist"
        assert "numpy" in row["source_code"]

    def test_get_for_topic_excludes_source_code(self):
        with get_db() as conn:
            cursor = conn.execute(
                "INSERT INTO Topic (summary, detail, status) VALUES (?, ?, 'Started')",
                ("t", "d"),
            )
            topic_id = cursor.lastrowid

        insert_code_evidence(
            origin_topic_id=topic_id,
            origin_subtopic_id=None,
            hypothesis="test",
            source_code="print('hello')\n# padding",
            stdout="hello",
            stderr="",
            exit_code=0,
            execution_time_s=0.1,
            iterations=1,
            success=True,
        )

        rows = get_code_evidence_for_topic(topic_id)
        assert len(rows) == 1
        assert "source_code" not in rows[0]
        assert rows[0]["hypothesis"] == "test"

    def test_get_for_topic_full_includes_source_code(self):
        with get_db() as conn:
            cursor = conn.execute(
                "INSERT INTO Topic (summary, detail, status) VALUES (?, ?, 'Started')",
                ("t", "d"),
            )
            topic_id = cursor.lastrowid

        insert_code_evidence(
            origin_topic_id=topic_id,
            origin_subtopic_id=None,
            hypothesis="test",
            source_code="print('hello')\n# padding",
            stdout="hello",
            stderr="",
            exit_code=0,
            execution_time_s=0.1,
            iterations=1,
            success=True,
        )

        rows = get_code_evidence_for_topic_full(topic_id)
        assert len(rows) == 1
        assert "source_code" in rows[0]

    def test_get_nonexistent_returns_none(self):
        row = get_code_evidence_by_id(99999)
        assert row is None

    def test_table_exists_after_init(self):
        with get_db() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='CodeEvidence'"
            )
            assert cursor.fetchone() is not None

    def test_code_evidence_preserves_full_source_below_global_content_cap(self):
        with get_db() as conn:
            cursor = conn.execute(
                "INSERT INTO Topic (summary, detail, status) VALUES (?, ?, 'Started')",
                ("t", "d"),
            )
            topic_id = cursor.lastrowid

        long_code = "x = 1\n" * 5000
        eid = insert_code_evidence(
            origin_topic_id=topic_id,
            origin_subtopic_id=None,
            hypothesis="test truncation",
            source_code=long_code,
            stdout="",
            stderr="",
            exit_code=0,
            execution_time_s=0.0,
            iterations=1,
            success=True,
        )
        row = get_code_evidence_by_id(eid)
        assert row["source_code"] == long_code


# ---------------------------------------------------------------------------
# _render_code_evidence tests
# ---------------------------------------------------------------------------


class TestRenderCodeEvidence:
    def test_success(self):
        text = _render_code_evidence(
            hypothesis="Mean is 2.0",
            success=True,
            iterations=1,
            execution_time_s=0.5,
            stdout="2.0",
            stderr="",
        )
        assert "PASSED" in text
        assert "Mean is 2.0" in text
        assert "2.0" in text

    def test_failure_includes_error(self):
        text = _render_code_evidence(
            hypothesis="test",
            success=False,
            iterations=3,
            execution_time_s=1.5,
            stdout="",
            stderr="NameError: name 'x' is not defined",
        )
        assert "FAILED" in text
        assert "NameError" in text

    def test_phase_is_rendered(self):
        text = _render_code_evidence(
            hypothesis="phase test",
            success=False,
            iterations=2,
            execution_time_s=0.7,
            stdout="",
            stderr="boom",
            phase="smoke",
        )
        assert "SMOKE" in text


# ---------------------------------------------------------------------------
# _render_code_evidence_section tests (RAG)
# ---------------------------------------------------------------------------


class TestRenderCodeEvidenceSection:
    def test_empty_records(self):
        assert _render_code_evidence_section([]) == ""

    def test_renders_entries(self):
        records = [
            {"id": 1, "success": True, "hypothesis": "Mean is 2", "stdout": "2.0"},
            {"id": 2, "success": False, "hypothesis": "Variance is 0", "stdout": ""},
        ]
        text = _render_code_evidence_section(records)
        assert "[E1]" in text
        assert "[E2]" in text
        assert "PASSED" in text
        assert "FAILED" in text

    def test_caps_at_max(self):
        records = [
            {"id": i, "success": True, "hypothesis": f"h{i}", "stdout": ""}
            for i in range(20)
        ]
        text = _render_code_evidence_section(records, max_entries=5)
        assert "[E4]" in text
        assert "[E5]" not in text  # 0-indexed: entries 0..4


class TestPromptGrounding:
    @pytest.mark.asyncio
    async def test_generate_code_prompt_mentions_500_line_limit_and_environment(
        self, monkeypatch
    ):
        captured = {}

        async def fake_env_desc():
            return (
                f"# ORBIT Sandbox Environment\n"
                f"- Metadata path: {ENVIRONMENT_DESCRIPTION_PATH}\n"
                f"- Timeouts: smoke={SMOKE_TIMEOUT}s, full={FULL_TIMEOUT}s\n"
            )

        async def fake_call_text(prompt, **kwargs):
            captured["prompt"] = prompt
            return (
                "import sys\n"
                "mode = sys.argv[1] if len(sys.argv) > 1 else 'full'\n"
                "print(mode)\n"
            )

        monkeypatch.setattr(
            code_sandbox, "_get_sandbox_environment_description", fake_env_desc
        )
        monkeypatch.setattr(code_sandbox, "call_text", fake_call_text)

        await code_sandbox._generate_code("hypothesis", "ctx", role="engineer")

        assert f"Keep under {code_sandbox.MAX_SOURCE_LINES} lines" in captured["prompt"]
        assert "python3 - smoke" in captured["prompt"]
        assert "python3 - full" in captured["prompt"]
        assert ENVIRONMENT_DESCRIPTION_PATH in captured["prompt"]


class TestRunCodeEvidenceWorkflow:
    @pytest.mark.asyncio
    async def test_smoke_then_full_uses_two_timeouts(self, monkeypatch):
        with get_db() as conn:
            cursor = conn.execute(
                "INSERT INTO Topic (summary, detail, status) VALUES (?, ?, 'Started')",
                ("topic", "detail"),
            )
            topic_id = cursor.lastrowid

        async def fake_env_desc():
            return "env"

        async def fake_call_text(prompt, **kwargs):
            return (
                "import sys\n"
                "mode = sys.argv[1] if len(sys.argv) > 1 else 'full'\n"
                "print(mode)\n"
            )

        calls = []

        async def fake_execute(source, timeout=0, mode=None):
            calls.append((timeout, mode))
            return ExperimentResult(
                stdout=f"{mode}-ok\nCONCLUSION: test passed\n",
                stderr="",
                exit_code=0,
                execution_time_s=0.5,
                success=True,
            )

        async def fake_review(*args, **kwargs):
            return True, ""

        monkeypatch.setattr(
            code_sandbox, "_get_sandbox_environment_description", fake_env_desc
        )
        monkeypatch.setattr(code_sandbox, "call_text", fake_call_text)
        monkeypatch.setattr(code_sandbox, "_execute_in_sandbox", fake_execute)
        monkeypatch.setattr(code_sandbox, "_review_code_logic", fake_review)

        result = await run_code_evidence(
            "test hypothesis",
            "ctx",
            topic_id=topic_id,
            subtopic_id=None,
            role="scientist",
        )

        assert result.success is True
        assert result.iterations == 2
        assert calls == [(SMOKE_TIMEOUT, "smoke"), (FULL_TIMEOUT, "full")]
        assert "FULL" in result.rendered_results

    @pytest.mark.asyncio
    async def test_smoke_failure_never_reaches_full(self, monkeypatch):
        with get_db() as conn:
            cursor = conn.execute(
                "INSERT INTO Topic (summary, detail, status) VALUES (?, ?, 'Started')",
                ("topic", "detail"),
            )
            topic_id = cursor.lastrowid

        async def fake_env_desc():
            return "env"

        script = (
            "import sys\n"
            "mode = sys.argv[1] if len(sys.argv) > 1 else 'full'\n"
            "print(mode)\n"
        )

        async def fake_call_text(prompt, **kwargs):
            return script

        calls = []

        async def fake_execute(source, timeout=0, mode=None):
            calls.append((timeout, mode))
            return ExperimentResult(
                stdout="",
                stderr="boom",
                exit_code=1,
                execution_time_s=0.5,
                success=False,
            )

        async def fake_fix_code(
            source, stderr, hypothesis, *, phase="", plan="", provider="minimax"
        ):
            return script

        monkeypatch.setattr(
            code_sandbox, "_get_sandbox_environment_description", fake_env_desc
        )
        monkeypatch.setattr(code_sandbox, "call_text", fake_call_text)
        monkeypatch.setattr(code_sandbox, "_execute_in_sandbox", fake_execute)
        monkeypatch.setattr(code_sandbox, "_fix_code", fake_fix_code)

        result = await run_code_evidence(
            "test hypothesis",
            "ctx",
            topic_id=topic_id,
            subtopic_id=None,
            role="scientist",
            max_iterations=3,
        )

        assert result.success is False
        # max_iterations=3: budget split = 1 smoke_fix + 1 full_fix
        # Smoke: 1 initial + 1 fix = 2 executions, both fail → never reaches full
        assert result.iterations == 2
        assert calls == [
            (SMOKE_TIMEOUT, "smoke"),
            (SMOKE_TIMEOUT, "smoke"),
        ]
        assert "SMOKE" in result.rendered_results
        assert "FAILED" in result.rendered_results

    @pytest.mark.asyncio
    async def test_run_code_evidence_respects_timeout_and_iteration_budget(
        self, monkeypatch
    ):
        with get_db() as conn:
            cursor = conn.execute(
                "INSERT INTO Topic (summary, detail, status) VALUES (?, ?, 'Started')",
                ("topic", "detail"),
            )
            topic_id = cursor.lastrowid

        async def fake_env_desc():
            return "env"

        script = (
            "import sys\n"
            "mode = sys.argv[1] if len(sys.argv) > 1 else 'full'\n"
            "print(mode)\n"
        )

        async def fake_call_text(prompt, **kwargs):
            return script

        calls = []

        async def fake_execute(source, timeout=0, mode=None):
            calls.append((timeout, mode))
            if mode == "smoke":
                return ExperimentResult(
                    stdout="smoke-ok\nCONCLUSION: test passed\n",
                    stderr="",
                    exit_code=0,
                    execution_time_s=0.5,
                    success=True,
                )
            return ExperimentResult(
                stdout="",
                stderr="full boom",
                exit_code=1,
                execution_time_s=0.5,
                success=False,
            )

        async def fake_fix_code(source, stderr, hypothesis, *, phase="", plan=""):
            raise AssertionError("fix loop should not run when max_iterations=1")

        monkeypatch.setattr(
            code_sandbox, "_get_sandbox_environment_description", fake_env_desc
        )
        monkeypatch.setattr(code_sandbox, "call_text", fake_call_text)
        monkeypatch.setattr(code_sandbox, "_execute_in_sandbox", fake_execute)
        monkeypatch.setattr(code_sandbox, "_fix_code", fake_fix_code)

        result = await run_code_evidence(
            "test hypothesis",
            "ctx",
            topic_id=topic_id,
            subtopic_id=None,
            role="scientist",
            max_iterations=1,
            timeout=450,
        )

        assert result.success is False
        assert calls == [(SMOKE_TIMEOUT, "smoke"), (450, "full")]

    @pytest.mark.asyncio
    async def test_run_code_evidence_ignores_non_object_plan_json(self, monkeypatch):
        with get_db() as conn:
            cursor = conn.execute(
                "INSERT INTO Topic (summary, detail, status) VALUES (?, ?, 'Started')",
                ("topic", "detail"),
            )
            topic_id = cursor.lastrowid

        async def fake_env_desc():
            return "env"

        async def fake_plan(*args, **kwargs):
            return "[]"

        async def fake_generate(*args, **kwargs):
            return "print('ok')"

        async def fake_run_phase(
            source,
            hypothesis,
            *,
            phase,
            timeout,
            max_fix_attempts,
            review_logic,
            plan,
            provider="minimax",
        ):
            return PhaseRunResult(
                phase=phase,
                source_code=source,
                result=ExperimentResult(
                    stdout=f"{phase}\nCONCLUSION: ok\n",
                    stderr="",
                    exit_code=0,
                    execution_time_s=0.5,
                    success=True,
                ),
                attempts=1,
            )

        inserted = {}

        def fake_insert_code_evidence(**kwargs):
            inserted.update(kwargs)
            return 123

        monkeypatch.setattr(
            code_sandbox, "_get_sandbox_environment_description", fake_env_desc
        )
        monkeypatch.setattr(code_sandbox, "_plan_experiment", fake_plan)
        monkeypatch.setattr(code_sandbox, "_generate_code", fake_generate)
        monkeypatch.setattr(code_sandbox, "_run_code_phase", fake_run_phase)
        monkeypatch.setattr(code_sandbox.api, "insert_code_evidence", fake_insert_code_evidence)

        result = await run_code_evidence(
            "test hypothesis",
            "ctx",
            topic_id=topic_id,
            subtopic_id=None,
            role="scientist",
        )

        assert result.success is True
        assert result.code_evidence_id == 123
        assert result.planning_veto is False
        assert inserted["hypothesis"] == "test hypothesis"

    @pytest.mark.asyncio
    async def test_run_code_evidence_planning_veto_does_not_persist_code_evidence(
        self, monkeypatch
    ):
        with get_db() as conn:
            cursor = conn.execute(
                "INSERT INTO Topic (summary, detail, status) VALUES (?, ?, 'Started')",
                ("topic", "detail"),
            )
            topic_id = cursor.lastrowid

        async def fake_env_desc():
            return "env"

        async def fake_plan(*args, **kwargs):
            return json.dumps(
                {
                    "feasible": False,
                    "rejection_reason": "Dataset is unavailable in sandbox.",
                }
            )

        def fail_insert_code_evidence(**kwargs):
            raise AssertionError("planning veto must not persist CodeEvidence")

        monkeypatch.setattr(
            code_sandbox, "_get_sandbox_environment_description", fake_env_desc
        )
        monkeypatch.setattr(code_sandbox, "_plan_experiment", fake_plan)
        monkeypatch.setattr(code_sandbox.api, "insert_code_evidence", fail_insert_code_evidence)

        result = await run_code_evidence(
            "test hypothesis",
            "ctx",
            topic_id=topic_id,
            subtopic_id=None,
            role="scientist",
        )

        assert result.success is False
        assert result.code_evidence_id == 0
        assert result.planning_veto is True
        assert "Rejected during planning" in result.stderr
        assert "Dataset is unavailable in sandbox." in result.rendered_results


class TestEnsureSandbox:
    def test_ensure_sandbox_rebuilds_and_recreates_container(self, monkeypatch):
        calls = []

        class Result:
            def __init__(self, returncode=0):
                self.returncode = returncode
                self.stdout = b""

        def fake_run(cmd, capture_output=False, timeout=None, check=False):
            calls.append(cmd)
            if cmd[:3] == ["docker", "inspect", "orbit-or-sandbox"]:
                return Result(returncode=0)
            return Result(returncode=0)

        monkeypatch.setattr(code_sandbox.subprocess, "run", fake_run)

        assert code_sandbox.ensure_sandbox() is True
        assert [
            "docker",
            "build",
            "-t",
            "orbit-or-sandbox:latest",
            str(Path(code_sandbox.__file__).parent / "sandbox"),
        ] in calls
        assert ["docker", "rm", "-f", "orbit-or-sandbox"] in calls


class TestSandboxBuildFiles:
    def test_dockerfile_bakes_environment_description(self):
        text = Path("src/orbit_or/sandbox/Dockerfile").read_text()
        assert "/opt/orbit/ENVIRONMENT.md" in text

    def test_build_script_refreshes_container(self):
        text = Path("src/orbit_or/sandbox/build.sh").read_text()
        assert 'docker rm -f "$CONTAINER_NAME"' in text


# ---------------------------------------------------------------------------
# Sympy import validation
# ---------------------------------------------------------------------------


class TestValidateSourceSympy:
    def test_sympy_allowed(self):
        code = "from sympy import symbols, solve\nx = symbols('x')\nprint(solve(x**2 - 4, x))"
        ok, reason = validate_source(code)
        assert ok, reason

    def test_import_sympy(self):
        code = "import sympy\nprint(sympy.pi)"
        ok, reason = validate_source(code)
        assert ok, reason


# ---------------------------------------------------------------------------
# _extract_calc_requests tests
# ---------------------------------------------------------------------------


class TestExtractCalcRequests:
    def test_single_calc(self):
        content = "Let me check: [CALC: 3.14 * 2.5**2] that should be about 19.6"
        result = _extract_calc_requests(content)
        assert result == ["3.14 * 2.5**2"]

    def test_multiple_calcs(self):
        content = "[CALC: 1+1] and [CALC: 2*3] and [CALC: 4/2]"
        result = _extract_calc_requests(content)
        assert result == ["1+1", "2*3", "4/2"]

    def test_max_three_calcs(self):
        content = "[CALC: 1] [CALC: 2] [CALC: 3] [CALC: 4]"
        result = _extract_calc_requests(content)
        assert len(result) == 3

    def test_no_calc(self):
        content = "No calculations here."
        result = _extract_calc_requests(content)
        assert result == []

    def test_calc_with_spaces(self):
        content = "[CALC:   100 / 3   ]"
        result = _extract_calc_requests(content)
        assert result == ["100 / 3"]


# ---------------------------------------------------------------------------
# _extract_code_verify_requests (plural) tests
# ---------------------------------------------------------------------------


class TestExtractCodeVerifyRequests:
    def test_multiple_verify(self):
        content = (
            "[CODE_VERIFY: hyp1] text [CODE_VERIFY: hyp2] more [CODE_VERIFY: hyp3]"
        )
        result = _extract_code_verify_requests(content)
        assert result == ["hyp1", "hyp2", "hyp3"]

    def test_max_three_verify(self):
        content = "[CODE_VERIFY: a] [CODE_VERIFY: b] [CODE_VERIFY: c] [CODE_VERIFY: d]"
        result = _extract_code_verify_requests(content)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# _extract_code_review_request tests
# ---------------------------------------------------------------------------


class TestExtractCodeReviewRequest:
    def test_simple_review(self):
        content = "[CODE_REVIEW: E42, t-test assumes normality but data is skewed]"
        result = _extract_code_review_request(content)
        assert result == (42, "t-test assumes normality but data is skewed")

    def test_review_with_spaces(self):
        content = "[CODE_REVIEW: E7,  the regression is overfitting  ]"
        result = _extract_code_review_request(content)
        assert result == (7, "the regression is overfitting")

    def test_no_review(self):
        content = "No code review here."
        result = _extract_code_review_request(content)
        assert result is None

    def test_review_only_first(self):
        content = "[CODE_REVIEW: E1, issue1] and [CODE_REVIEW: E2, issue2]"
        result = _extract_code_review_request(content)
        assert result == (1, "issue1")


# ---------------------------------------------------------------------------
# _get_code_tier tests
# ---------------------------------------------------------------------------


class TestGetCodeTier:
    def _state(self, phase, round_number=2):
        return {"phase": phase, "round_number": round_number}

    # Tier 2: verify — analyst/scientist/engineer/contrarian in EVIDENCE+ANALYSIS
    def test_scientist_evidence_verify(self):
        assert (
            _get_code_tier(self._state(EVIDENCE_PHASE), "scientist", BASE_TURN)
            == "verify"
        )

    def test_engineer_analysis_verify(self):
        assert (
            _get_code_tier(self._state(ANALYSIS_PHASE), "engineer", BASE_TURN) == "verify"
        )

    def test_analyst_evidence_verify(self):
        assert (
            _get_code_tier(self._state(EVIDENCE_PHASE), "analyst", BASE_TURN)
            == "verify"
        )

    def test_contrarian_analysis_verify(self):
        assert (
            _get_code_tier(self._state(ANALYSIS_PHASE), "contrarian", BASE_TURN)
            == "verify"
        )

    # Tier 3: review — critic in EVIDENCE+ANALYSIS
    def test_critic_evidence_review(self):
        assert (
            _get_code_tier(self._state(EVIDENCE_PHASE), "critic", BASE_TURN) == "review"
        )

    def test_critic_analysis_review(self):
        assert (
            _get_code_tier(self._state(ANALYSIS_PHASE), "critic", BASE_TURN) == "review"
        )

    # Tier 1: calc — all voting agents in any phase
    def test_dreamer_opening_calc(self):
        assert (
            _get_code_tier(self._state(OPENING_PHASE, 1), "dreamer", BASE_TURN)
            == "calc"
        )

    def test_dreamer_evidence_calc(self):
        assert (
            _get_code_tier(self._state(EVIDENCE_PHASE), "dreamer", BASE_TURN) == "calc"
        )

    def test_cat_analysis_calc(self):
        assert _get_code_tier(self._state(ANALYSIS_PHASE), "cat", BASE_TURN) == "calc"

    def test_dog_opening_calc(self):
        assert _get_code_tier(self._state(OPENING_PHASE, 1), "dog", BASE_TURN) == "calc"

    def test_skynet_calc(self):
        assert _get_code_tier(self._state(ANALYSIS_PHASE), "skynet", BASE_TURN) == "calc"

    # Scientist in OPENING only gets calc (not verify)
    def test_scientist_opening_calc(self):
        assert (
            _get_code_tier(self._state(OPENING_PHASE, 1), "scientist", BASE_TURN)
            == "calc"
        )

    # Non-BASE_TURN: always None (prevents recursion)
    def test_followup_turn_none(self):
        assert (
            _get_code_tier(self._state(EVIDENCE_PHASE), "scientist", CODE_FOLLOWUP_TURN)
            is None
        )

    def test_tron_remediation_none(self):
        assert (
            _get_code_tier(self._state(EVIDENCE_PHASE), "scientist", "tron_remediation")
            is None
        )

    # NPC agents: no code tier
    def test_writer_none(self):
        assert _get_code_tier(self._state(ANALYSIS_PHASE), "writer", BASE_TURN) is None

    def test_librarian_none(self):
        assert _get_code_tier(self._state(ANALYSIS_PHASE), "librarian", BASE_TURN) is None

    # Backward compat: should_enable_code_exec
    def test_should_enable_code_exec_compat_scientist(self):
        assert should_enable_code_exec(
            self._state(EVIDENCE_PHASE), "scientist", BASE_TURN
        )

    def test_should_enable_code_exec_compat_critic_now_true(self):
        """Critic has review tier, which should_enable_code_exec now returns True for."""
        assert should_enable_code_exec(self._state(ANALYSIS_PHASE), "critic", BASE_TURN)


# ---------------------------------------------------------------------------
# DB: parent_evidence_id tests
# ---------------------------------------------------------------------------


class TestCodeEvidenceParentId:
    def test_insert_with_parent(self):
        with get_db() as conn:
            cursor = conn.execute(
                "INSERT INTO Topic (summary, detail, status) VALUES (?, ?, 'Started')",
                ("t", "d"),
            )
            topic_id = cursor.lastrowid

        # Insert original
        eid1 = insert_code_evidence(
            origin_topic_id=topic_id,
            origin_subtopic_id=None,
            hypothesis="original",
            source_code="print('orig')\n# padding",
            stdout="orig",
            stderr="",
            exit_code=0,
            execution_time_s=0.5,
            iterations=1,
            success=True,
            requesting_role="scientist",
        )

        # Insert review with parent
        eid2 = insert_code_evidence(
            origin_topic_id=topic_id,
            origin_subtopic_id=None,
            hypothesis="review of E1",
            source_code="print('review')\n# padding",
            stdout="review",
            stderr="",
            exit_code=0,
            execution_time_s=0.3,
            iterations=1,
            success=True,
            requesting_role="critic",
            parent_evidence_id=eid1,
        )

        row = get_code_evidence_by_id(eid2)
        assert row is not None
        assert row["parent_evidence_id"] == eid1

    def test_parent_none_by_default(self):
        with get_db() as conn:
            cursor = conn.execute(
                "INSERT INTO Topic (summary, detail, status) VALUES (?, ?, 'Started')",
                ("t", "d"),
            )
            topic_id = cursor.lastrowid

        eid = insert_code_evidence(
            origin_topic_id=topic_id,
            origin_subtopic_id=None,
            hypothesis="no parent test",
            source_code="print('test')\n# padding",
            stdout="test",
            stderr="",
            exit_code=0,
            execution_time_s=0.1,
            iterations=1,
            success=True,
        )

        row = get_code_evidence_by_id(eid)
        assert row["parent_evidence_id"] is None

    def test_parent_in_topic_full_query(self):
        with get_db() as conn:
            cursor = conn.execute(
                "INSERT INTO Topic (summary, detail, status) VALUES (?, ?, 'Started')",
                ("t", "d"),
            )
            topic_id = cursor.lastrowid

        eid1 = insert_code_evidence(
            origin_topic_id=topic_id,
            origin_subtopic_id=None,
            hypothesis="orig",
            source_code="print('a')\n# padding line",
            stdout="a",
            stderr="",
            exit_code=0,
            execution_time_s=0.1,
            iterations=1,
            success=True,
        )
        insert_code_evidence(
            origin_topic_id=topic_id,
            origin_subtopic_id=None,
            hypothesis="review",
            source_code="print('b')\n# padding line",
            stdout="b",
            stderr="",
            exit_code=0,
            execution_time_s=0.1,
            iterations=1,
            success=True,
            parent_evidence_id=eid1,
        )

        rows = get_code_evidence_for_topic_full(topic_id)
        assert len(rows) == 2
        # Ordered by id DESC, so review comes first
        assert rows[0]["parent_evidence_id"] == eid1
        assert rows[1]["parent_evidence_id"] is None

    def test_parent_in_topic_query(self):
        with get_db() as conn:
            cursor = conn.execute(
                "INSERT INTO Topic (summary, detail, status) VALUES (?, ?, 'Started')",
                ("t", "d"),
            )
            topic_id = cursor.lastrowid

        eid1 = insert_code_evidence(
            origin_topic_id=topic_id,
            origin_subtopic_id=None,
            hypothesis="orig",
            source_code="print('a')\n# padding line",
            stdout="a",
            stderr="",
            exit_code=0,
            execution_time_s=0.1,
            iterations=1,
            success=True,
        )
        insert_code_evidence(
            origin_topic_id=topic_id,
            origin_subtopic_id=None,
            hypothesis="review",
            source_code="print('b')\n# padding line",
            stdout="b",
            stderr="",
            exit_code=0,
            execution_time_s=0.1,
            iterations=1,
            success=True,
            parent_evidence_id=eid1,
        )

        rows = get_code_evidence_for_topic(topic_id)
        assert len(rows) == 2
        assert rows[0]["parent_evidence_id"] == eid1


# ---------------------------------------------------------------------------
# ROLE_CODE_HINTS tests
# ---------------------------------------------------------------------------


class TestRoleCodeHints:
    def test_all_four_roles_have_hints(self):
        for role in ("analyst", "scientist", "engineer", "contrarian"):
            assert role in ROLE_CODE_HINTS
            assert len(ROLE_CODE_HINTS[role]) > 20

    def test_no_hint_for_dreamer(self):
        assert "dreamer" not in ROLE_CODE_HINTS


# ---------------------------------------------------------------------------
# validate_calc_expression (AST whitelist) tests
# ---------------------------------------------------------------------------


class TestValidateCalcExpression:
    def test_simple_arithmetic(self):
        ok, _ = validate_calc_expression("3.14 * 2.5**2")
        assert ok

    def test_addition(self):
        ok, _ = validate_calc_expression("1 + 2 + 3")
        assert ok

    def test_comparison(self):
        ok, _ = validate_calc_expression("3 > 2")
        assert ok

    def test_ternary(self):
        ok, _ = validate_calc_expression("1 if True else 0")
        assert ok

    def test_builtin_functions(self):
        ok, _ = validate_calc_expression("max(1, 2, 3)")
        assert ok

    def test_abs(self):
        ok, _ = validate_calc_expression("abs(-42)")
        assert ok

    def test_round(self):
        ok, _ = validate_calc_expression("round(3.14159, 2)")
        assert ok

    def test_list_literal(self):
        ok, _ = validate_calc_expression("[1, 2, 3]")
        assert ok

    def test_nested_arithmetic(self):
        ok, _ = validate_calc_expression("(100 / 3) * (2 + 1)")
        assert ok

    # --- Rejected expressions ---

    def test_reject_empty(self):
        ok, _ = validate_calc_expression("")
        assert not ok

    def test_reject_too_long(self):
        ok, _ = validate_calc_expression("1 + " * 200)
        assert not ok

    def test_reject_import(self):
        ok, reason = validate_calc_expression("__import__('os')")
        assert not ok

    def test_reject_function_call(self):
        ok, reason = validate_calc_expression("eval('1+1')")
        assert not ok

    def test_reject_lambda(self):
        ok, reason = validate_calc_expression("(lambda: 1)()")
        assert not ok

    def test_reject_variable(self):
        ok, reason = validate_calc_expression("x + 1")
        assert not ok

    def test_reject_dict_literal(self):
        ok, reason = validate_calc_expression("{'a': 1}")
        assert not ok

    def test_reject_attribute_access(self):
        ok, reason = validate_calc_expression("'hello'.upper()")
        assert not ok

    def test_reject_class_dunder(self):
        ok, reason = validate_calc_expression("(1).__class__")
        assert not ok

    def test_reject_semicolon_injection(self):
        """Semicolons create a statement, not an expression — SyntaxError."""
        ok, reason = validate_calc_expression("1); import os; print(")
        assert not ok

    def test_reject_multiline_injection(self):
        ok, reason = validate_calc_expression("1)\nimport os\nprint(")
        assert not ok

    def test_reject_type_call(self):
        ok, reason = validate_calc_expression("type(1)")
        assert not ok

    def test_reject_comprehension(self):
        ok, reason = validate_calc_expression("[x for x in range(10)]")
        assert not ok

    def test_reject_walrus(self):
        ok, reason = validate_calc_expression("(x := 42)")
        assert not ok


# ---------------------------------------------------------------------------
# New blocked patterns tests (sys.exit, locals, etc.)
# ---------------------------------------------------------------------------


class TestNewBlockedPatterns:
    def test_allow_sys_exit(self):
        """sys.exit() is harmless in sandbox — container process just exits."""
        ok, _ = validate_source("import sys\nsys.exit(1)\nprint('done')")
        assert ok

    def test_reject_sys_setrecursionlimit(self):
        ok, _ = validate_source(
            "import sys\nsys.setrecursionlimit(999999)\nprint('ok')"
        )
        assert not ok

    def test_reject_sys_settrace(self):
        ok, _ = validate_source("import sys\nsys.settrace(None)\nprint('traced')")
        assert not ok

    def test_reject_sys_setprofile(self):
        ok, _ = validate_source("import sys\nsys.setprofile(None)\nprint('profiled')")
        assert not ok

    def test_reject_sys_stdin(self):
        ok, _ = validate_source("import sys\ndata = sys.stdin.read()\nprint(data)")
        assert not ok

    def test_reject_locals(self):
        ok, _ = validate_source("l = locals()\nprint(l)\nextra line here")
        assert not ok

    def test_allow_sys_stdout(self):
        """sys.stdout should still be allowed for normal output."""
        ok, reason = validate_source(
            "import sys\nsys.stdout.write('hello')\nprint('ok')"
        )
        assert ok, reason

    def test_reject_setattr(self):
        ok, _ = validate_source("import math\nsetattr(math, 'x', 1)\nprint('done')")
        assert not ok

    def test_reject_delattr(self):
        ok, _ = validate_source("import math\ndelattr(math, 'sin')\nprint('gone')")
        assert not ok

    def test_reject_breakpoint(self):
        ok, _ = validate_source("breakpoint()\nprint('debug')\nextra line")
        assert not ok

    def test_reject_bare_compile(self):
        ok, _ = validate_source(
            "code = compile('x=1', '<string>', 'exec')\nprint(code)\npadding"
        )
        assert not ok

    def test_allow_re_compile(self):
        """re.compile should still be allowed."""
        ok, reason = validate_source(
            "import re\np = re.compile(r'\\d+')\nprint(p.findall('a1b2'))"
        )
        assert ok, reason


# ---------------------------------------------------------------------------
# Additional validate_calc_expression tests (round 2)
# ---------------------------------------------------------------------------


class TestCalcExpressionRound2:
    def test_math_sqrt(self):
        ok, _ = validate_calc_expression("math.sqrt(144)")
        assert ok

    def test_math_log(self):
        ok, _ = validate_calc_expression("math.log(100, 10)")
        assert ok

    def test_statistics_mean(self):
        ok, _ = validate_calc_expression("statistics.mean([1, 2, 3, 4, 5])")
        assert ok

    def test_math_pi(self):
        """math.pi is an attribute, not a call — should be allowed."""
        ok, _ = validate_calc_expression("math.pi * 2")
        assert ok

    def test_starred_in_call(self):
        ok, _ = validate_calc_expression("max(*[1, 2, 3])")
        assert ok

    def test_reject_fstring(self):
        ok, _ = validate_calc_expression("f'{1+1}'")
        assert not ok

    def test_reject_generator_expression(self):
        ok, _ = validate_calc_expression("sum(x for x in [1,2,3])")
        assert not ok

    def test_reject_subscript(self):
        ok, _ = validate_calc_expression("[1,2,3][0]")
        assert not ok

    def test_reject_set_literal(self):
        ok, _ = validate_calc_expression("{1, 2, 3}")
        assert not ok

    def test_reject_range(self):
        """range was removed from safe calls to prevent DoS."""
        ok, _ = validate_calc_expression("sum(range(10))")
        assert not ok

    def test_reject_sorted(self):
        """sorted was removed from safe calls."""
        ok, _ = validate_calc_expression("sorted([3,1,2])")
        assert not ok

    def test_reject_arbitrary_module_attr(self):
        """Only math and statistics modules are allowed."""
        ok, _ = validate_calc_expression("os.getcwd()")
        assert not ok

    def test_reject_math_unlisted_function(self):
        """Only whitelisted math functions are allowed as calls."""
        ok, _ = validate_calc_expression("math.evil()")
        assert not ok

    def test_allow_math_attribute_value(self):
        """math.pi as a value (not a call) is allowed via Attribute check."""
        ok, _ = validate_calc_expression("math.e")
        assert ok

    def test_double_validation_consistency(self):
        """Both validators must agree for an expression to execute."""
        expr = "3.14 * 2.5**2"
        ok1, _ = validate_calc_expression(expr)
        ok2, _ = validate_source(f"print({expr})")
        assert ok1 and ok2

    def test_double_validation_math(self):
        """math.sqrt requires import injection — validate_source must see the import."""
        expr = "math.sqrt(4)"
        ok1, _ = validate_calc_expression(expr)
        assert ok1
        # Without import, validate_source would reject 'math' as unknown import
        # But run_calc auto-injects the import, so test the injected source
        source = f"import math\nprint({expr})"
        ok2, _ = validate_source(source)
        assert ok2

    def test_keyword_argument(self):
        """round(3.14, ndigits=2) should be allowed."""
        ok, _ = validate_calc_expression("round(3.14, ndigits=2)")
        assert ok

    def test_bitwise_not(self):
        """~5 should be allowed (ast.Invert)."""
        ok, _ = validate_calc_expression("~5")
        assert ok


# ---------------------------------------------------------------------------
# Grid pipeline tests
# ---------------------------------------------------------------------------


def _create_parent_evidence(topic_id, success=True):
    """Insert a code evidence row to serve as grid parent."""
    return insert_code_evidence(
        origin_topic_id=topic_id,
        origin_subtopic_id=None,
        hypothesis="Full-mode experiment",
        source_code="print('CONCLUSION: works')",
        stdout="CONCLUSION: works",
        stderr="",
        exit_code=0,
        execution_time_s=1.0,
        iterations=2,
        success=success,
        requesting_role="scientist",
        summary="[PASSED FULL] Full-mode experiment",
    )


class TestTryParseGridStdout:
    def test_empty(self):
        ok, msg = _try_parse_grid_stdout("")
        assert not ok

    def test_short(self):
        ok, msg = _try_parse_grid_stdout("some output")
        assert not ok

    def test_with_conclusion(self):
        ok, msg = _try_parse_grid_stdout(
            "Running grid experiment with multiple seeds and configurations over datasets\n"
            "CONCLUSION: MLP beats RF on 4/6 datasets with mean F1 improvement of 0.03"
        )
        assert ok
        assert "MLP beats RF" in msg

    def test_partial_data_points(self):
        stdout = (
            "Running grid search over multiple seeds for the scaling experiment:\n"
            "seed=42: MLP F1=0.85 on synthetic data\n"
            "seed=123: MLP F1=0.83 on synthetic data\n"
            "seed=456: MLP F1=0.81 on synthetic data\n"
        )
        ok, msg = _try_parse_grid_stdout(stdout)
        assert ok
        assert "3" in msg  # 3 data points

    def test_numeric_lines(self):
        stdout = (
            "MLP  0.85  0.02  wine\n"
            "RF   0.83  0.01  wine\n"
            "GBT  0.84  0.03  wine\n"
        )
        ok, msg = _try_parse_grid_stdout(stdout)
        assert ok

    def test_insufficient(self):
        ok, msg = _try_parse_grid_stdout("seed=42: F1=0.85")
        assert not ok


class TestClassifyFailureGrid:
    def test_timeout_non_grid(self):
        result = _classify_failure(-1, "Execution timed out", "")
        assert "TIMEOUT_REDUCE_SCOPE" in result

    def test_timeout_grid(self):
        result = _classify_failure(-1, "Execution timed out", "", is_grid=True)
        assert "TIMEOUT_GRID" in result

    def test_code_invalid_not_affected(self):
        result = _classify_failure(
            -1, "All validation attempts failed", "", is_grid=True
        )
        assert "CODE_INVALID" in result


@pytest.mark.asyncio
class TestRunCodeEvidenceGrid:
    async def test_grid_early_return_missing_evidence(self):
        with get_db() as conn:
            conn.execute(
                "INSERT INTO Topic (id, summary, detail, status) VALUES (1, 's', 'd', 'Running')"
            )
        result = await run_code_evidence_grid(
            "test",
            "ctx",
            topic_id=1,
            subtopic_id=None,
            role="scientist",
            full_evidence_id=99999,
        )
        assert not result.success
        assert "not found" in result.stderr

    async def test_grid_early_return_not_passed(self):
        with get_db() as conn:
            conn.execute(
                "INSERT INTO Topic (id, summary, detail, status) VALUES (1, 's', 'd', 'Running')"
            )
        parent_id = _create_parent_evidence(1, success=False)
        result = await run_code_evidence_grid(
            "test",
            "ctx",
            topic_id=1,
            subtopic_id=None,
            role="scientist",
            full_evidence_id=parent_id,
        )
        assert not result.success
        assert "not found or not PASSED" in result.stderr

    async def test_grid_cross_topic_rejection(self):
        with get_db() as conn:
            conn.execute(
                "INSERT INTO Topic (id, summary, detail, status) VALUES (1, 's', 'd', 'Running')"
            )
            conn.execute(
                "INSERT INTO Topic (id, summary, detail, status) VALUES (2, 's2', 'd2', 'Running')"
            )
        parent_id = _create_parent_evidence(1)
        result = await run_code_evidence_grid(
            "test",
            "ctx",
            topic_id=2,
            subtopic_id=None,
            role="scientist",
            full_evidence_id=parent_id,
        )
        assert not result.success
        assert "different topic" in result.stderr

    async def test_grid_smoke_failure(self, monkeypatch):
        with get_db() as conn:
            conn.execute(
                "INSERT INTO Topic (id, summary, detail, status) VALUES (1, 's', 'd', 'Running')"
            )
        parent_id = _create_parent_evidence(1)

        async def fake_call_text(prompt, **kwargs):
            return "import sys\nmode=sys.argv[1] if len(sys.argv)>1 else 'grid'\nprint(mode)"

        async def fake_execute(source, timeout=0, mode=None):
            return ExperimentResult(
                stderr="Syntax error",
                exit_code=1,
                success=False,
            )

        monkeypatch.setattr(code_sandbox, "call_text", fake_call_text)
        monkeypatch.setattr(code_sandbox, "_execute_in_sandbox", fake_execute)

        result = await run_code_evidence_grid(
            "test grid",
            "ctx",
            topic_id=1,
            subtopic_id=None,
            role="scientist",
            full_evidence_id=parent_id,
        )
        assert not result.success
        assert result.code_evidence_id != 0  # persisted

    async def test_grid_full_success(self, monkeypatch):
        with get_db() as conn:
            conn.execute(
                "INSERT INTO Topic (id, summary, detail, status) VALUES (1, 's', 'd', 'Running')"
            )
        parent_id = _create_parent_evidence(1)

        async def fake_call_text(prompt, **kwargs):
            return (
                "import sys\n"
                "mode=sys.argv[1] if len(sys.argv)>1 else 'grid'\n"
                "if mode=='smoke': print('CONCLUSION: smoke ok')\n"
                "else: print('CONCLUSION: grid results confirmed')\n"
            )

        async def fake_execute(source, timeout=0, mode=None):
            return ExperimentResult(
                stdout=f"CONCLUSION: {mode} ok\n",
                stderr="",
                exit_code=0,
                execution_time_s=1.0,
                success=True,
            )

        async def fake_review(*args, **kwargs):
            return True, ""

        monkeypatch.setattr(code_sandbox, "call_text", fake_call_text)
        monkeypatch.setattr(code_sandbox, "_execute_in_sandbox", fake_execute)
        monkeypatch.setattr(code_sandbox, "_review_code_logic", fake_review)
        monkeypatch.setattr(code_sandbox, "_review_smoke_output", lambda *a: (True, ""))

        result = await run_code_evidence_grid(
            "test scaling",
            "ctx",
            topic_id=1,
            subtopic_id=None,
            role="scientist",
            full_evidence_id=parent_id,
        )
        assert result.success
        assert "Grid sweep" in result.hypothesis

    async def test_grid_timeout_stdout_salvage(self, monkeypatch):
        with get_db() as conn:
            conn.execute(
                "INSERT INTO Topic (id, summary, detail, status) VALUES (1, 's', 'd', 'Running')"
            )
        parent_id = _create_parent_evidence(1)

        call_count = [0]

        async def fake_call_text(prompt, **kwargs):
            return "import sys\nprint('CONCLUSION: ok')"

        async def fake_execute(source, timeout=0, mode=None):
            call_count[0] += 1
            if mode == "smoke":
                return ExperimentResult(
                    stdout="CONCLUSION: smoke ok\n",
                    stderr="",
                    exit_code=0,
                    execution_time_s=0.5,
                    success=True,
                )
            # Grid: timeout with partial data (>50 chars for _try_parse_grid_stdout)
            return ExperimentResult(
                stdout="Running grid experiment across seeds:\nseed=1: MLP F1=0.85 on synthetic binary\nseed=2: MLP F1=0.83 on synthetic binary\nseed=3: MLP F1=0.81 on synthetic binary\n",
                stderr="Execution timed out",
                exit_code=-1,
                execution_time_s=1200.0,
                success=False,
            )

        async def fake_review(*args, **kwargs):
            return True, ""

        monkeypatch.setattr(code_sandbox, "call_text", fake_call_text)
        monkeypatch.setattr(code_sandbox, "_execute_in_sandbox", fake_execute)
        monkeypatch.setattr(code_sandbox, "_review_code_logic", fake_review)
        monkeypatch.setattr(code_sandbox, "_review_smoke_output", lambda *a: (True, ""))

        result = await run_code_evidence_grid(
            "test scaling",
            "ctx",
            topic_id=1,
            subtopic_id=None,
            role="scientist",
            full_evidence_id=parent_id,
        )
        assert result.success  # salvaged from stdout
        assert "parsed from stdout" in result.stderr

    async def test_grid_logic_review_no_override(self, monkeypatch):
        with get_db() as conn:
            conn.execute(
                "INSERT INTO Topic (id, summary, detail, status) VALUES (1, 's', 'd', 'Running')"
            )
        parent_id = _create_parent_evidence(1)

        async def fake_call_text(prompt, **kwargs):
            return "import sys\nprint('CONCLUSION: ok')"

        async def fake_execute(source, timeout=0, mode=None):
            if mode == "smoke":
                return ExperimentResult(
                    stdout="CONCLUSION: smoke ok\n",
                    stderr="",
                    exit_code=0,
                    execution_time_s=0.5,
                    success=True,
                )
            # Grid: passes execution but has data
            return ExperimentResult(
                stdout="seed=1 F1=0.85\nseed=2 F1=0.83\nseed=3 F1=0.81\nCONCLUSION: ok\n",
                stderr="",
                exit_code=0,
                execution_time_s=10.0,
                success=True,
            )

        async def fake_review(*args, **kwargs):
            # Logic review FAILS
            return False, "Logic review failed: methodology is flawed"

        async def fake_fix(source, stderr, hypothesis, **kwargs):
            return source

        monkeypatch.setattr(code_sandbox, "call_text", fake_call_text)
        monkeypatch.setattr(code_sandbox, "_execute_in_sandbox", fake_execute)
        monkeypatch.setattr(code_sandbox, "_review_code_logic", fake_review)
        monkeypatch.setattr(code_sandbox, "_review_smoke_output", lambda *a: (True, ""))
        monkeypatch.setattr(code_sandbox, "_fix_code", fake_fix)

        result = await run_code_evidence_grid(
            "test scaling",
            "ctx",
            topic_id=1,
            subtopic_id=None,
            role="scientist",
            full_evidence_id=parent_id,
        )
        # Should NOT be salvaged despite having data — logic review rejection is final
        assert not result.success

    async def test_grid_fix_rerun_path(self, monkeypatch):
        """When grid fails, stdout not parseable → fix code → rerun once."""
        with get_db() as conn:
            conn.execute(
                "INSERT INTO Topic (id, summary, detail, status) VALUES (1, 's', 'd', 'Running')"
            )
        parent_id = _create_parent_evidence(1)

        async def fake_call_text(prompt, **kwargs):
            return "import sys\nprint('CONCLUSION: ok')"

        run_count = [0]

        async def fake_execute(source, timeout=0, mode=None):
            run_count[0] += 1
            if mode == "smoke":
                return ExperimentResult(
                    stdout="CONCLUSION: smoke ok\n",
                    stderr="",
                    exit_code=0,
                    execution_time_s=0.5,
                    success=True,
                )
            # Grid: first run fails with no parseable data, second run succeeds
            if run_count[0] <= 3:  # smoke(1) + grid_fail(2) + grid_retry(3)
                return ExperimentResult(
                    stdout="error output only\n",
                    stderr="Runtime crash",
                    exit_code=1,
                    execution_time_s=5.0,
                    success=False,
                )
            return ExperimentResult(
                stdout="CONCLUSION: fixed and works\n",
                stderr="",
                exit_code=0,
                execution_time_s=10.0,
                success=True,
            )

        async def fake_review(*args, **kwargs):
            return True, ""

        async def fake_fix(source, stderr, hypothesis, **kwargs):
            return source  # return same code, fake_execute will succeed on retry

        monkeypatch.setattr(code_sandbox, "call_text", fake_call_text)
        monkeypatch.setattr(code_sandbox, "_execute_in_sandbox", fake_execute)
        monkeypatch.setattr(code_sandbox, "_review_code_logic", fake_review)
        monkeypatch.setattr(code_sandbox, "_review_smoke_output", lambda *a: (True, ""))
        monkeypatch.setattr(code_sandbox, "_fix_code", fake_fix)

        grid_result = await run_code_evidence_grid(
            "test fix rerun",
            "ctx",
            topic_id=1,
            subtopic_id=None,
            role="scientist",
            full_evidence_id=parent_id,
        )
        # The fix-rerun path was exercised (run_count > 2 means grid ran twice)
        assert run_count[0] >= 3
        assert grid_result.code_evidence_id != 0  # persisted
