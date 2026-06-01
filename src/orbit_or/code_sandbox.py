"""Code execution sandbox: generate → validate → run → fix loop.

Agents can verify numerical claims by writing [CODE_VERIFY: <hypothesis>]
in their messages.  This module generates Python code via MiniMax, validates
it, runs it in an air-gapped Docker container, and (on failure) feeds the
error back to the LLM for up to *max_iterations* fix attempts.

Pure functions (validate_source, execute_in_sandbox) are adapted from an
earlier experiment runner to avoid pulling in a broader import chain.
"""

import ast
import asyncio
import json
import logging
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import analytics
from . import api
from .broker import DEFAULT_MAX_TOKENS, call_text

logger = logging.getLogger(__name__)

CONTAINER_NAME = "orbit-or-sandbox"
IMAGE_NAME = "orbit-or-sandbox:latest"
MAX_SOURCE_LINES = 500
SMOKE_TIMEOUT = 120
FULL_TIMEOUT = 300
GRID_TIMEOUT = 1200
GRID_SMOKE_TIMEOUT = 120
DEFAULT_TIMEOUT = SMOKE_TIMEOUT
MAX_SMOKE_FIX_ATTEMPTS = 10
MAX_FULL_FIX_ATTEMPTS = 3
MAX_GRID_SMOKE_FIX_ATTEMPTS = 3
# Grid full fix: 1 attempt via _fix_code → _run_code_phase(max_fix_attempts=0) in run_code_evidence_grid
ENVIRONMENT_DESCRIPTION_PATH = "/opt/orbit/ENVIRONMENT.md"
# Must match the UID set by `useradd -u 1000` in sandbox/Dockerfile
_SANDBOX_UID = 1000

ALLOWED_IMPORTS = frozenset(
    {
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "torch",
        "sklearn",
        "math",
        "random",
        "statistics",
        "itertools",
        "collections",
        "functools",
        "json",
        "csv",
        "io",
        "sys",
        "time",
        "typing",
        "dataclasses",
        "pathlib",
        "re",
        "struct",
        "sympy",
    }
)

CODE_FOLLOWUP_TURN = "code_followup"

CALC_TIMEOUT = 10

# Mutable set: extended at runtime when per-topic sandbox packages are installed
_extra_allowed_imports: dict[int, set[str]] = {}  # per-topic extras keyed by topic_id


def get_allowed_imports(topic_id: int | None = None) -> frozenset[str]:
    """Return the current allowed import set (base + per-topic extras).

    When topic_id is None, returns base + ALL per-topic extras (conservative).
    TODO: Thread topic_id through validate_source and prompt builders for proper isolation.
    """
    if topic_id is not None:
        extras = _extra_allowed_imports.get(topic_id, set())
    else:
        # No topic_id provided — include all extras to avoid false rejections
        extras: set[str] = set()
        for s in _extra_allowed_imports.values():
            extras |= s
    return ALLOWED_IMPORTS | frozenset(extras)


def clear_extra_imports(topic_id: int | None = None) -> None:
    """Reset per-topic extra imports."""
    if topic_id is not None:
        _extra_allowed_imports.pop(topic_id, None)
    else:
        _extra_allowed_imports.clear()


_SAFE_PKG_RE = re.compile(
    r"^[a-zA-Z0-9_][a-zA-Z0-9._\-]*(?:\[[a-zA-Z0-9_,\- ]*\])?(?:[<>=!~]+[a-zA-Z0-9._\-*]+)?$"
)

# Hardcoded allowlist — only these packages can be pip-installed in the sandbox.
# Prevents LLM-suggested typosquatting or malicious packages.
_ALLOWED_PIP_PACKAGES = frozenset(
    {
        # Base sandbox packages (safe to re-install/upgrade)
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "torch",
        "sympy",
        # Allowed extras
        "xgboost",
        "lightgbm",
        "catboost",
        "statsmodels",
        "networkx",
        "seaborn",
        "matplotlib",
        "plotly",
        "opencv-python-headless",
        "pillow",
        "scikit-image",
        "transformers",
        "datasets",
        "optuna",
        "hyperopt",
        "shap",
        "lime",
    }
)


def rebuild_sandbox_with_packages(topic_id: int, extra_packages: list[str]) -> bool:
    """Build a per-topic sandbox image with extra pip packages.

    - Validates packages against allowlist and regex
    - Generates a Dockerfile extending the base image
    - Builds as orbit-or-sandbox:topic-{topic_id}
    - Restarts the container with the same security flags
    - Updates allowed imports (per-topic)
    """
    if not extra_packages:
        return True

    import tempfile

    # Validate package names against allowlist and regex
    for pkg in extra_packages:
        base_name = re.split(r"[<>=!~\[]", pkg)[0].strip().lower()
        if base_name not in _ALLOWED_PIP_PACKAGES:
            logger.error("[CodeSandbox] Package '%s' not in allowlist", pkg)
            return False
        if not _SAFE_PKG_RE.fullmatch(pkg):
            logger.error("[CodeSandbox] Rejected unsafe package name: %s", pkg)
            return False

    tag = f"orbit-or-sandbox:topic-{topic_id}"
    packages_str = " ".join(extra_packages)

    dockerfile_content = (
        f"FROM {IMAGE_NAME}\nRUN pip install --no-cache-dir {packages_str}\n"
    )

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile_path = Path(tmpdir) / "Dockerfile"
            dockerfile_path.write_text(dockerfile_content)
            subprocess.run(
                ["docker", "build", "-t", tag, tmpdir],
                check=True,
                timeout=600,
            )
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
    ) as exc:
        logger.error(
            "[CodeSandbox] Failed to build per-topic image for topic %d: %s",
            topic_id,
            exc,
        )
        return False

    # Stop old container, start new one
    try:
        subprocess.run(
            ["docker", "rm", "-f", CONTAINER_NAME],
            capture_output=True,
            timeout=30,
        )
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                CONTAINER_NAME,
                "--network=none",
                "--restart=unless-stopped",
                "--memory=2g",
                "--cpus=2",
                "--pids-limit=256",
                "--security-opt=no-new-privileges",
                "--cap-drop=ALL",
                "--read-only",
                "--tmpfs",
                f"/tmp:size=100m,uid={_SANDBOX_UID},gid={_SANDBOX_UID}",
                "--tmpfs",
                f"/workspace/output:size=100m,uid={_SANDBOX_UID},gid={_SANDBOX_UID}",
                tag,
                "sleep",
                "infinity",
            ],
            check=True,
            timeout=30,
        )
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
    ) as exc:
        logger.error(
            "[CodeSandbox] Failed to restart sandbox for topic %d: %s",
            topic_id,
            exc,
        )
        return False

    # Set per-topic extras
    topic_extras: set[str] = set()
    for pkg in extra_packages:
        base_name = pkg.split("=")[0].split("[")[0].strip().replace("-", "_").lower()
        topic_extras.add(base_name)
    _extra_allowed_imports[topic_id] = topic_extras

    logger.info(
        "[CodeSandbox] Per-topic sandbox rebuilt for topic %d with packages: %s",
        topic_id,
        extra_packages,
    )
    return True


MAX_REVIEW_COUNT = (
    2  # After 2 clean reviews, refuse further CODE_REVIEW on this evidence
)

ROLE_CODE_HINTS: dict[str, str] = {
    "analyst": (
        "Specialize in: bootstrap resampling, Monte Carlo simulation, "
        "regression/curve fitting, synthetic dataset generation, "
        "distribution fitting, time series analysis."
    ),
    "scientist": (
        "Specialize in: symbolic math (sympy for derivations/proofs), "
        "model verification, sensitivity analysis, numerical integration, "
        "scaling law verification, dimensional analysis."
    ),
    "engineer": (
        "Specialize in: performance benchmarking, algorithm complexity testing, "
        "system simulation (queuing/throughput), optimization, "
        "cost modeling, trade-off Pareto analysis."
    ),
    "contrarian": (
        "Specialize in: assumption stress-testing, parameter sweeps, "
        "boundary/edge case analysis, counter-example construction, "
        "worst-case scenario modeling, robustness testing."
    ),
}

BLOCKED_PATTERNS = [
    re.compile(r"\bsubprocess\b"),
    re.compile(r"\bos\."),  # block all os.* attribute access
    re.compile(r"\bio\.open\s*\("),  # block io.open() bypass of open() validation
    re.compile(r"\b__import__\b"),
    re.compile(r"\bimportlib\b"),
    # Block builtins that enable dynamic code execution / introspection
    re.compile(r"(?<!\.)\beval\s*\("),
    re.compile(r"(?<!\.)\bexec\s*\("),
    re.compile(r"(?<!\.)\bgetattr\s*\("),
    re.compile(r"(?<!\.)\bsetattr\s*\("),
    re.compile(r"(?<!\.)\bdelattr\s*\("),
    re.compile(r"\b__builtins__\b"),
    re.compile(r"\bbuiltins\b"),
    re.compile(r"\bglobals\s*\("),
    re.compile(r"\bvars\s*\("),
    re.compile(r"\bbreakpoint\s*\("),
    # Block compile() but allow re.compile()
    re.compile(r"(?<!\w\.)(?<!re\.)\bcompile\s*\("),
    # Block class hierarchy / introspection escape chains
    re.compile(r"__subclasses__"),
    re.compile(r"__bases__"),
    re.compile(r"__mro__"),
    re.compile(r"__globals__"),
    re.compile(r"__code__"),
    re.compile(r"__class__"),
    re.compile(r"__traceback__"),
    re.compile(r"_getframe"),
    re.compile(r"f_globals"),
    re.compile(r"f_builtins"),
    re.compile(r"f_locals"),
    # File / network / dangerous modules
    # open() is allowed only for /workspace/output/ — checked separately in validate_source
    # re.compile(r"\bopen\s*\("),  # removed: sandbox is isolated, open() validated below
    re.compile(r"\bshutil\b"),
    re.compile(r"\bsocket\b"),
    re.compile(r"\brequests\b"),
    re.compile(r"\bhttpx\b"),
    re.compile(r"\burllib\b"),
    re.compile(r"\bctypes\b"),
    re.compile(r"\bpickle\b"),
    # Block sys.modules and sys.path manipulation
    re.compile(r"\bsys\.modules\b"),
    re.compile(r"\bsys\.path\b"),
    # Block sys functions that enable DoS or introspection
    # sys.exit() is harmless in sandbox (container process exits) — allowed
    re.compile(r"\bsys\.setrecursionlimit\s*\("),
    re.compile(r"\bsys\.settrace\s*\("),
    re.compile(r"\bsys\.setprofile\s*\("),
    re.compile(r"\bsys\.stdin\b"),
    # Block locals() — in module scope it returns globals, giving __builtins__ access
    re.compile(r"\blocals\s*\("),
    # Block pathlib file I/O — bypasses the open() check entirely
    re.compile(r"\.read_text\s*\("),
    re.compile(r"\.read_bytes\s*\("),
    re.compile(r"\.write_text\s*\("),
    re.compile(r"\.write_bytes\s*\("),
    # Block open() aliasing — prevents `o = open; o(path)` bypass
    re.compile(r"=\s*open\b(?!\s*\()"),
    # Block open as function argument — prevents `map(open, [...])`, `[open][0](...)`
    re.compile(r"(?:map|filter|sorted|reduce|apply)\s*\(\s*open\b"),
    re.compile(r"\[\s*open\s*\]"),
]

# open() validation patterns (module-level for performance)
_SAFE_OPEN_RE = re.compile(r"""\bopen\s*\(\s*f?['"](?:/workspace/output/|/tmp/)""")
_ANY_OPEN_RE = re.compile(r"""\bopen\s*\(""")
_PATH_TRAVERSAL_RE = re.compile(r"""\bopen\s*\(.*\.\.""")

# Concurrency guard: limit parallel sandbox executions
_SANDBOX_SEMAPHORE = asyncio.Semaphore(2)

RESULTS_EPILOGUE = """
import json as _json, sys as _sys, pathlib as _pathlib
try:
    _imgs = [str(p) for p in _pathlib.Path("/workspace/output").glob("*.png")]
    _results = {"status": "ok", "images": _imgs}
    _sys.stdout.write("\\n__RESULTS_JSON__=" + _json.dumps(_results))
except Exception:
    pass
"""

# Environment variables injected into docker exec for clean runtime
_SANDBOX_ENV = [
    "-e",
    "MPLCONFIGDIR=/tmp",
    "-e",
    "PYTHONDONTWRITEBYTECODE=1",
]

# Where to store extracted images on the host (inside project)
_HOST_OUTPUT_DIR = Path(__file__).parent / ".." / ".." / "sandbox_output"

# Minimum meaningful source length (chars) — guards against empty LLM output
_MIN_SOURCE_LEN = 10


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ExperimentResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    results_json: Optional[dict] = None
    output_images: list[str] = field(default_factory=list)
    execution_time_s: float = 0.0
    success: bool = False


@dataclass(frozen=True)
class CodeEvidenceItem:
    hypothesis: str
    source_code: str
    stdout: str
    stderr: str
    exit_code: int
    execution_time_s: float
    success: bool
    iterations: int
    rendered_results: str
    code_evidence_id: int = 0
    planning_veto: bool = False


@dataclass(frozen=True)
class PhaseRunResult:
    phase: str
    source_code: str
    result: ExperimentResult
    attempts: int


# ---------------------------------------------------------------------------
# Validation helpers adapted from an earlier experiment runner.
# ---------------------------------------------------------------------------


def validate_source(source: str) -> tuple[bool, str]:
    """Validate Python source code for safety. Returns (is_valid, reason)."""
    if len(source.strip()) < _MIN_SOURCE_LEN:
        return False, "Source is empty or trivially short"

    lines = source.strip().splitlines()
    if len(lines) > MAX_SOURCE_LINES:
        return (
            False,
            f"Source exceeds {MAX_SOURCE_LINES} line limit ({len(lines)} lines)",
        )

    for pattern in BLOCKED_PATTERNS:
        match = pattern.search(source)
        if match:
            return False, f"Blocked pattern detected: {match.group()}"

    # Validate open() calls: only /workspace/output/ and /tmp/ paths allowed
    for line_no, line in enumerate(lines, 1):
        code_part = line.split("#")[0]
        if _ANY_OPEN_RE.search(code_part):
            if not _SAFE_OPEN_RE.search(code_part):
                return (
                    False,
                    f"open() on line {line_no} must target /workspace/output/ or /tmp/",
                )
            if _PATH_TRAVERSAL_RE.search(code_part):
                return (
                    False,
                    f"open() on line {line_no} contains path traversal (..)",
                )

    allowed = get_allowed_imports()
    for line in lines:
        for stmt in line.split(";"):
            stripped = stmt.split("#")[0].strip()
            if not stripped:
                continue
            if stripped.startswith("from "):
                parts = stripped.split()
                if len(parts) >= 2:
                    module = parts[1].split(".")[0]
                    if module and module not in allowed:
                        return False, f"Import not in allowlist: {module}"
            elif stripped.startswith("import "):
                rest = stripped[len("import ") :]
                for token in rest.split(","):
                    token = token.strip()
                    if not token:
                        continue
                    module = token.split(".")[0].split()[0]
                    if module and module not in allowed:
                        return False, f"Import not in allowlist: {module}"

    return True, "ok"


def _default_environment_description() -> str:
    return (
        f"# ORBIT Sandbox Environment\n"
        f"- Metadata path: {ENVIRONMENT_DESCRIPTION_PATH}\n"
        f"- Runtime: Python 3.12-slim\n"
        f"- Network: disabled\n"
        f"- Root filesystem: read-only\n"
        f"- Writable paths: /tmp, /workspace/output\n"
        f"- Resource limits: 2 CPU, 2 GB RAM, 256 PIDs\n"
        f"- Script contract: single-file Python via stdin; smoke/full mode arrives in sys.argv[1]\n"
        f"- Timeouts: smoke={SMOKE_TIMEOUT}s, full={FULL_TIMEOUT}s\n"
    )


def _load_sandbox_environment_description() -> str:
    if not is_sandbox_ready():
        return _default_environment_description()

    try:
        result = subprocess.run(
            ["docker", "exec", CONTAINER_NAME, "cat", ENVIRONMENT_DESCRIPTION_PATH],
            capture_output=True,
            timeout=10,
        )
        if result.returncode == 0:
            text = result.stdout.decode("utf-8", errors="replace").strip()
            if text:
                return text
    except Exception as exc:
        logger.debug(
            "[CodeSandbox] Failed to load env description from container: %s", exc
        )

    return _default_environment_description()


async def _get_sandbox_environment_description() -> str:
    return await asyncio.to_thread(_load_sandbox_environment_description)


# ---------------------------------------------------------------------------
# Sandbox execution helpers adapted from an earlier experiment runner.
# ---------------------------------------------------------------------------


def _inject_results_epilogue(source: str) -> str:
    return source.rstrip() + "\n" + RESULTS_EPILOGUE


def _parse_results_json(stdout: str) -> Optional[dict]:
    marker = "__RESULTS_JSON__="
    idx = stdout.rfind(marker)
    if idx < 0:
        return None
    try:
        return json.loads(stdout[idx + len(marker) :].strip().splitlines()[0])
    except (json.JSONDecodeError, IndexError):
        return None


def is_sandbox_ready() -> bool:
    """Check whether the orbit-or sandbox image exists and container is running."""
    if not shutil.which("docker"):
        return False
    try:
        # Check that the image exists
        result = subprocess.run(
            ["docker", "image", "inspect", IMAGE_NAME],
            capture_output=True,
            timeout=10,
        )
        if result.returncode != 0:
            return False
        # Check that the container is actually running (not just exists/exited)
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER_NAME],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0 and b"true" in result.stdout.lower()
    except Exception:
        return False


def ensure_sandbox() -> bool:
    """Build the sandbox image and refresh the running container."""
    sandbox_dir = Path(__file__).parent / "sandbox"
    dockerfile = sandbox_dir / "Dockerfile"
    if not dockerfile.exists():
        logger.warning("[CodeSandbox] Dockerfile not found at %s", dockerfile)
        return False

    # Always build so Dockerfile changes are picked up on normal server startup.
    try:
        subprocess.run(
            ["docker", "build", "-t", IMAGE_NAME, str(sandbox_dir)],
            check=True,
            timeout=600,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        logger.error("[CodeSandbox] Failed to build sandbox image: %s", exc)
        return False

    # Recreate container so the running sandbox matches the latest image.
    try:
        result = subprocess.run(
            ["docker", "inspect", CONTAINER_NAME],
            capture_output=True,
            timeout=10,
        )
        if result.returncode == 0:
            subprocess.run(
                ["docker", "rm", "-f", CONTAINER_NAME],
                check=True,
                timeout=30,
            )
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                CONTAINER_NAME,
                "--network=none",
                "--restart=unless-stopped",
                "--memory=2g",
                "--cpus=2",
                "--pids-limit=256",
                "--security-opt=no-new-privileges",
                "--cap-drop=ALL",
                "--read-only",
                "--tmpfs",
                f"/tmp:size=100m,uid={_SANDBOX_UID},gid={_SANDBOX_UID}",
                "--tmpfs",
                f"/workspace/output:size=100m,uid={_SANDBOX_UID},gid={_SANDBOX_UID}",
                IMAGE_NAME,
                "sleep",
                "infinity",
            ],
            check=True,
            timeout=30,
        )
    except Exception as exc:
        logger.error("[CodeSandbox] Failed to start sandbox container: %s", exc)
        return False

    return True


async def _execute_in_sandbox(
    source: str,
    timeout: int = DEFAULT_TIMEOUT,
    mode: Optional[str] = None,
) -> ExperimentResult:
    """Execute source code in the Docker sandbox container.

    Guarded by _SANDBOX_SEMAPHORE to limit concurrent Docker exec processes.
    """
    async with _SANDBOX_SEMAPHORE:
        return await _execute_in_sandbox_inner(source, timeout=timeout, mode=mode)


async def _execute_in_sandbox_inner(
    source: str,
    timeout: int = DEFAULT_TIMEOUT,
    mode: Optional[str] = None,
) -> ExperimentResult:
    """Inner execution — called under the semaphore."""
    sandbox_ok = await asyncio.to_thread(is_sandbox_ready)
    if not sandbox_ok:
        return ExperimentResult(
            stderr="Docker sandbox not available",
            exit_code=-1,
            success=False,
        )

    # Clean output dir before each run to avoid stale images
    clean_proc = await asyncio.create_subprocess_exec(
        "docker",
        "exec",
        CONTAINER_NAME,
        "sh",
        "-c",
        "rm -f /workspace/output/*.png && rm -rf /tmp/*",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await clean_proc.wait()

    full_source = _inject_results_epilogue(source)

    try:
        exec_args = [
            "docker",
            "exec",
            "-i",
            *_SANDBOX_ENV,
            CONTAINER_NAME,
            "python3",
            "-",
        ]
        if mode:
            exec_args.append(mode)
        proc = await asyncio.create_subprocess_exec(
            *exec_args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        start = time.monotonic()
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(input=full_source.encode("utf-8")),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            # Drain buffered stdout so partial grid results can be salvaged
            partial_stdout = ""
            partial_stderr = "Execution timed out"
            try:
                out_bytes, err_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=5
                )
                partial_stdout = (out_bytes or b"").decode("utf-8", errors="replace")
                if err_bytes:
                    partial_stderr += "\n" + err_bytes.decode("utf-8", errors="replace")
            except Exception:
                pass
            await proc.wait()
            return ExperimentResult(
                stdout=partial_stdout,
                stderr=partial_stderr,
                exit_code=-1,
                execution_time_s=timeout,
                success=False,
            )
        elapsed = time.monotonic() - start

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        exit_code = proc.returncode
        results_json = _parse_results_json(stdout)

        # Extract output images from container (if any were generated)
        output_images = await _extract_output_images()

        return ExperimentResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            results_json=results_json,
            output_images=output_images,
            execution_time_s=elapsed,
            success=(exit_code == 0),
        )
    except Exception as exc:
        return ExperimentResult(
            stderr=f"Execution error: {exc}",
            exit_code=-1,
            success=False,
        )


async def _extract_output_images() -> list[str]:
    """Copy .png files from the container's output dir to the host.

    Does NOT trust RESULTS_JSON paths — lists the fixed output directory
    directly to prevent path traversal.
    """
    # List actual files in the sandboxed output dir
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            CONTAINER_NAME,
            "find",
            "/workspace/output",
            "-maxdepth",
            "1",
            "-name",
            "*.png",
            "-type",
            "f",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        if proc.returncode != 0 or not out:
            return []
    except Exception:
        return []

    container_files = [
        line.strip()
        for line in out.decode("utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    if not container_files:
        return []

    _HOST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []
    for cpath in container_files[:5]:  # cap at 5 images
        # Extra safety: must be under /workspace/output/ with a simple filename
        fname = Path(cpath).name
        if not fname.endswith(".png") or "/" in fname or ".." in fname:
            continue
        host_path = _HOST_OUTPUT_DIR / f"{int(time.time())}_{fname}"
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "exec",
                CONTAINER_NAME,
                "cat",
                f"/workspace/output/{fname}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            data, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
            if proc.returncode == 0 and data:
                host_path.write_bytes(data)
                extracted.append(str(host_path))
                logger.info("[CodeSandbox] Extracted image: %s", host_path)
        except Exception as exc:
            logger.warning("[CodeSandbox] Failed to extract %s: %s", fname, exc)
    return extracted


async def analyze_output_images(images: list[str], hypothesis: str) -> str:
    """Use MiniMax vision to describe generated plots. Returns text description."""
    if not images:
        return ""
    descriptions: list[str] = []
    try:
        from .minimax_client import query_minimax_understand_image

        for img_path in images[:3]:  # cap at 3
            desc = await query_minimax_understand_image(
                prompt=f"Describe this chart concisely. Does it support or refute the hypothesis: {hypothesis}",
                image_source=img_path,
            )
            if desc:
                descriptions.append(f"[Chart: {Path(img_path).name}] {desc}")
    except ImportError:
        logger.info(
            "[CodeSandbox] MiniMax image understanding not available, skipping chart analysis"
        )
    except Exception as exc:
        logger.warning("[CodeSandbox] Chart analysis failed: %s", exc)
    return "\n".join(descriptions)


# ---------------------------------------------------------------------------
# Tier 1: Calculator — no LLM, inline result
# ---------------------------------------------------------------------------

# Safe AST node types for Tier 1 calculator expressions.
# Only arithmetic, comparisons, basic data structures, and whitelisted function calls.
_CALC_SAFE_NODES: frozenset[type] = frozenset(
    {
        ast.Expression,
        ast.Constant,  # literals
        ast.UnaryOp,
        ast.UAdd,
        ast.USub,
        ast.Not,
        ast.Invert,  # unary ops (incl ~)
        ast.BinOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.LShift,
        ast.RShift,
        ast.BitOr,
        ast.BitXor,
        ast.BitAnd,
        ast.BoolOp,
        ast.And,
        ast.Or,  # boolean ops
        ast.Compare,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.IfExp,  # ternary
        ast.Tuple,
        ast.List,  # containers (no Dict to avoid {}.keys() chains)
        ast.Load,  # context
        ast.keyword,  # keyword arguments in function calls (e.g., round(3.14, ndigits=2))
    }
)
# Add deprecated ast.Num/ast.Str if present (Python <3.12 compat)
for _compat_name in ("Num", "Str"):
    _compat_cls = getattr(ast, _compat_name, None)
    if _compat_cls is not None:
        _CALC_SAFE_NODES = _CALC_SAFE_NODES | {_compat_cls}

# Functions allowed in calc expressions (no side effects, pure math).
_CALC_SAFE_CALLS = frozenset(
    {
        "abs",
        "round",
        "min",
        "max",
        "sum",
        "len",
        "int",
        "float",
        "bool",
        "pow",
        "divmod",
        "math.sqrt",
        "math.log",
        "math.log2",
        "math.log10",
        "math.exp",
        "math.sin",
        "math.cos",
        "math.tan",  # math.pi/math.e are attrs, covered by Attribute branch
        "math.ceil",
        "math.floor",
        "math.factorial",
        "math.gcd",
        "sqrt",
        "log",
        "log2",
        "log10",
        "exp",
        "sin",
        "cos",
        "tan",
        "ceil",
        "floor",
        "factorial",
        "gcd",
        "statistics.mean",
        "statistics.median",
        "statistics.stdev",
        "mean",
        "median",
        "stdev",
    }
)


def validate_calc_expression(expression: str) -> tuple[bool, str]:
    """Validate a CALC expression using AST whitelist — much safer than blocklist.

    Returns (is_valid, reason). Only allows arithmetic, comparisons, and
    whitelisted function calls (math/statistics builtins).
    """
    expr = expression.strip()
    if not expr:
        return False, "Empty expression"
    if len(expr) > 500:
        return False, "Expression too long (max 500 chars)"

    # Auto-fix common LLM math notation: ^ for exponentiation
    if "^" in expr:
        expr = expr.replace("^", "**")

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        return False, f"Invalid expression syntax: {e}"

    for node in ast.walk(tree):
        node_type = type(node)

        # Allow safe node types
        if node_type in _CALC_SAFE_NODES:
            continue

        # Allow Name nodes for module refs and safe builtin function names
        if node_type is ast.Name:
            if node.id in ("math", "statistics", "True", "False", "None"):
                continue
            # Allow names that are safe callable builtins (they appear as Call.func too)
            if node.id in _CALC_SAFE_CALLS:
                continue
            return False, f"Variable not allowed in calc: {node.id}"

        # Allow Attribute access for module.func patterns
        if node_type is ast.Attribute:
            if isinstance(node.value, ast.Name) and node.value.id in (
                "math",
                "statistics",
            ):
                continue
            return False, f"Attribute access not allowed in calc: {ast.dump(node)}"

        # Allow Call but only for whitelisted functions
        if node_type is ast.Call:
            func = node.func
            if isinstance(func, ast.Name) and func.id in _CALC_SAFE_CALLS:
                continue
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                full = f"{func.value.id}.{func.attr}"
                if full in _CALC_SAFE_CALLS:
                    continue
            return False, f"Function call not allowed in calc: {ast.dump(func)}"

        # Allow starred in function args (e.g., max(*[1,2,3]))
        if node_type is ast.Starred:
            continue

        return False, f"Expression node not allowed in calc: {node_type.__name__}"

    return True, "ok"


async def run_calc(
    expression: str,
    *,
    topic_id: int,
    subtopic_id: int,
    role: str,
) -> CodeEvidenceItem:
    """Tier 1: wrap an expression in print(), validate, execute with short timeout.

    No LLM call, no fix loop, no self-review, no FactCandidate.
    Uses AST whitelist validation — only safe arithmetic and whitelisted functions.
    """
    # AST-level validation: only allow safe expression nodes
    valid, reason = validate_calc_expression(expression)
    if not valid:
        rendered = _render_code_evidence(
            hypothesis=f"CALC: {expression}",
            success=False,
            iterations=0,
            execution_time_s=0,
            stdout="",
            stderr=f"Calc rejected: {reason}",
        )
        return CodeEvidenceItem(
            hypothesis=f"CALC: {expression}",
            source_code=f"print({expression})",
            stdout="",
            stderr=f"Calc rejected: {reason}",
            exit_code=-1,
            execution_time_s=0.0,
            success=False,
            iterations=0,
            rendered_results=rendered,
        )

    # Auto-inject imports for whitelisted modules used in the expression
    imports: list[str] = []
    if "math." in expression or expression.strip().startswith("math"):
        imports.append("import math")
    if "statistics." in expression or expression.strip().startswith("statistics"):
        imports.append("import statistics")
    source = "\n".join(imports + [f"print({expression})"])
    valid, reason = validate_source(source)
    if not valid:
        rendered = _render_code_evidence(
            hypothesis=f"CALC: {expression}",
            success=False,
            iterations=0,
            execution_time_s=0,
            stdout="",
            stderr=reason,
        )
        return CodeEvidenceItem(
            hypothesis=f"CALC: {expression}",
            source_code=source,
            stdout="",
            stderr=reason,
            exit_code=-1,
            execution_time_s=0.0,
            success=False,
            iterations=0,
            rendered_results=rendered,
        )

    result = await _execute_in_sandbox(source, timeout=CALC_TIMEOUT)

    # Strip the RESULTS_EPILOGUE marker from calc stdout
    calc_stdout = result.stdout
    marker = "__RESULTS_JSON__="
    marker_idx = calc_stdout.find(marker)
    if marker_idx >= 0:
        calc_stdout = calc_stdout[:marker_idx].rstrip()

    rendered = _render_code_evidence(
        hypothesis=f"CALC: {expression}",
        success=result.success,
        iterations=1,
        execution_time_s=result.execution_time_s,
        stdout=calc_stdout,
        stderr=result.stderr,
        exit_code=result.exit_code,
    )

    return CodeEvidenceItem(
        hypothesis=f"CALC: {expression}",
        source_code=source,
        stdout=calc_stdout,
        stderr=result.stderr,
        exit_code=result.exit_code,
        execution_time_s=result.execution_time_s,
        success=result.success,
        iterations=1,
        rendered_results=rendered,
        code_evidence_id=0,
    )


# ---------------------------------------------------------------------------
# Code generation and fixing via LLM
# ---------------------------------------------------------------------------


async def _plan_experiment(
    hypothesis: str,
    context: str,
    env_desc: str,
    *,
    role: str = "",
    provider: str = "minimax",
) -> str:
    """Plan an experiment before writing code. Returns a structured plan."""
    role_hint = ROLE_CODE_HINTS.get(role, "")
    role_line = f"Your specialization: {role_hint}\n" if role_hint else ""
    prompt = (
        f"Plan a computational experiment to test this hypothesis:\n\n"
        f"{hypothesis}\n\n"
        f"Workspace context:\n{context}\n\n"
        "Exact sandbox environment facts:\n"
        f"{env_desc[:3000]}\n\n"
        f"{role_line}"
        "CRITICAL FEASIBILITY CHECK:\n"
        "1. You are running in an isolated Docker sandbox. The ONLY available files and paths are listed in the environment facts above.\n"
        "2. If your hypothesis requires a specific external dataset (e.g., a benchmark dataset or history DB) that is NOT explicitly listed in the environment facts, you MUST reject the experiment as unfeasible.\n"
        "3. DO NOT plan to 'mock' or 'synthesize' massive 10,000-line text datasets just because the real data is missing. Massive text generation exceeds sandbox code length limits.\n"
        "4. If feasible, design a tight, self-contained experiment using small math/numpy mock data or standard sklearn toy datasets.\n\n"
        "Output a concise experiment plan in JSON:\n"
        "{\n"
        '  "feasible": true or false,\n'
        '  "rejection_reason": "If feasible is false, explain exactly what data/resource is missing. If true, leave empty.",\n'
        '  "approach": "1-2 sentence summary of the experimental method",\n'
        '  "data": "what data to generate or use (datasets, sizes, parameters)",\n'
        '  "models": ["list of models/configs to compare"],\n'
        '  "metrics": ["list of metrics to compute"],\n'
        '  "statistical_test": "how to determine significance (e.g., paired bootstrap, t-test)",\n'
        '  "smoke_strategy": "what the cheap smoke test should verify",\n'
        '  "scope": "single (1 dataset, 1 seed, 1 config) or grid (multiple seeds/sizes/datasets)",\n'
        f'  "scope_note": "single-config experiments have {FULL_TIMEOUT}s timeout. '
        f'Grid sweeps require [CODE_VERIFY_GRID] and get {GRID_TIMEOUT}s. Plan for single unless grid is essential.",\n'
        '  "expected_outcome": "what result would support vs refute the hypothesis",\n'
        '  "conclusion_template": "CONCLUSION: [metric] shows [result], therefore hypothesis is [supported/refuted]"\n'
        "}\n"
    )
    text = await call_text(
        prompt,
        system_instruction="You are a scientific experiment planner. Output only valid JSON.",
        provider=provider,
        fallback_role="code_sandbox",
        temperature=0.3,
        max_tokens=DEFAULT_MAX_TOKENS,
        require_json=True,
    )
    return (text or "").strip()


async def _generate_code(
    hypothesis: str,
    context: str,
    *,
    role: str = "",
    plan: str = "",
    provider: str = "minimax",
) -> str:
    """Generate experiment code via the configured provider."""
    allowed = ", ".join(sorted(get_allowed_imports()))
    env_desc = await _get_sandbox_environment_description()
    role_hint = ROLE_CODE_HINTS.get(role, "")
    role_line = f"- Role specialization: {role_hint}\n" if role_hint else ""
    plan_block = f"Experiment plan (follow this closely):\n{plan}\n\n" if plan else ""
    prompt = (
        f"Write a self-contained Python script to test this hypothesis:\n\n"
        f"{hypothesis}\n\n"
        f"{plan_block}"
        f"Workspace context:\n{context}\n\n"
        "Exact sandbox environment facts:\n"
        f"{env_desc[:3000]}\n\n"
        "Requirements:\n"
        "- Single file only\n"
        f"- Use only these imports: {allowed}\n"
        "- Print all results to stdout as numbers/text — stdout IS the evidence\n"
        "- MUST print a CONCLUSION section at the end that directly answers the hypothesis with computed comparisons "
        "(e.g., 'CONCLUSION: MLP beats RF on 3/5 datasets, loses on 2/5. Beat-both rate: 40%.'). "
        "Do NOT leave interpretation to the reader — compute and print the verdict.\n"
        "- Do NOT use tqdm, progress bars, or verbose per-iteration logging. Print only final results and key intermediate summaries.\n"
        "- Do NOT generate plots or import matplotlib unless the hypothesis specifically requires visual pattern analysis\n"
        f"- Keep under {MAX_SOURCE_LINES} lines\n"
        "- No network access\n"
        "- Do NOT use os module, subprocess, eval, exec, or any blocked builtins\n"
        "- Do NOT import warnings or call warnings.filterwarnings(). Keep warnings visible.\n"
        "- You MAY use open() ONLY to write results to /workspace/output/ (e.g., open('/workspace/output/results.json', 'w'))\n"
        f"- The sandbox will execute the script as `python3 - smoke` and then `python3 - full`\n"
        "- Use `sys.argv[1]` to switch modes; default to `full` if the argument is missing\n"
        f"- Smoke mode must finish within {SMOKE_TIMEOUT}s and should do the cheapest end-to-end verification possible\n"
        f"- Full mode must finish within {FULL_TIMEOUT}s — SINGLE CONFIGURATION ONLY.\n"
        "  ONE dataset, ONE set of hyperparameters, ONE random seed. Do NOT run grid search,\n"
        "  parameter sweeps, or multi-seed loops in Full mode. If you need multi-config\n"
        "  experiments, the agent should use [CODE_VERIFY_GRID] instead.\n"
        "  If your experiment times out, your scope is too large — reduce it.\n"
        "- Prefer clear helper functions such as load_data(), run_smoke(), run_full(), and main() instead of one long top-level script\n"
        f"{role_line}"
        "Output only the Python code, no markdown fences."
    )
    text = await call_text(
        prompt,
        system_instruction="You are a scientific computing code generator. Output only valid Python code.",
        provider=provider,
        fallback_role="code_sandbox",
        temperature=0.3,
        max_tokens=DEFAULT_MAX_TOKENS,
    )
    stripped = (text or "").strip()
    stripped = re.sub(r"^```(?:python)?\s*\n", "", stripped)
    stripped = re.sub(r"\n```\s*$", "", stripped)
    return stripped.strip()


async def _fix_code(
    source: str,
    stderr: str,
    hypothesis: str,
    *,
    phase: str = "",
    plan: str = "",
    provider: str = "minimax",
) -> str:
    """Ask LLM to fix code based on error output."""
    # Pass specific validation errors to the LLM so it can fix them precisely.
    # Only hide blocked-pattern regex details to avoid teaching the LLM what
    # patterns are checked; keep import-related errors verbatim.
    safe_stderr = stderr[:2000]
    if "Blocked pattern detected:" in safe_stderr:
        safe_stderr = "Source code failed security validation. Use only standard math/data operations."
    elif "Validation error:" in safe_stderr:
        # Keep the precise error (e.g. "Import not in allowlist: warnings")
        pass
    env_desc = await _get_sandbox_environment_description()
    phase_note = ""
    if phase:
        phase_timeouts = {
            "smoke": SMOKE_TIMEOUT,
            "grid_smoke": GRID_SMOKE_TIMEOUT,
            "grid": GRID_TIMEOUT,
        }
        phase_timeout = phase_timeouts.get(phase, FULL_TIMEOUT)
        mode_contract = (
            "smoke/grid" if phase in ("grid", "grid_smoke") else "smoke/full"
        )
        phase_note = (
            f"The failing phase was `{phase}` mode.\n"
            f"- Preserve the single-file script and the `sys.argv[1]` {mode_contract} contract\n"
            f"- `{phase}` mode must complete within {phase_timeout}s\n"
        )
    plan_block = f"Experiment plan (follow this closely):\n{plan}\n\n" if plan else ""
    prompt = (
        f"The following Python script failed when testing the hypothesis:\n"
        f"{hypothesis}\n\n"
        f"{plan_block}"
        f"{phase_note}"
        "Exact sandbox environment facts:\n"
        f"{env_desc[:3000]}\n\n"
        f"Source code:\n```python\n{source}\n```\n\n"
        f"Error output:\n```\n{safe_stderr}\n```\n\n"
        "Fix the code. Use only numpy, scipy, pandas, matplotlib, sklearn, torch, math, "
        "random, statistics for computation. Do NOT use os, subprocess, eval, exec, "
        "or any introspection like __globals__, __subclasses__, __class__. "
        "Do NOT import warnings or call warnings.filterwarnings(). Keep warnings visible. "
        "You MAY use open() ONLY for /workspace/output/ paths. "
        f"Keep the script under {MAX_SOURCE_LINES} lines.\n"
        "Output only the corrected Python code, no markdown fences."
    )
    text = await call_text(
        prompt,
        system_instruction="You are a Python debugging expert. Output only valid Python code.",
        provider=provider,
        fallback_role="code_sandbox",
        temperature=0.3,
        max_tokens=DEFAULT_MAX_TOKENS,
    )
    stripped = (text or "").strip()
    stripped = re.sub(r"^```(?:python)?\s*\n", "", stripped)
    stripped = re.sub(r"\n```\s*$", "", stripped)
    return stripped.strip()


def _review_smoke_output(stdout: str, source: str) -> tuple[bool, str]:
    """Check smoke output format: must have CONCLUSION section and not be empty."""
    if not stdout or not stdout.strip():
        return False, "Smoke produced no stdout output"
    # Check for CONCLUSION section (case-insensitive)
    if "CONCLUSION" not in stdout.upper():
        return False, (
            "Missing CONCLUSION section in output. The script MUST print a "
            "'CONCLUSION:' section at the end that directly answers the hypothesis "
            "with computed comparisons. Add a print('CONCLUSION: ...') statement."
        )
    return True, ""


async def _review_code_logic(
    hypothesis: str,
    source: str,
    stdout: str,
    *,
    phase: str = "full",
    provider: str = "minimax",
) -> tuple[bool, str]:
    """Quick LLM self-check: is the code logic sound for this hypothesis?

    Returns (is_sound, feedback). If the logic has issues, feedback explains
    what to fix — this feeds back into the generate→fix loop.
    """
    prompt = (
        f"Hypothesis being tested: {hypothesis}\n\n"
        f"Execution phase: {phase}\n\n"
        f"Python code:\n```python\n{source}\n```\n\n"
        f"Execution output:\n{stdout[:32768]}\n\n"
        "Review this code for correctness AND hypothesis alignment.\n"
        "Check:\n"
        "1. Correct formula, appropriate statistical test, no data fabrication\n"
        "2. CRITICAL — Hypothesis-output alignment: does the output DIRECTLY answer the hypothesis?\n"
        "   - If the hypothesis claims to test a specific criterion (e.g., 'beat both', 'outperform'), "
        "the code MUST compute and print that specific comparison, not just raw metrics.\n"
        "   - Printing individual model scores without computing the claimed comparison is NOT sound.\n"
        "   - IMPORTANT: If the code correctly computes the metrics and the result REFUTES the hypothesis, this is SOUND SCIENCE. Do NOT fail the logic review for a refuted hypothesis.\n"
        "3. No trivial bugs (wrong variable, off-by-one, missing import)\n"
        'Reply with strict JSON: {"sound": true} or {"sound": false, "issue": "what is wrong"}'
    )
    try:
        resp = await call_text(
            prompt,
            system_instruction="You are a code review expert. Reply with JSON only.",
            provider=provider,
            fallback_role="code_sandbox",
            temperature=0.2,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        parsed = json.loads(resp.strip())
        if parsed.get("sound"):
            return True, ""
        return False, parsed.get("issue", "Code logic issue detected")
    except Exception:
        # On failure, assume sound — don't block the pipeline
        return True, ""


def _classify_failure(
    exit_code: int, stderr: str, stdout: str, *, is_grid: bool = False
) -> str:
    """Classify a code evidence failure for agents.

    Returns a human-readable failure category so agents can distinguish
    'experiment code was broken' from 'hypothesis was disproved'.
    """
    stderr_lower = (stderr or "").lower()
    # Check timeout FIRST — timeout also sets exit_code=-1
    if "timed out" in stderr_lower or "timeout" in stderr_lower:
        if is_grid:
            return "TIMEOUT_GRID (grid experiment exceeded 1200s — check partial results in stdout, do NOT blindly retry)"
        return "TIMEOUT_REDUCE_SCOPE (experiment too large for this mode — reduce to single config/seed, or use [CODE_VERIFY_GRID] for multi-config sweeps)"
    if "rejected during planning" in stderr_lower:
        return "FEASIBILITY_REJECTED (experiment requires unavailable resources or data — abandon this specific technical path)"
    if exit_code == -1:
        return "CODE_INVALID (code failed security/syntax validation — rewrite needed)"
    if "logic review failed" in stderr_lower:
        return (
            "LOGIC_ERROR (code ran but conclusion is unreliable — review methodology)"
        )
    if (
        "missing conclusion" in stderr_lower
        or "smoke format check failed" in stderr_lower
    ):
        return "MISSING_CONCLUSION (code ran but did not print a CONCLUSION section — rewrite needed)"
    if exit_code != 0:
        return "RUNTIME_ERROR (code crashed during execution — fix the bug and retry)"
    return "UNKNOWN_FAILURE (inspect stderr for details)"


_CODE_POLLUTION_RE = re.compile(r"\b(import |from \w+ import |def \w+\(|class \w+[:(])")


def _clean_hypothesis(raw: str) -> str:
    """Strip code pollution from hypothesis, returning a clean short description.

    Agents sometimes paste full source code into [CODE_VERIFY: ...].
    This extracts just the human-readable hypothesis name/description.
    """
    if not raw:
        return raw
    # If hypothesis contains code indicators, take everything before the code starts
    if _CODE_POLLUTION_RE.search(raw):
        # Try splitting on common separators
        for sep in (" — ", " -- ", "\n#", "\nimport ", "\nfrom ", "\ndef ", "\nclass "):
            idx = raw.find(sep)
            if idx > 0:
                raw = raw[:idx].strip()
                break
        else:
            # No separator found — first line is likely code itself
            first_line = raw.split("\n")[0].strip()
            if _CODE_POLLUTION_RE.match(first_line):
                # Entire hypothesis is code — no recoverable name
                return "Unnamed experiment (CE)"
            raw = first_line
    # If still contains code after cleanup, fall back
    if _CODE_POLLUTION_RE.match(raw):
        return "Unnamed experiment (CE)"
    # Strip trailing separators
    return raw.rstrip(" —-")


def _extract_conclusion(stdout: str) -> str:
    """Extract first meaningful line from CONCLUSION section of stdout."""
    if not stdout:
        return ""
    upper = stdout.upper()
    idx = upper.find("CONCLUSION")
    if idx < 0:
        # Try HYPOTHESIS VERDICT
        idx = upper.find("HYPOTHESIS VERDICT")
    if idx < 0:
        return ""
    rest = stdout[idx:]
    lines = rest.split("\n")
    # Check if conclusion text is on the same line as the header (e.g. "CONCLUSION: text")
    header = lines[0]
    for prefix in ("CONCLUSION:", "CONCLUSION -", "HYPOTHESIS VERDICT:"):
        if prefix in header.upper():
            after = header[header.upper().index(prefix) + len(prefix) :].strip()
            if after and len(after) > 10:
                return after
            break
    # Otherwise scan subsequent lines
    for line in lines[1:]:
        stripped = line.strip().strip("=-")
        if stripped and len(stripped) > 10:
            return stripped
    return ""


def _build_summary(
    hypothesis: str,
    success: bool,
    iterations: int,
    execution_time_s: float,
    phase: str | None,
    exit_code: int,
    stderr: str,
    stdout: str,
) -> str:
    """Build a structured, human-readable summary for CodeEvidence.

    No truncation, no LLM — pure rule-based extraction from structured data.
    """
    clean_hyp = _clean_hypothesis(hypothesis)
    status = "EXECUTION_SUCCESS" if success else "EXECUTION_ERROR"
    phase_note = f" {phase.upper()}" if phase else ""

    parts = [
        f"[{status}{phase_note}] {clean_hyp} | {iterations} iter, {execution_time_s:.1f}s"
    ]

    if success:
        conclusion = _extract_conclusion(stdout)
        if conclusion:
            parts.append(f"Finding: {conclusion}")
    else:
        is_grid = phase in ("grid", "grid_smoke") if phase else False
        category = _classify_failure(exit_code, stderr, stdout, is_grid=is_grid)
        # Extract just the category name, not the full description
        cat_short = category.split("(")[0].strip()
        parts.append(f"Failure: {cat_short}")

    return "\n".join(parts)


def _render_code_evidence(
    hypothesis: str,
    success: bool,
    iterations: int,
    execution_time_s: float,
    stdout: str,
    stderr: str,
    phase: Optional[str] = None,
    exit_code: int = -1,
) -> str:
    """Format a code evidence result for RAG injection."""
    status = "PASSED" if success else "FAILED"
    phase_note = f" during {phase.upper()}" if phase else ""
    lines = [
        f"Code Verification {status}{phase_note} ({iterations} iteration(s), {execution_time_s:.1f}s):",
        f"Hypothesis: {hypothesis}",
    ]
    if not success:
        is_grid = phase in ("grid", "grid_smoke") if phase else False
        category = _classify_failure(exit_code, stderr, stdout, is_grid=is_grid)
        lines.append(f"Failure: {category}")
    stdout_preview = (stdout or "").strip()[:2000]
    if stdout_preview:
        lines.append(f"Output: {stdout_preview}")
    if not success and stderr:
        lines.append(f"Error: {stderr[:500]}")
    return "\n".join(lines)


async def _run_code_phase(
    source: str,
    hypothesis: str,
    *,
    phase: str,
    timeout: int,
    max_fix_attempts: int,
    review_logic: bool,
    plan: str = "",
    provider: str = "minimax",
) -> PhaseRunResult:
    attempts = 0
    fixes_used = 0
    current_source = source

    while True:
        attempts += 1
        valid, reason = validate_source(current_source)
        if not valid:
            logger.warning(
                "[CodeSandbox] %s validation failed (attempt %d): %s",
                phase,
                attempts,
                reason,
            )
            if fixes_used < max_fix_attempts:
                fixes_used += 1
                current_source = await _fix_code(
                    current_source,
                    f"Validation error: {reason}",
                    hypothesis,
                    phase=phase,
                    plan=plan,
                    provider=provider,
                )
                continue
            return PhaseRunResult(
                phase=phase,
                source_code=current_source,
                result=ExperimentResult(
                    stderr="All validation attempts failed",
                    exit_code=-1,
                    success=False,
                ),
                attempts=attempts,
            )

        # Map phase to sandbox mode — grid_smoke runs as "smoke" in the script
        sandbox_mode = "smoke" if phase == "grid_smoke" else phase
        result = await _execute_in_sandbox(
            current_source, timeout=timeout, mode=sandbox_mode
        )
        if not result.success:
            logger.info(
                "[CodeSandbox] %s execution failed (attempt %d)", phase, attempts
            )
            if fixes_used < max_fix_attempts:
                fixes_used += 1
                current_source = await _fix_code(
                    current_source,
                    result.stderr,
                    hypothesis,
                    phase=phase,
                    plan=plan,
                    provider=provider,
                )
                continue
            return PhaseRunResult(
                phase=phase,
                source_code=current_source,
                result=result,
                attempts=attempts,
            )

        if review_logic:
            logic_ok, logic_feedback = await _review_code_logic(
                hypothesis,
                current_source,
                result.stdout,
                phase=phase,
                provider=provider,
            )
            if not logic_ok:
                logger.info(
                    "[CodeSandbox] %s logic review failed (attempt %d): %s",
                    phase,
                    attempts,
                    logic_feedback[:100],
                )
                if fixes_used < max_fix_attempts:
                    fixes_used += 1
                    current_source = await _fix_code(
                        current_source,
                        f"Code ran in {phase} mode but logic review failed: {logic_feedback}",
                        hypothesis,
                        phase=phase,
                        plan=plan,
                        provider=provider,
                    )
                    continue
                return PhaseRunResult(
                    phase=phase,
                    source_code=current_source,
                    result=ExperimentResult(
                        stdout=result.stdout,
                        stderr=f"Logic review failed: {logic_feedback}",
                        exit_code=result.exit_code,
                        results_json=result.results_json,
                        output_images=result.output_images,
                        execution_time_s=result.execution_time_s,
                        success=False,
                    ),
                    attempts=attempts,
                )

        return PhaseRunResult(
            phase=phase,
            source_code=current_source,
            result=result,
            attempts=attempts,
        )


# ---------------------------------------------------------------------------
# Main entry point: generate → run → fix loop
# ---------------------------------------------------------------------------


async def run_code_evidence(
    hypothesis: str,
    context: str,
    *,
    topic_id: int,
    subtopic_id: int,
    role: str,
    max_iterations: int = 15,
    timeout: int = DEFAULT_TIMEOUT,
    provider: str = "minimax",
) -> CodeEvidenceItem:
    """Lightweight loop: generate once, smoke debug, full run, then logic review.

    Budget split: 1 generate + up to MAX_SMOKE_FIX_ATTEMPTS smoke fixes
    + up to MAX_FULL_FIX_ATTEMPTS full fixes.  Total iterations capped
    at max_iterations.
    """
    if max_iterations < 1:
        max_iterations = 1
    # Reserve full fix budget first, rest goes to smoke
    full_fix_attempts = min(MAX_FULL_FIX_ATTEMPTS, max(max_iterations - 2, 0))
    smoke_fix_attempts = min(
        MAX_SMOKE_FIX_ATTEMPTS, max(max_iterations - 1 - full_fix_attempts, 0)
    )
    full_timeout = max(timeout, FULL_TIMEOUT)

    # Step 1: Plan the experiment
    env_desc = await _get_sandbox_environment_description()
    plan_json_str = await _plan_experiment(
        hypothesis, context, env_desc, role=role, provider=provider
    )
    logger.info("[CodeSandbox] Experiment plan generated (%d chars)", len(plan_json_str))
    
    # Check feasibility
    try:
        parsed_plan = json.loads(plan_json_str)
        if isinstance(parsed_plan, dict):
            plan_dict = parsed_plan
        else:
            logger.warning(
                "[CodeSandbox] Plan JSON was %s instead of an object; ignoring feasibility gate.",
                type(parsed_plan).__name__,
            )
            plan_dict = {}
        if not plan_dict.get("feasible", True):
            rejection_reason = plan_dict.get("rejection_reason", "Experiment deemed unfeasible during planning.")
            logger.info("[CodeSandbox] Experiment rejected during planning: %s", rejection_reason)
            
            rendered = _render_code_evidence(
                hypothesis=hypothesis,
                success=False,
                iterations=0,
                execution_time_s=0.0,
                stdout="",
                stderr=f"Rejected during planning: {rejection_reason}",
                exit_code=-1,
            )
            summary = _build_summary(
                hypothesis=hypothesis,
                success=False,
                iterations=0,
                execution_time_s=0.0,
                phase="plan",
                exit_code=-1,
                stderr=f"Rejected during planning: {rejection_reason}",
                stdout="",
            )

            return CodeEvidenceItem(
                hypothesis=hypothesis,
                source_code="# REJECTED DURING PLANNING\n",
                stdout="",
                stderr=f"Rejected during planning: {rejection_reason}",
                exit_code=-1,
                execution_time_s=0.0,
                success=False,
                iterations=0,
                rendered_results=f"{rendered}\n\nSummary: {summary}",
                code_evidence_id=0,
                planning_veto=True,
            )
    except json.JSONDecodeError:
        logger.warning("[CodeSandbox] Plan was not valid JSON, proceeding anyway.")
        plan_dict = {}

    # Step 2: Generate code from plan
    source = await _generate_code(
        hypothesis, context, role=role, plan=plan_json_str, provider=provider
    )
    smoke_phase = await _run_code_phase(
        source,
        hypothesis,
        phase="smoke",
        timeout=SMOKE_TIMEOUT,
        max_fix_attempts=smoke_fix_attempts,
        review_logic=False,
        plan=plan_json_str,
        provider=provider,
    )

    # Smoke format check: verify output has CONCLUSION section
    if smoke_phase.result.success:
        fmt_ok, fmt_feedback = _review_smoke_output(
            smoke_phase.result.stdout, smoke_phase.source_code
        )
        if not fmt_ok:
            logger.info(
                "[CodeSandbox] Smoke format check failed: %s", fmt_feedback[:100]
            )
            # Fix and re-run smoke with remaining attempts
            remaining_smoke = max(smoke_fix_attempts - smoke_phase.attempts, 1)
            fixed_source = await _fix_code(
                smoke_phase.source_code,
                fmt_feedback,
                hypothesis,
                phase="smoke",
                plan=plan_json_str,
                provider=provider,
            )
            smoke_phase = await _run_code_phase(
                fixed_source,
                hypothesis,
                phase="smoke",
                timeout=SMOKE_TIMEOUT,
                max_fix_attempts=remaining_smoke,
                review_logic=False,
                plan=plan_json_str,
                provider=provider,
            )
            # Re-check format after fix
            if smoke_phase.result.success:
                fmt_ok2, fmt_feedback2 = _review_smoke_output(
                    smoke_phase.result.stdout, smoke_phase.source_code
                )
                if not fmt_ok2:
                    smoke_phase = PhaseRunResult(
                        phase="smoke",
                        source_code=smoke_phase.source_code,
                        result=ExperimentResult(
                            stdout=smoke_phase.result.stdout,
                            stderr=f"Smoke format check failed: {fmt_feedback2}",
                            exit_code=smoke_phase.result.exit_code,
                            execution_time_s=smoke_phase.result.execution_time_s,
                            success=False,
                        ),
                        attempts=smoke_phase.attempts,
                    )

    final_phase = smoke_phase
    total_iterations = smoke_phase.attempts
    if smoke_phase.result.success:
        full_phase = await _run_code_phase(
            smoke_phase.source_code,
            hypothesis,
            phase="full",
            timeout=full_timeout,
            max_fix_attempts=full_fix_attempts,
            review_logic=True,
            plan=plan_json_str,
            provider=provider,
        )
        final_phase = full_phase
        total_iterations += full_phase.attempts

    # Note: output images are extracted and saved to sandbox_output/ by
    # _execute_in_sandbox, but NOT analyzed by default. Call
    # analyze_output_images() on-demand if visual interpretation is needed
    # (e.g., during Librarian review of ambiguous results).

    rendered = _render_code_evidence(
        hypothesis=hypothesis,
        success=final_phase.result.success,
        iterations=total_iterations,
        execution_time_s=final_phase.result.execution_time_s,
        stdout=final_phase.result.stdout,
        stderr=final_phase.result.stderr,
        phase=final_phase.phase,
        exit_code=final_phase.result.exit_code,
    )
    summary = _build_summary(
        hypothesis=hypothesis,
        success=final_phase.result.success,
        iterations=total_iterations,
        execution_time_s=final_phase.result.execution_time_s,
        phase=final_phase.phase,
        exit_code=final_phase.result.exit_code,
        stderr=final_phase.result.stderr,
        stdout=final_phase.result.stdout,
    )

    # Persist to DB
    try:
        eid = api.insert_code_evidence(
            origin_topic_id=topic_id,
            origin_subtopic_id=subtopic_id,
            hypothesis=hypothesis,
            source_code=final_phase.source_code,
            stdout=final_phase.result.stdout,
            stderr=final_phase.result.stderr,
            exit_code=final_phase.result.exit_code,
            execution_time_s=final_phase.result.execution_time_s,
            iterations=total_iterations,
            success=final_phase.result.success,
            requesting_role=role,
            summary=summary,
        )
    except Exception as exc:
        logger.error("[CodeSandbox] Failed to persist code evidence: %s", exc)
        eid = 0

    analytics.capture(
        f"topic_{topic_id}",
        "code_evidence_completed",
        {
            "success": final_phase.result.success,
            "iterations": total_iterations,
            "execution_time_s": final_phase.result.execution_time_s,
            "phase": final_phase.phase,
            "role": role,
        },
    )

    return CodeEvidenceItem(
        hypothesis=hypothesis,
        source_code=final_phase.source_code,
        stdout=final_phase.result.stdout,
        stderr=final_phase.result.stderr,
        exit_code=final_phase.result.exit_code,
        execution_time_s=final_phase.result.execution_time_s,
        success=final_phase.result.success,
        iterations=total_iterations,
        rendered_results=rendered,
        code_evidence_id=eid,
    )


# ---------------------------------------------------------------------------
# Tier 2b: Grid — multi-seed/size sweep (optional, agent-triggered)
# ---------------------------------------------------------------------------


def _try_parse_grid_stdout(stdout: str) -> tuple[bool, str]:
    """Try to extract a conclusion from grid stdout even if the run failed.

    Checks for partial results that may be sufficient to judge the hypothesis.
    Returns (success, conclusion_or_reason).
    """
    if not stdout or len(stdout.strip()) < 50:
        return False, "No stdout to parse"

    # Check if there's a CONCLUSION section (even partial runs may print one)
    conclusion = _extract_conclusion(stdout)
    if conclusion:
        return True, conclusion

    # Check for partial results — count how many data points are present
    data_lines = re.findall(
        r"(?:seed|n_train|N|n_samples|dataset|config|fold|trial)\s*[=:]\s*\S+.*?"
        r"(?:F1|accuracy|score|AUC|RMSE|loss|precision|recall|MSE|R2|MAE|time)\s*[=:]\s*[\d.]+",
        stdout,
        re.IGNORECASE,
    )
    # Also count lines with 2+ floating point numbers (table-format results)
    numeric_lines = re.findall(r"^.*\d+\.\d+.*\d+\.\d+.*$", stdout, re.MULTILINE)
    total_data_points = max(len(data_lines), len(numeric_lines))
    if total_data_points >= 3:
        return (
            True,
            f"Partial results ({total_data_points} data points extracted from stdout before timeout/failure)",
        )

    return (
        False,
        f"Insufficient data in stdout ({len(stdout)} chars, {total_data_points} data points)",
    )


async def _generate_grid_code(
    full_source: str,
    hypothesis: str,
    context: str,
    *,
    role: str = "",
    provider: str = "minimax",
) -> str:
    """Generate grid sweep code based on a working Full-mode single-config script."""
    allowed = ", ".join(sorted(get_allowed_imports()))
    role_hint = ROLE_CODE_HINTS.get(role, "")
    role_line = f"- Role specialization: {role_hint}\n" if role_hint else ""
    prompt = (
        f"Expand this WORKING single-config experiment into a multi-seed, multi-size grid sweep.\n\n"
        f"Original hypothesis: {hypothesis}\n\n"
        f"Working single-config code:\n```python\n{full_source}\n```\n\n"
        f"Context:\n{context}\n\n"
        "Requirements:\n"
        "- Keep the core experiment logic from the working code\n"
        "- Add outer loops for multiple seeds (at least 3) and/or multiple dataset sizes\n"
        "- Collect results into a summary table\n"
        f"- Grid mode has {GRID_TIMEOUT}s (20 minutes) timeout — plan scope accordingly\n"
        "- The script receives `sys.argv[1]` = 'smoke' or 'grid'\n"
        "- Smoke mode: run with 2 seeds, smallest size only (verify format)\n"
        "- Grid mode: run the full sweep\n"
        "- MUST print a CONCLUSION section summarizing the grid results\n"
        "  (e.g., 'CONCLUSION: MLP beats RF on 4/6 datasets, mean advantage 0.03 F1, p=0.02')\n"
        f"- Use only these imports: {allowed}\n"
        "- No network access, no os/subprocess/eval/exec\n"
        f"- Keep under {MAX_SOURCE_LINES} lines\n"
        f"{role_line}"
        "Output only the Python code, no markdown fences."
    )
    text = await call_text(
        prompt,
        system_instruction="You are a scientific computing code generator. Output only valid Python code.",
        provider=provider,
        fallback_role="code_sandbox",
        temperature=0.3,
        max_tokens=DEFAULT_MAX_TOKENS,
    )
    stripped = (text or "").strip()
    stripped = re.sub(r"^```(?:python)?\s*\n", "", stripped)
    stripped = re.sub(r"\n```\s*$", "", stripped)
    return stripped.strip()


async def run_code_evidence_grid(
    hypothesis: str,
    context: str,
    *,
    topic_id: int,
    subtopic_id: int,
    role: str,
    full_evidence_id: int,
    provider: str = "minimax",
) -> CodeEvidenceItem:
    """Grid sweep: expand a working Full-mode experiment into multi-seed/size.

    Requires a PASSED Full evidence as starting point.
    Flow: generate grid code → grid smoke (120s) → grid full (1200s).
    On failure: try parsing stdout first, then 1 real fix attempt.
    """
    # Load the working Full evidence
    original = api.get_code_evidence_by_id(full_evidence_id)
    if not original or not original.get("success"):
        err_msg = f"Full evidence E{full_evidence_id} not found or not PASSED"
        return CodeEvidenceItem(
            hypothesis=hypothesis,
            source_code="",
            stdout="",
            stderr=err_msg,
            exit_code=-1,
            execution_time_s=0.0,
            success=False,
            iterations=0,
            rendered_results=f"Grid FAILED: {err_msg}",
            code_evidence_id=0,
        )
    if original.get("origin_topic_id") != topic_id:
        err_msg = f"Full evidence E{full_evidence_id} belongs to a different topic"
        return CodeEvidenceItem(
            hypothesis=hypothesis,
            source_code="",
            stdout="",
            stderr=err_msg,
            exit_code=-1,
            execution_time_s=0.0,
            success=False,
            iterations=0,
            rendered_results=f"Grid FAILED: {err_msg}",
            code_evidence_id=0,
        )

    full_source = original.get("source_code", "")
    grid_hypothesis = f"Grid sweep of E{full_evidence_id}: {hypothesis}"

    # Step 1: Generate grid code from working Full code
    source = await _generate_grid_code(
        full_source, hypothesis, context, role=role, provider=provider
    )

    # Step 2: Grid Smoke — verify format with minimal grid (2 seeds, smallest size)
    smoke_phase = await _run_code_phase(
        source,
        grid_hypothesis,
        phase="grid_smoke",
        timeout=GRID_SMOKE_TIMEOUT,
        max_fix_attempts=MAX_GRID_SMOKE_FIX_ATTEMPTS,
        review_logic=False,
        provider=provider,
    )
    # Grid smoke format check: verify CONCLUSION section
    if smoke_phase.result.success:
        fmt_ok, fmt_feedback = _review_smoke_output(
            smoke_phase.result.stdout, smoke_phase.source_code
        )
        if not fmt_ok:
            logger.info(
                "[GridSandbox] Grid smoke format check failed: %s", fmt_feedback[:100]
            )
            smoke_phase = PhaseRunResult(
                phase="grid_smoke",
                source_code=smoke_phase.source_code,
                result=ExperimentResult(
                    stdout=smoke_phase.result.stdout,
                    stderr=f"Smoke format check failed: {fmt_feedback}",
                    exit_code=0,
                    execution_time_s=smoke_phase.result.execution_time_s,
                    success=False,
                ),
                attempts=smoke_phase.attempts,
            )

    if not smoke_phase.result.success:
        # Grid smoke failed — persist and return
        rendered = _render_code_evidence(
            hypothesis=grid_hypothesis,
            success=False,
            iterations=smoke_phase.attempts,
            execution_time_s=smoke_phase.result.execution_time_s,
            stdout=smoke_phase.result.stdout,
            stderr=smoke_phase.result.stderr,
            phase="grid_smoke",
            exit_code=smoke_phase.result.exit_code,
        )
        summary = _build_summary(
            hypothesis=grid_hypothesis,
            success=False,
            iterations=smoke_phase.attempts,
            execution_time_s=smoke_phase.result.execution_time_s,
            phase="grid_smoke",
            exit_code=smoke_phase.result.exit_code,
            stderr=smoke_phase.result.stderr,
            stdout=smoke_phase.result.stdout,
        )
        try:
            eid = api.insert_code_evidence(
                origin_topic_id=topic_id,
                origin_subtopic_id=subtopic_id,
                hypothesis=grid_hypothesis,
                source_code=smoke_phase.source_code,
                stdout=smoke_phase.result.stdout,
                stderr=smoke_phase.result.stderr,
                exit_code=smoke_phase.result.exit_code,
                execution_time_s=smoke_phase.result.execution_time_s,
                iterations=smoke_phase.attempts,
                success=False,
                requesting_role=role,
                summary=summary,
                parent_evidence_id=full_evidence_id,
            )
        except Exception as exc:
            logger.error("[GridSandbox] Failed to persist grid smoke: %s", exc)
            eid = 0
        return CodeEvidenceItem(
            hypothesis=grid_hypothesis,
            source_code=smoke_phase.source_code,
            stdout=smoke_phase.result.stdout,
            stderr=smoke_phase.result.stderr,
            exit_code=smoke_phase.result.exit_code,
            execution_time_s=smoke_phase.result.execution_time_s,
            success=False,
            iterations=smoke_phase.attempts,
            rendered_results=rendered,
            code_evidence_id=eid,
        )

    # Step 3: Grid Full — run the actual sweep (1200s)
    grid_phase = await _run_code_phase(
        smoke_phase.source_code,
        grid_hypothesis,
        phase="grid",
        timeout=GRID_TIMEOUT,
        max_fix_attempts=0,  # No fix attempts on first run
        review_logic=True,
        provider=provider,
    )

    total_iterations = smoke_phase.attempts + grid_phase.attempts
    final_source = grid_phase.source_code
    final_result = grid_phase.result

    # Step 4: If grid failed, try parsing stdout before giving up
    # Do NOT override logic review rejections — those are intentional quality gates
    is_logic_review_failure = (
        "logic review failed" in (final_result.stderr or "").lower()
    )
    if not final_result.success and not is_logic_review_failure:
        parsed_ok, parsed_conclusion = _try_parse_grid_stdout(final_result.stdout)
        if parsed_ok:
            logger.info(
                "[GridSandbox] Grid failed but stdout has usable results: %s",
                parsed_conclusion[:100],
            )
            # Mark as success with parsed results
            final_result = ExperimentResult(
                stdout=final_result.stdout,
                stderr=f"Grid execution failed but results parsed from stdout: {parsed_conclusion}",
                exit_code=0,  # Mark as clean since we extracted usable results
                results_json=final_result.results_json,
                output_images=final_result.output_images,
                execution_time_s=final_result.execution_time_s,
                success=True,
            )
        else:
            # Fix code first (using the error from the failed run), then run once
            logger.info("[GridSandbox] Grid failed, fixing code then re-running")
            fixed_source = await _fix_code(
                final_source,
                final_result.stderr or "Grid execution failed",
                grid_hypothesis,
                phase="grid",
                provider=provider,
            )
            fix_phase = await _run_code_phase(
                fixed_source,
                grid_hypothesis,
                phase="grid",
                timeout=GRID_TIMEOUT,
                max_fix_attempts=0,  # Already fixed — just run once
                review_logic=True,
                provider=provider,
            )
            total_iterations += fix_phase.attempts
            final_source = fix_phase.source_code
            final_result = fix_phase.result

    # Persist — always save stdout even on failure
    rendered = _render_code_evidence(
        hypothesis=grid_hypothesis,
        success=final_result.success,
        iterations=total_iterations,
        execution_time_s=final_result.execution_time_s,
        stdout=final_result.stdout,
        stderr=final_result.stderr,
        phase="grid",
        exit_code=final_result.exit_code,
    )
    summary = _build_summary(
        hypothesis=grid_hypothesis,
        success=final_result.success,
        iterations=total_iterations,
        execution_time_s=final_result.execution_time_s,
        phase="grid",
        exit_code=final_result.exit_code,
        stderr=final_result.stderr,
        stdout=final_result.stdout,
    )
    try:
        eid = api.insert_code_evidence(
            origin_topic_id=topic_id,
            origin_subtopic_id=subtopic_id,
            hypothesis=grid_hypothesis,
            source_code=final_source,
            stdout=final_result.stdout,
            stderr=final_result.stderr,
            exit_code=final_result.exit_code,
            execution_time_s=final_result.execution_time_s,
            iterations=total_iterations,
            success=final_result.success,
            requesting_role=role,
            summary=summary,
            parent_evidence_id=full_evidence_id,
        )
    except Exception as exc:
        logger.error("[GridSandbox] Failed to persist grid evidence: %s", exc)
        eid = 0

    return CodeEvidenceItem(
        hypothesis=grid_hypothesis,
        source_code=final_source,
        stdout=final_result.stdout,
        stderr=final_result.stderr,
        exit_code=final_result.exit_code,
        execution_time_s=final_result.execution_time_s,
        success=final_result.success,
        iterations=total_iterations,
        rendered_results=rendered,
        code_evidence_id=eid,
    )


# ---------------------------------------------------------------------------
# Tier 3: Code Review — critic pulls existing code, modifies, re-runs
# ---------------------------------------------------------------------------


async def _generate_review_code(
    original_source: str,
    critique: str,
    hypothesis: str,
    *,
    provider: str = "minimax",
) -> str:
    """Ask LLM to modify existing code to address a critique."""
    allowed = ", ".join(sorted(get_allowed_imports()))
    env_desc = await _get_sandbox_environment_description()
    prompt = (
        f"Original hypothesis being tested: {hypothesis}\n\n"
        "Exact sandbox environment facts:\n"
        f"{env_desc[:3000]}\n\n"
        f"Original code:\n```python\n{original_source}\n```\n\n"
        f"Critique to address: {critique}\n\n"
        "Modify the code to test the critique. For example, if the critique says "
        "'t-test assumes normality but data is skewed', modify the code to test "
        "for normality and use an appropriate non-parametric test.\n\n"
        "Requirements:\n"
        "- Single file only\n"
        f"- Use only these imports: {allowed}\n"
        "- Print all results to stdout — stdout IS the evidence\n"
        f"- Keep under {MAX_SOURCE_LINES} lines\n"
        "- No network access, no os/subprocess/eval/exec. open() allowed ONLY for /workspace/output/\n"
        "- Do NOT import warnings or call warnings.filterwarnings(). Keep warnings visible.\n"
        "- Preserve the script's smoke/full mode contract if the original code already uses it\n"
        "Output only the Python code, no markdown fences."
    )
    text = await call_text(
        prompt,
        system_instruction="You are a scientific code review expert. Output only valid Python code.",
        provider=provider,
        fallback_role="code_sandbox",
        temperature=0.3,
        max_tokens=DEFAULT_MAX_TOKENS,
    )
    stripped = (text or "").strip()
    stripped = re.sub(r"^```(?:python)?\s*\n", "", stripped)
    stripped = re.sub(r"\n```\s*$", "", stripped)
    return stripped.strip()


async def run_code_review(
    evidence_id: int,
    critique: str,
    *,
    topic_id: int,
    subtopic_id: int,
    role: str,
    max_iterations: int = 8,
    timeout: int = DEFAULT_TIMEOUT,
    provider: str = "minimax",
) -> CodeEvidenceItem:
    """Tier 3: Pull existing CodeEvidence, modify per critique, validate→execute→fix loop."""
    original = api.get_code_evidence_by_id(evidence_id)
    if not original:
        rendered = f"Code Review FAILED: E{evidence_id} not found"
        return CodeEvidenceItem(
            hypothesis=f"Review of E{evidence_id}: {critique}",
            source_code="",
            stdout="",
            stderr=rendered,
            exit_code=-1,
            execution_time_s=0.0,
            success=False,
            iterations=0,
            rendered_results=rendered,
        )

    # Cross-topic safety: ensure evidence belongs to this topic
    if original.get("origin_topic_id") != topic_id:
        rendered = f"Code Review FAILED: E{evidence_id} belongs to a different topic"
        logger.warning(
            "[CodeReview] Cross-topic access attempt: E%d (topic %s) from topic %d",
            evidence_id,
            original.get("origin_topic_id"),
            topic_id,
        )
        return CodeEvidenceItem(
            hypothesis=f"Review of E{evidence_id}: {critique}",
            source_code="",
            stdout="",
            stderr=rendered,
            exit_code=-1,
            execution_time_s=0.0,
            success=False,
            iterations=0,
            rendered_results=rendered,
        )

    # Gate: refuse review if already reviewed MAX_REVIEW_COUNT times without issues
    review_count = original.get("review_count", 0) or 0
    if review_count >= MAX_REVIEW_COUNT:
        rendered = f"Code Review skipped: E{evidence_id} already reviewed {review_count} times without issues"
        logger.info(
            "[CodeReview] E%d at review cap (%d), skipping", evidence_id, review_count
        )
        return CodeEvidenceItem(
            hypothesis=f"Review of E{evidence_id}: {critique}",
            source_code="",
            stdout="",
            stderr=rendered,
            exit_code=0,
            execution_time_s=0.0,
            success=True,
            iterations=0,
            rendered_results=rendered,
        )

    original_source = original.get("source_code", "")
    original_hypothesis = original.get("hypothesis", "")

    source = await _generate_review_code(
        original_source, critique, original_hypothesis, provider=provider
    )
    last_result: Optional[ExperimentResult] = None
    iterations_used = 0
    review_hypothesis = f"Review of E{evidence_id}: {critique}"

    for iterations_used in range(1, max_iterations + 1):
        valid, reason = validate_source(source)
        if not valid:
            logger.warning(
                "[CodeReview] Validation failed (iter %d): %s", iterations_used, reason
            )
            if iterations_used < max_iterations:
                source = await _fix_code(
                    source,
                    f"Validation error: {reason}",
                    review_hypothesis,
                    provider=provider,
                )
                continue
            break

        last_result = await _execute_in_sandbox(source, timeout=timeout)

        if last_result.success:
            logic_ok, logic_feedback = await _review_code_logic(
                review_hypothesis,
                source,
                last_result.stdout,
                provider=provider,
            )
            if logic_ok:
                break
            if iterations_used < max_iterations:
                logger.info(
                    "[CodeReview] Logic review failed (iter %d): %s",
                    iterations_used,
                    logic_feedback[:100],
                )
                source = await _fix_code(
                    source,
                    f"Code ran but logic review failed: {logic_feedback}",
                    review_hypothesis,
                    provider=provider,
                )
                last_result = None
                continue
            break

        if iterations_used < max_iterations:
            logger.info(
                "[CodeReview] Execution failed (iter %d), asking LLM to fix...",
                iterations_used,
            )
            source = await _fix_code(
                source, last_result.stderr, review_hypothesis, provider=provider
            )

    if last_result is None:
        last_result = ExperimentResult(
            stderr="All validation attempts failed",
            exit_code=-1,
            success=False,
        )

    iterations_used = max(iterations_used, 1)

    rendered = _render_code_evidence(
        hypothesis=review_hypothesis,
        success=last_result.success,
        iterations=iterations_used,
        execution_time_s=last_result.execution_time_s,
        stdout=last_result.stdout,
        stderr=last_result.stderr,
        exit_code=last_result.exit_code,
    )
    summary = _build_summary(
        hypothesis=review_hypothesis,
        success=last_result.success,
        iterations=iterations_used,
        execution_time_s=last_result.execution_time_s,
        phase=None,
        exit_code=last_result.exit_code,
        stderr=last_result.stderr,
        stdout=last_result.stdout,
    )

    try:
        eid = api.insert_code_evidence(
            origin_topic_id=topic_id,
            origin_subtopic_id=subtopic_id,
            hypothesis=review_hypothesis,
            source_code=source,
            stdout=last_result.stdout,
            stderr=last_result.stderr,
            exit_code=last_result.exit_code,
            execution_time_s=last_result.execution_time_s,
            iterations=iterations_used,
            success=last_result.success,
            requesting_role=role,
            summary=summary,
            parent_evidence_id=evidence_id,
        )
    except Exception as exc:
        logger.error("[CodeSandbox] Failed to persist code review evidence: %s", exc)
        eid = 0

    # Update parent's review_count based on review outcome
    try:
        original_success = bool(original.get("success"))
        review_success = last_result.success
        if review_success and review_success != original_success:
            # Review found a real problem: the corrected code contradicts original
            api.reset_code_evidence_review_count(evidence_id)
            logger.info(
                "[CodeReview] E%d review_count RESET — review contradicts original",
                evidence_id,
            )
        else:
            # Review confirms original (or review itself failed — don't punish the original)
            api.increment_code_evidence_review_count(evidence_id)
            logger.info("[CodeReview] E%d review_count incremented", evidence_id)
    except Exception as exc:
        logger.warning(
            "[CodeReview] Failed to update review_count for E%d: %s", evidence_id, exc
        )

    return CodeEvidenceItem(
        hypothesis=review_hypothesis,
        source_code=source,
        stdout=last_result.stdout,
        stderr=last_result.stderr,
        exit_code=last_result.exit_code,
        execution_time_s=last_result.execution_time_s,
        success=last_result.success,
        iterations=iterations_used,
        rendered_results=rendered,
        code_evidence_id=eid,
    )
