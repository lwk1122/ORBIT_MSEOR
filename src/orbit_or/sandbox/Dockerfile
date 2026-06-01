FROM python:3.12-slim

# Install build tools, compile native extensions, then remove build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    pip install --no-cache-dir \
    numpy scipy pandas matplotlib scikit-learn sympy \
    torch --extra-index-url https://download.pytorch.org/whl/cpu && \
    apt-get purge -y build-essential && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/orbit && python - <<'PY'
import sys
from importlib.metadata import version
from pathlib import Path

packages = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("scikit-learn", "scikit-learn"),
    ("sympy", "sympy"),
    ("torch", "torch"),
]

lines = [
    "# ORBIT Sandbox Environment",
    "",
    "## Runtime",
    f"- Python: {sys.version.split()[0]}",
    "- Network: disabled (`--network=none`)",
    "- Root filesystem: read-only",
    "- Writable paths: `/tmp`, `/workspace/output`",
    "- Resource limits: 2 CPU, 2 GB RAM, 256 PIDs",
    "- Execution contract: single-file Python via stdin",
    "- Mode contract: scripts should use `sys.argv[1]` for `smoke` / `full` when present",
    "- Timeouts: smoke=120s, full=300s",
    "",
    "## Installed Packages",
]

for label, dist_name in packages:
    lines.append(f"- {label}: {version(dist_name)}")

Path("/opt/orbit/ENVIRONMENT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

# Explicit UID/GID — must match tmpfs uid= in ensure_sandbox() and build.sh
RUN useradd -u 1000 -m -s /usr/sbin/nologin sandbox
USER sandbox
WORKDIR /workspace

RUN mkdir -p /workspace/output

CMD ["sleep", "infinity"]
