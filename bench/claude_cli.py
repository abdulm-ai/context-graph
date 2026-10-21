"""Run `claude -p` as a child process for the benchmark and the end-to-end test.

The API key comes from BENCH_ANTHROPIC_API_KEY and is passed as
ANTHROPIC_API_KEY to the child process only. It is never written to disk,
logged, or put in this process's own environment.

Each run gets its own CLAUDE_CONFIG_DIR, so no user settings, plugins or
memory leak into a measured session, and its transcripts stay in the run dir.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

KEY_VAR = "BENCH_ANTHROPIC_API_KEY"
_PASS_THROUGH = ("PATH", "LANG", "LC_ALL", "TMPDIR", "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy",
                 "http_proxy", "no_proxy", "NODE_EXTRA_CA_CERTS", "SSL_CERT_FILE")


def have_key() -> bool:
    return bool(os.environ.get(KEY_VAR))


def claude_version() -> str:
    exe = shutil.which("claude")
    if not exe:
        return ""
    out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=60)
    return out.stdout.strip()


def child_env(config_dir: str, home_dir: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {k: os.environ[k] for k in _PASS_THROUGH if k in os.environ}
    env["HOME"] = home_dir
    env["CLAUDE_CONFIG_DIR"] = config_dir
    env["DISABLE_AUTOUPDATER"] = "1"
    key = os.environ.get(KEY_VAR)
    if key:
        env["ANTHROPIC_API_KEY"] = key
    if extra:
        env.update(extra)
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(home_dir, exist_ok=True)
    return env


@dataclass
class Call:
    prompt: str
    ok: bool
    result: str
    session_id: str
    usage: dict[str, Any]
    cost_usd: float
    wall_s: float
    duration_ms: int | None
    num_turns: int | None
    models: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def input_tokens(self) -> int:
        """Every input token the API processed: uncached, cache writes and cache reads."""
        u = self.usage or {}
        return int(u.get("input_tokens", 0)) + int(u.get("cache_creation_input_tokens", 0)) + int(
            u.get("cache_read_input_tokens", 0))

    @property
    def output_tokens(self) -> int:
        return int((self.usage or {}).get("output_tokens", 0))

    def to_json(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["input_tokens"] = self.input_tokens
        d["output_tokens"] = self.output_tokens
        return d


def run(prompt: str, cwd: str, env: dict[str, str], *, model: str, resume: str | None = None,
        max_budget_usd: float = 1.0, tools: str | None = "Bash,Read,Edit,Write,Glob,Grep",
        plugin_dir: str | None = None, extra_args: list[str] | None = None, timeout: int = 900) -> Call:
    exe = shutil.which("claude") or "claude"
    args = [exe, "-p", prompt, "--model", model, "--output-format", "json",
            "--max-budget-usd", f"{max_budget_usd:.2f}"]
    if resume:
        args += ["--resume", resume]
    if tools is not None:
        args += ["--tools", tools]
        if tools:
            args += ["--allowedTools", tools.replace(",", " ")]
    if plugin_dir:
        args += ["--plugin-dir", plugin_dir]
    args += extra_args or []
    t0 = time.time()
    try:
        proc = subprocess.run(args, cwd=cwd, env=env, stdin=subprocess.DEVNULL, capture_output=True,
                              text=True, timeout=timeout)
        stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr, rc = "timeout", -1
    wall = time.time() - t0
    data: dict[str, Any] = {}
    try:
        data = json.loads(stdout.strip().splitlines()[-1]) if stdout.strip() else {}
    except (ValueError, IndexError):
        data = {}
    return Call(
        prompt=prompt,
        ok=rc == 0 and not data.get("is_error", False),
        result=str(data.get("result", "")),
        session_id=str(data.get("session_id", resume or "")),
        usage=data.get("usage") or {},
        cost_usd=float(data.get("total_cost_usd") or 0.0),
        wall_s=round(wall, 2),
        duration_ms=data.get("duration_ms"),
        num_turns=data.get("num_turns"),
        models=sorted((data.get("modelUsage") or {}).keys()),
        error="" if rc == 0 else (stderr or "")[-800:],
    )


def transcript_path(config_dir: str, session_id: str) -> str | None:
    root = os.path.join(config_dir, "projects")
    for dirpath, _, files in os.walk(root):
        if f"{session_id}.jsonl" in files:
            return os.path.join(dirpath, f"{session_id}.jsonl")
    return None
