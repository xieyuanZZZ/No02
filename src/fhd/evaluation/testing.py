"""在阶段脚本里调用 pytest 并结构化记录结果。

直接使用 pytest 的插件钩子收集用例结果，而不是解析文本摘要——后者在不同
pytest 版本/终端宽度下不稳定。
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def run_pytest(selection: list[str], quiet: bool = True) -> dict:
    import pytest

    outcomes: dict[str, str] = {}
    durations: dict[str, float] = {}

    class Collector:
        def pytest_runtest_logreport(self, report):  # noqa: D401
            if report.when == "call":
                outcomes[report.nodeid] = report.outcome
                durations[report.nodeid] = report.duration
            elif report.when == "setup" and report.outcome != "passed":
                outcomes[report.nodeid] = report.outcome

    args = [*selection, "--no-header", "-p", "no:cacheprovider"]
    if quiet:
        args.append("-q")
    buffer = io.StringIO()
    old_stdout = sys.stdout
    try:
        with redirect_stdout(buffer):
            code = pytest.main(args, plugins=[Collector()])
    finally:
        sys.stdout = old_stdout
    output = buffer.getvalue()

    passed = sum(1 for v in outcomes.values() if v == "passed")
    failed = [k for k, v in outcomes.items() if v == "failed"]
    errors = [k for k, v in outcomes.items() if v == "error"]
    skipped = sum(1 for v in outcomes.values() if v == "skipped")
    return {
        "passed": passed,
        "failed": len(failed),
        "error": len(errors),
        "skipped": skipped,
        "failed_tests": failed + errors,
        "all_passed": bool(failed == [] and errors == [] and passed > 0),
        "returncode": int(code),
        "selection": selection,
        "total_duration_s": round(sum(durations.values()), 3),
        "output_tail": "\n".join(output.strip().splitlines()[-40:]),
    }


def parse_summary(output: str) -> dict:  # 保留给外部日志用
    import re

    result = {"passed": 0, "failed": 0, "error": 0, "skipped": 0, "failed_tests": []}
    for match in re.finditer(r"(\d+) (passed|failed|error|errors|skipped)", output):
        count = int(match.group(1))
        kind = match.group(2)
        if kind == "passed":
            result["passed"] = count
        elif kind == "failed":
            result["failed"] = count
        elif kind.startswith("error"):
            result["error"] = count
        elif kind == "skipped":
            result["skipped"] = count
    result["failed_tests"] = re.findall(r"^FAILED ([^\s:]+)", output, flags=re.MULTILINE)
    result["all_passed"] = result["failed"] == 0 and result["error"] == 0 and result["passed"] > 0
    return result
