"""PostToolUse hook: format and lint-fix edited Python files with ruff."""

import json
import subprocess
import sys


def ruff(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["uv", "run", "ruff", *args], capture_output=True, text=True)


def main() -> None:
    event = json.load(sys.stdin)
    path = event.get("tool_input", {}).get("file_path", "")
    if not path.endswith(".py"):
        return

    ruff("format", path)
    ruff("check", "--fix", "--quiet", path)
    remaining = ruff("check", "--quiet", path)
    if remaining.returncode != 0:
        output = {
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": f"ruff reports remaining issues in {path}:\n"
                + remaining.stdout,
            }
        }
        print(json.dumps(output))


if __name__ == "__main__":
    main()
