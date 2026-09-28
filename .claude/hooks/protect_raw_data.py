"""PreToolUse hook: block any modification of data/raw/ (raw data is read-only)."""

import json
import re
import sys

RAW = "data/raw"
FILE_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}
# Shell commands that write, move or delete files *in* data/raw (not just mention it).
_ARGS = r"[^;|&\n]*"  # arguments within the same simple command
_RAW = r"\S*data/raw\b"
WRITE_PATTERNS = [
    re.compile(rf"\b(rm|mv|unlink|truncate|chmod|touch|sed\s+-i){_ARGS}{_RAW}"),
    re.compile(rf"\b(cp|rsync|ln){_ARGS}\s{_RAW}\S*\s*($|[;|&])"),  # raw dir is the target
    re.compile(rf">\s*{_RAW}"),  # redirect into raw
    re.compile(rf"\btee\b{_ARGS}\s{_RAW}"),
]


def block(reason: str) -> None:
    print(reason, file=sys.stderr)
    sys.exit(2)


def main() -> None:
    event = json.load(sys.stdin)
    tool = event.get("tool_name", "")
    tool_input = event.get("tool_input", {})

    if tool in FILE_TOOLS:
        path = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
        if f"{RAW}/" in path.replace("\\", "/"):
            block(f"{RAW}/ is read-only. Write derived data to data/interim/ or data/processed/.")

    if tool == "Bash":
        command = tool_input.get("command", "")
        if any(p.search(command) for p in WRITE_PATTERNS):
            block(
                f"Command would modify {RAW}/, which is read-only. "
                "Raw files are copied there by hand; write outputs elsewhere."
            )


if __name__ == "__main__":
    main()
