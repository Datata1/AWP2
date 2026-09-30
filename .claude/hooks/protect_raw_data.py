"""PreToolUse hook: block any modification of data/raw/ (raw data is read-only)."""

import json
import re
import sys

RAW = "data/raw"
FILE_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}

_SAME_COMMAND = r"[^;|&\n]*"
_RAW_PATH = r"\S*data/raw\b"
# Only commands that write *into* data/raw are blocked: reading raw files or copying them
# elsewhere must keep working.
WRITE_RULES = {
    "modify or delete": re.compile(
        rf"\b(rm|mv|unlink|truncate|chmod|touch|sed\s+-i){_SAME_COMMAND}{_RAW_PATH}"
    ),
    "copy into": re.compile(rf"\b(cp|rsync|ln){_SAME_COMMAND}\s{_RAW_PATH}\S*\s*($|[;|&])"),
    "redirect into": re.compile(rf">\s*{_RAW_PATH}"),
    "tee into": re.compile(rf"\btee\b{_SAME_COMMAND}\s{_RAW_PATH}"),
}


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
        for rule, pattern in WRITE_RULES.items():
            if pattern.search(command):
                block(
                    f"Command would {rule} {RAW}/, which is read-only. "
                    "Raw files are copied there by hand; write outputs elsewhere."
                )


if __name__ == "__main__":
    main()
