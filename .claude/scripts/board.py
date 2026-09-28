"""Set the status of an issue/PR on the GitHub project board (adds it to the board if missing).

Usage: python3 .claude/scripts/board.py <issue-or-pr-number> "<Status>"
       python3 .claude/scripts/board.py --statuses
"""

import json
import subprocess
import sys

OWNER = "Datata1"
REPO = "AWP2"
PROJECT_TITLE = "AWP2"


def gh(*args: str) -> dict | list:
    out = subprocess.run(["gh", *args], capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"gh {' '.join(args[:3])} failed: {out.stderr.strip()}")
    return json.loads(out.stdout) if out.stdout.strip() else {}


def project() -> dict:
    projects = gh("project", "list", "--owner", OWNER, "--format", "json")["projects"]
    for p in projects:
        if p["title"] == PROJECT_TITLE:
            return p
    sys.exit(f"Project '{PROJECT_TITLE}' not found for {OWNER}.")


def status_field(number: int) -> dict:
    fields = gh("project", "field-list", str(number), "--owner", OWNER, "--format", "json")
    return next(f for f in fields["fields"] if f["name"] == "Status")


def find_or_add_item(number: int, content_number: int) -> str:
    items = gh(
        "project", "item-list", str(number), "--owner", OWNER, "--format", "json", "--limit", "500"
    )["items"]
    for item in items:
        content = item.get("content", {})
        if content.get("number") == content_number and REPO in content.get("repository", ""):
            return item["id"]
    # The issues endpoint works for both issues and pull requests.
    url = gh("api", f"repos/{OWNER}/{REPO}/issues/{content_number}")["html_url"]
    added = gh("project", "item-add", str(number), "--owner", OWNER, "--url", url,
               "--format", "json")  # fmt: skip
    return added["id"]


def main() -> None:
    proj = project()
    field = status_field(proj["number"])
    options = {o["name"]: o["id"] for o in field["options"]}
    if sys.argv[1:] == ["--statuses"]:
        print(", ".join(options))
        return
    if len(sys.argv) != 3 or sys.argv[2] not in options:
        sys.exit(f"Usage: board.py <number> <status>; statuses: {', '.join(options)}")

    content_number, status = int(sys.argv[1].lstrip("#")), sys.argv[2]
    item_id = find_or_add_item(proj["number"], content_number)
    gh("project", "item-edit", "--id", item_id, "--project-id", proj["id"],
       "--field-id", field["id"], "--single-select-option-id", options[status])  # fmt: skip
    print(f"#{content_number} → {status}")


if __name__ == "__main__":
    main()
