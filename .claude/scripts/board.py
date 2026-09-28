"""Set Status and/or Bereich of an issue or PR on the GitHub project board.

Adds the item to the board if it is missing. Uses one read and one write request.

Usage:
    python3 .claude/scripts/board.py <nr> "<Status>" [--bereich "<Bereich>"]
    python3 .claude/scripts/board.py <nr> --bereich "<Bereich>"
    python3 .claude/scripts/board.py --options
"""

import argparse
import json
import subprocess
import sys

OWNER = "Datata1"
REPO = "AWP2"
PROJECT_NUMBER = 2
FIELDS = {"status": "Status", "bereich": "Bereich"}

_FIELD = "... on ProjectV2SingleSelectField { id options { id name } }"
_ITEMS = "id projectItems(first: 20) { nodes { id project { number } } }"


def graphql(query: str) -> dict:
    out = subprocess.run(
        ["gh", "api", "graphql", "-f", f"query={query}"], capture_output=True, text=True
    )
    data = json.loads(out.stdout or "{}")
    if out.returncode != 0 or "errors" in data:
        sys.exit(f"GitHub API error: {data.get('errors') or out.stderr.strip()}")
    return data["data"]


def load(number: int | None) -> tuple[dict, dict | None]:
    """Fetch project fields and (optionally) the issue/PR with its board items."""
    content = ""
    if number is not None:
        content = (
            f'repository(owner: "{OWNER}", name: "{REPO}") {{ '
            f"issueOrPullRequest(number: {number}) {{ "
            f"... on Issue {{ {_ITEMS} }} ... on PullRequest {{ {_ITEMS} }} }} }}"
        )
    fields = " ".join(
        f'{key}: field(name: "{name}") {{ {_FIELD} }}' for key, name in FIELDS.items()
    )
    data = graphql(
        f'{{ user(login: "{OWNER}") {{ projectV2(number: {PROJECT_NUMBER}) {{ id {fields} }} }} '
        f"{content} }}"
    )
    project = data["user"]["projectV2"]
    item = data.get("repository", {}).get("issueOrPullRequest") if number is not None else None
    if number is not None and item is None:
        sys.exit(f"#{number} not found in {OWNER}/{REPO}.")
    return project, item


def option_id(project: dict, key: str, value: str) -> str:
    options = {o["name"]: o["id"] for o in project[key]["options"]}
    if value not in options:
        sys.exit(f"Unknown {FIELDS[key]} '{value}'. Options: {', '.join(options)}")
    return options[value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("number", nargs="?", type=lambda s: int(s.lstrip("#")))
    parser.add_argument("status", nargs="?")
    parser.add_argument("--bereich")
    parser.add_argument("--options", action="store_true", help="list Status/Bereich options")
    args = parser.parse_args()

    if args.options:
        project, _ = load(None)
        for key, name in FIELDS.items():
            print(f"{name}: {', '.join(o['name'] for o in project[key]['options'])}")
        return
    if args.number is None or not (args.status or args.bereich):
        parser.error("need an issue number and a status and/or --bereich")

    project, content = load(args.number)
    # Validate before writing anything.
    values = {"status": args.status, "bereich": args.bereich}
    selected = {k: option_id(project, k, v) for k, v in values.items() if v}

    items = [n["id"] for n in content["projectItems"]["nodes"]
             if n["project"]["number"] == PROJECT_NUMBER]  # fmt: skip
    if items:
        item_id = items[0]
    else:  # not on the board yet: add it (one extra request)
        added = graphql(
            f'mutation {{ add: addProjectV2ItemById(input: {{projectId: "{project["id"]}", '
            f'contentId: "{content["id"]}"}}) {{ item {{ id }} }} }}'
        )
        item_id = added["add"]["item"]["id"]

    updates = [
        f'{key}: updateProjectV2ItemFieldValue(input: {{projectId: "{project["id"]}", '
        f'itemId: "{item_id}", fieldId: "{project[key]["id"]}", '
        f'value: {{singleSelectOptionId: "{opt}"}}}}) {{ projectV2Item {{ id }} }}'
        for key, opt in selected.items()
    ]
    graphql("mutation { " + " ".join(updates) + " }")
    print(f"#{args.number} → " + ", ".join(f"{FIELDS[k]}: {values[k]}" for k in selected))


if __name__ == "__main__":
    main()
