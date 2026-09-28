---
name: protokoll
description: Turn raw meeting notes into a structured German meeting protocol in docs/protokolle/.
argument-hint: "<raw notes or topic>"
disable-model-invocation: true
---

# Meeting protocol

Input notes: $ARGUMENTS

If the notes are empty or too thin, ask for: date, attendees, topics, decisions, action items.

Create `docs/protokolle/<YYYY-MM-DD>_<topic-kebab>.md` (date of the meeting, default today)
in **German** with this structure; leave out empty sections except "Action Items":

```markdown
# <Art des Termins>: <Thema> – <TT.MM.JJJJ>

**Teilnehmende:** …

## Themen
- …

## Entscheidungen
- …

## Action Items
- [ ] <Aufgabe> – **<Person>** – bis <Datum>

## Offene Fragen
- …
```

Rules:
- Keep the wording of decisions precise; do not invent content that is not in the notes.
- If a decision affects the project setup or conventions, point out which file should change
  (e.g. `AGENTS.md`, `docs/daten/index.md`) and offer to update it.
- If a milestone task from `docs/projekt/zeitplan.md` is reported as done, tick its checkbox.
- Show the file path at the end.
