# Zeitplan & Meilensteine

Projektstart 28.09.2026 · Vollzeit-Blockveranstaltung (225 h) · Endabgabe **31.10.2026**

!!! warning "Termine klären"
    Die Daten der Meilensteine unterscheiden sich: Aufgabenbeschreibung 08./15./22.10.,
    Kickoff-Folien 09./16./23.10. Die Folien sagen aber auch „bis zum Statusmeeting KW 41/42/43“ –
    mit unserem Termin am **Dienstag** wäre das schon der **06./13./20.10.** Bis zur Klärung
    ([Offene Fragen](offene-fragen.md)) planen wir mit dem früheren Dienstagstermin.

## Meilenstein 1 – Domain & Baseline (KW 41)

- [ ] Kurzreport Problemverständnis (Domäne & Ziel) + Data Understanding
- [ ] Erste Preprocessing-Pipeline
- [ ] Erster Baseline-Klassifikator (Code + kurzer Demo-Run)

## Meilenstein 2 – Modellreife & Validierung (KW 42)

- [ ] Optimierte Preprocessing-Pipeline
- [ ] Verbessertes Modell mit sauberer Validierung
- [ ] Vergleich gegen Baseline, Fehleranalyse (Confusion Matrix, pro Klasse)

## Meilenstein 3 – Dateninput minimieren (KW 43)

- [ ] Studie zur Datenreduktion inkl. Trade-off-Analyse „Qualität vs. Input"
      (bei uns vermutlich: Anzahl/Auswahl der Spektralbänder)
- [ ] Bonus: Feature-Importance / Erklärbarkeit

## Endabgabe (31.10.)

Siehe [Bewertung & Abgabe](bewertung.md#abgabe-checkliste).

## Danach (November, Termine TBD)

- Abschlusspräsentation: 10–15 min Pitch + Q&A (Pflicht für alle)
- Einzelgespräch: Reflexion, Beitrag, Learnings
- Gemeinsamer Abschlusstermin (online)

## Statusmeetings

**Jeden Dienstag, 11:30–12:00 Uhr** mit den Betreuer:innen – Pflicht für alle, abwechselnd in
Präsenz und online (erstes Meeting in Woche 2).

| KW | Statusmeeting | Stundenzettel abgeben bis | Thema |
| --- | --- | --- | --- |
| 41 | Di 06.10., 11:30 | **Mo 05.10., 11:30** | Meilenstein 1 |
| 42 | Di 13.10., 11:30 | **Mo 12.10., 11:30** | Meilenstein 2 |
| 43 | Di 20.10., 11:30 | **Mo 19.10., 11:30** | Meilenstein 3 |
| 44 | Di 27.10., 11:30 | **Mo 26.10., 11:30** | Stand vor der Endabgabe |

**Montags bis 11:30 Uhr** (24 h vorher):

- [ ] Stundendoku aktuell: `/stunden` für fehlende Tage, dann `make stunden` und die `.xlsx` in den
  Teamordner in BWSyncAndShare hochladen ([Stundendoku](../entwicklung/workflow.md#stundendoku))
- [ ] Vorbereitung beginnen: `/status` erstellt die Zusammenfassung aus Board, PRs und
  Experimenten; Blocker und Fragen an die Betreuer:innen sammeln
  ([Offene Fragen](offene-fragen.md))
- [ ] Festlegen, was live gezeigt wird (Plots, Demo) und dass es auf einem Rechner läuft

Blocker nicht bis Dienstag aufheben: sofort an `dozenten@domaenenprojekt2.de` oder in den
Matrix-Kanal Domänenprojekt2.
