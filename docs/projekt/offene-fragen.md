# Offene Fragen an die Dozenten

Hier sammeln wir Unklarheiten und halten die Antworten fest. Gestellt werden sie im
Statusmeeting oder per Matrix bzw. `dozenten@domaenenprojekt2.de`.

| # | Frage | Gestellt am | Antwort |
| --- | --- | --- | --- |
| 1 | Ist `test.csv` bereits der finale Validierungsdatensatz, oder kommt dieser später? | | |
| 2 | Meilenstein-Termine: 08./15./22.10. (Aufgabenbeschreibung), 09./16./23.10. (Folien) oder – laut Folien „bis zum Statusmeeting“ – unser Dienstagstermin 06./13./20.10.? | | |
| 3 | M3 spricht von „1–2-jährigen Sequenzen, Sommerfenster, wenigen Zeitstempeln" – ist bei uns die Reduktion der Spektralbänder gemeint? | | |
| 4 | Aus welcher Quelle stammt der Datensatz (GHISA?)? Welche Regionen Nordamerikas und Aufnahmejahre umfasst er, welche AEZ-Einteilung nutzen die Codes, und gibt es offizielle Definitionen der Stadien-Labels? | | |
| 5 | Nach welchen beobachtbaren Kriterien wurden die sechs `Stage`-Labels vergeben? Gibt es eine Zuordnung zu BBCH-, V/R- oder Zadoks-Stadien? | | |
| 6 | Entstehen fehlende Crop-Stage-Kombinationen durch die Auswahl der Stichprobe oder durch die Anbausaison der jeweiligen Kultur? | | |
| 7 | Beschreibt `Month` den tatsächlichen Aufnahmemonat oder eine nachträglich zugeordnete Kategorie? | | |
| 8 | Enthält der Datensatz wiederholte Beobachtungen desselben Feldes, und falls ja: Wie sollen diese bei der Validierung behandelt werden? | | |

## Vorschläge aus Domäne & Fernerkundung zur Teamprüfung

Diese Fragen sind **noch nicht** an die Dozenten gerichtet. Das Team prüft zunächst, ob sie
für die Aufgabenstellung relevant und präzise genug sind. Geeignete Fragen werden anschließend
gekürzt und in die Tabelle oben übernommen; nicht passende Fragen klären wir selbst durch
Recherche oder lassen sie weg.

| Bezug | Fragenvorschlag | Warum ist das relevant? | Teamentscheidung |
| --- | --- | --- | --- |
| Ergänzung zu #5: Vorverarbeitung | Gibt es Infos dazu, ob die Bandwerte bereits vorverarbeitet wurden und ob sie TOA- oder Oberflächenreflektanz zeigen? | Wir müssen vermeiden, atmosphärische Korrektur, Glättung oder Bandauswahl doppelt oder unbegründet anzuwenden. | |
| Räumliche Einheit | Steht jede Zeile für ein einzelnes 30-m-Pixel, ein Feld oder einen Durchschnitt mehrerer Pixel? | Das bestimmt, wie wir Mischpixel, räumliche Ähnlichkeit und die Aussagekraft der Validierung einordnen. | |
| Bänder und fehlende Werte | Warum enthält der Datensatz nur 198 statt 220 Hyperion-Bänder? Hängen die leeren Bänder und die Bänder um 900 nm mit dem Sensor oder der Vorverarbeitung zusammen? | Wir brauchen eine belegte Grundlage, bevor wir Bänder entfernen oder Besonderheiten als Sensorartefakte deuten. | |
