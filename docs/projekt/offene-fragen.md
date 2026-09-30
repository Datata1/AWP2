# Offene Fragen an die Dozenten

Hier sammeln wir Unklarheiten und halten die Antworten fest. Gestellt werden sie im
Statusmeeting oder per Matrix bzw. `dozenten@domaenenprojekt2.de`.

| # | Frage | Gestellt am | Antwort |
| --- | --- | --- | --- |
| 1 | Ist `test.csv` bereits der finale Validierungsdatensatz, oder kommt dieser später? | | |
| 2 | Meilenstein-Termine: 08./15./22.10. (Aufgabenbeschreibung), 09./16./23.10. (Folien) oder – laut Folien „bis zum Statusmeeting“ – unser Dienstagstermin 06./13./20.10.? | | |
| 3 | Was genau ist mit dem „70/30 Clusterplot-Split" für die Baseline in M1 gemeint? | | |
| 4 | M3 spricht von „1–2-jährigen Sequenzen, Sommerfenster, wenigen Zeitstempeln" – ist bei uns die Reduktion der Spektralbänder gemeint? | | |
| 5 | Aus welcher Quelle stammt der Datensatz (GHISA?), und gibt es offizielle Definitionen der Stadien-Labels und AEZ? | | |
| 6 | Welches Format soll der M1-Kurzreport haben (PDF, Folien, Doku-Seite), welcher Umfang, und wie wird er abgegeben? | | |

## Vorschläge aus Domäne & Fernerkundung zur Teamprüfung

Diese Fragen sind **noch nicht** an die Dozenten gerichtet. Das Team prüft zunächst, ob sie
für die Aufgabenstellung relevant und präzise genug sind. Geeignete Fragen werden anschließend
gekürzt und in die Tabelle oben übernommen; nicht passende Fragen klären wir selbst durch
Recherche oder lassen sie weg.

| Bezug | Fragenvorschlag | Warum ist das relevant? | Teamentscheidung |
| --- | --- | --- | --- |
| Ergänzung zu #5: Vorverarbeitung | Können Sie uns die Originalquelle oder technische Metadaten nennen, aus denen hervorgeht, ob die Bandwerte TOA- oder Oberflächenreflektanz sind und welche Vorverarbeitung bereits erfolgt ist? | Wir müssen vermeiden, atmosphärische Korrektur, Glättung oder Bandauswahl doppelt oder unbegründet anzuwenden. | |
| Räumliche Einheit | Beschreibt eine Zeile ein einzelnes 30-m-Hyperion-Pixel, ein gemitteltes Feldspektrum oder eine andere Aggregation? Gibt es Informationen zu Aufnahmejahren, Regionen oder Feldern? | Das bestimmt, wie wir Mischpixel, räumliche Ähnlichkeit und die Aussagekraft der Validierung einordnen. | |
| Bänder und fehlende Werte | Warum liegen 198 Bandspalten vor, obwohl Hyperion 220 Bänder erfasste? Sind die 67 vollständig leeren Bänder und die eng benachbarten Bänder um 900 nm erwartete Folgen der Datenaufbereitung oder des Sensors? | Wir brauchen eine belegte Grundlage, bevor wir Bänder entfernen oder Besonderheiten als Sensorartefakte deuten. | |
| Metadaten als Eingaben | Sind `AEZ` und `Month` beim späteren Validierungsdatensatz verlässlich verfügbar und ausdrücklich als Modellmerkmale vorgesehen? | Die Metadaten könnten nützlich sein, aber auch eine Vorhersage über Ort oder Monat statt über das Spektrum ermöglichen. | |
