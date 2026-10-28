# Handoff: Abschlussbericht schreiben

Stand: 2026-10-28. Diese Datei zu Beginn einer neuen Berichtssession lesen.
Die Projektregeln aus [AGENTS.md](../../AGENTS.md) gelten weiterhin.

## Auftrag und Schreibstil

- Zielumfang laut Nutzerfeedback: **10 Seiten +/- 2**, also 8 bis 12 Seiten.
- Klar, knapp und einfach auf Deutsch schreiben. Kein Lehrbuch und keine vollstaendige
  Wiedergabe der Projektdokumentation oder aller Experimente.
- Entscheidungen, wichtigste Ergebnisse und ihre Grenzen erklaeren. Technische Details nur
  aufnehmen, wenn sie zum Verstaendnis oder zur Beurteilung notwendig sind.
- Kurze Absaetze und wenige Untergliederungen. Fachbegriffe beim ersten Auftreten knapp
  erklaeren; keine langen Listen von APIs, Parametern, Dateipfaden oder Run-IDs im Fliesstext.
- Abbildungen und Tabellen sollen Text ersetzen, nicht denselben Inhalt wiederholen.
- Keine kuenstliche Verkuerzung durch kleinere Schrift oder engere Seitenraender.
- Die Gesamtseitenzahl mitdenken. Ob Titelblatt, Inhaltsverzeichnis, Literatur und Anhang
  zur Vorgabe zaehlen, ist nicht geklaert; diese nicht ungeprueft als zusaetzliches Budget nutzen.

## Dateien und tatsaechlicher Stand

- Hauptdatei: [abschlussbericht.tex](abschlussbericht.tex).
- PDF: [abschlussbericht.pdf](abschlussbericht.pdf).
- Kapitel 1 ist ausformuliert. Kapitel 2 bis 8 enthalten im aktuell gelesenen LaTeX-Stand
  weitgehend nur Ueberschriften und Platzhalterkommentare, auch Kapitel 3.
- In dieser Unterhaltung wurde bereits eine Kurzfassung von Kapitel 3 mit kompakter Pipeline
  und Split-Tabelle erarbeitet. Sie ist im aktuellen Quelltext **nicht enthalten**.
  Vor Weiterarbeit aktuellen Branch, Arbeitsbaum und bei Bedarf Git-Historie pruefen.
  Nicht behaupten, diese Fassung sei noch vorhanden; nichts ungefragt wiederherstellen.
- Platzhalter sind Arbeitsnotizen, keine belegten Ergebnisse. Aeltere Hinweise, Samples-F1
  ausfuehrlich zu definieren, entsprechen nicht dem aktuellen Nutzerfeedback.
- Teambezeichnung **AE-ROOT** beibehalten. Andere Kapitel nicht ungefragt umschreiben.

## Struktur und grobes Seitenbudget

Die vorhandene Gliederung weiterverwenden. Als Arbeitsbudget fuer den Haupttext:

| Kapitel | Inhalt | Richtwert |
| --- | --- | --- |
| 1. Ziel und Domaene | Aufgabe, notwendiges Fachwissen, Untersuchungsfragen | 1 Seite |
| 2. Daten und explorative Analyse | Datenumfang, Klassenungleichgewicht, zentrale Befunde | 1 Seite |
| 3. Methodik und Validierung | Vorverarbeitung, Split/CV, Metriken, Grenzen | 1,5 bis 2 Seiten |
| 4. Modellvergleich und Auswahl | Baseline, relevante Strategien, finales Modell | 1,5 Seiten |
| 5. Datenreduktion | Leistungsverlust gegen Eingabeumfang | 1 Seite |
| 6. Ergebnisse und Interpretation | Fehler, Erklaerbarkeit, Uebertragbarkeit | 1 Seite |
| 7. Modellkarte | Kompakter Steckbrief, keine Wiederholung aller Ergebnisse | 0,5 Seiten |
| 8. Fazit und Ausblick | Fragen beantworten, wichtigste Grenze, naechste Schritte | 0,5 Seiten |

Das ergibt etwa 8 bis 8,5 Seiten Haupttext, mit Reserve fuer Vorseiten und Literatur.
Es ist ein Vorschlag, keine zusaetzliche offizielle Kapitelvorgabe. Am fertigen PDF messen.

## Vereinbarungen fuer Kapitel 3

- Die kurze Fassung ist gewuenscht, nicht die fruehere ausfuehrliche Darstellung.
- **Balanced Accuracy (BAcc) und Macro-F1 stehen im Mittelpunkt.** Je eine kurze Erklaerung
  und bei Bedarf eine kompakte Formel genuegen. BAcc mittelt klassenweise Recall;
  Macro-F1 mittelt klassenweise F1, also das harmonische Mittel von Precision und Recall.
- Kulturart, Stadium und bei Modellvergleichen das kombinierte Crop-Stage-Paar unterscheiden.
  Die kombinierte BAcc dient intern zur Modellauswahl; nicht als alleinige offizielle
  Bewertungsmetrik der Challenge darstellen.
- Samples-F1 nicht ausfuehrlich behandeln und keine eigene Herleitung dafuer aufnehmen.
  Falls fuer die Abgabe benoetigt, nur ergaenzend berichten; nicht aus dem Evaluationscode entfernen.
- **Kompakte Pipeline-Grafik und kleine Split-Tabelle beibehalten bzw. wieder aufnehmen.**
  Der Nutzer moechte beide trotz der Kuerzung ausdruecklich haben.
- Pipeline: Datenpruefung/Bereinigung -> Split -> Trainings-CV und Auswahl -> finales Training
  -> Holdout-Bewertung. Den separaten Holdout-Pfad sichtbar machen; er darf nicht durch die
  CV oder Hyperparametersuche laufen. Vorverarbeitung gehoert in jedes Training.
- Split-Tabelle mit drei Zeilen: Training, interner Holdout und Trainings-CV.
  Zahlen nicht nochmals vollstaendig im Fliesstext wiederholen.
- Keine grosse Pipeline-Grafik mit allen Software- und Trackingdetails.
- Die Gruppen-CV-Grafik gehoert eher zu Kapitel 6 als zur knappen Methodenbeschreibung.
  Sie muss nicht in Kapitel 3 wiederhergestellt werden.

## Belege und fachliche Grenzen

Behauptungen mit **Dokumentation, Python-Implementierung und Notebook-Konfigurationen**
abgleichen. Nicht alle Dateien bei jeder kleinen Textaenderung erneut vollstaendig lesen,
sondern die fuer die jeweilige Aussage entscheidenden Stellen pruefen.

- Aufgabe und Abgabe: [docs/projekt](../../docs/projekt/).
- Fachwissen: [docs/domaene](../../docs/domaene/).
- Daten und EDA: [docs/daten](../../docs/daten/).
- Ergebnisse: [Experimentprotokoll](../../docs/modelle/experimente.md) und
  [Ansatzseiten](../../docs/modelle/ansaetze/).
- Tatsaechliches Verhalten: [src/awp2](../../src/awp2/), besonders Datenaufteilung,
  Vorverarbeitung, Experimentablauf und Metriken; konkrete Varianten in [notebooks](../../notebooks/).
- Bei Widerspruechen zwischen Quellen unterscheiden: aktueller Code, historische
  Konfiguration und tatsaechlich dokumentierter Lauf sind nicht dasselbe.
- Keine Kennzahlen, Run-IDs, Signifikanz oder externe Testergebnisse erfinden.
  Notebooks mit entfernten Ausgaben belegen Aufrufe, aber nicht allein deren Ausfuehrung.
- CV und Holdout klar trennen. Fruehere 70/30-Laeufe nicht direkt mit 80/20 vergleichen.
  Gleicher Seed allein garantiert keine vollstaendige Reproduzierbarkeit.
- Wiederholte Modellvergleiche auf demselben Holdout begrenzen dessen Unabhaengigkeit.
  Nicht als voellig unberuehrten Abschlusstest bezeichnen.
- AEZ-Monat-Zellen sind Kontextgruppen, keine nachgewiesenen Aufnahme- oder Feldkennungen.
  Hohe zufaellige CV-Werte belegen keine Uebertragbarkeit auf neue Regionen oder Zeitpunkte.
- Nicht beobachtete Crop-Stage-Paare sind nicht automatisch fachlich unmoeglich.
- Weniger PCA-Komponenten bedeuten nicht automatisch weniger benoetigte Messbaender.
- Befunde, plausible Erklaerungen und nicht gepruefte Annahmen sprachlich trennen.

Zuletzt am 2026-10-27 an lokalen Daten gepruefte Referenzwerte: 5.591 Rohzeilen,
zwei entfernte Duplikate, 5.589 bereinigte Beobachtungen, 198 Bandspalten, 131 nutzbare
Baender, 4.471 Trainings- und 1.118 Holdout-Zeilen, Seed 42, fuenf Trainingsfolds,
23 beobachtete Zielpaare. Vor Uebernahme in einen geaenderten Datenstand erneut pruefen.
Historische Gruppen-CV-Zahlen waren in Notebook und Doku konsistent, die zugehoerigen
MLflow-Laeufe jedoch nicht im lokalen Store vorhanden. Nicht als lokal bestaetigte Laeufe ausgeben.

## Bilder, Tabellen und Literatur

- Kapitel 2: bevorzugt eine aussagekraeftige Klassenverteilung.
- Kapitel 3: kompakte Pipeline und dreizeilige Split-Tabelle.
- Kapitel 4: kleine Vergleichstabelle statt jeder einzelnen Suchvariante.
- Kapitel 5: BAcc gegen Bandanzahl als Trade-off-Grafik.
- Kapitel 6: gezielt Confusion Matrix oder Gruppen-CV, sofern im Text interpretiert.
- Nicht jede moegliche Grafik aufnehmen. Aussage, Lesbarkeit und Platzbedarf entscheiden.
- Jede Abbildung im Text referenzieren und kurz einordnen; normale Bildunterschrift darunter.
  Abbildungsnummer nicht als eigene zentrierte Zeile von der Beschreibung trennen.
- Vorhandene Projektgrafiken bevorzugen. Keine Modelle neu trainieren, nur um einen
  Berichtstext oder eine bereits vorhandene Grafik zu uebernehmen.
- Quellen im Text mit LaTeX-Zitaten belegen. Knipling (1970) und Allen et al. (1998)
  wurden auf Nutzerwunsch entfernt; nicht ungefragt wieder einfuegen.

## Arbeitsablauf pro Session

1. Diese Datei, AGENTS.md und den aktuellen Abschnitt lesen. Branch und Arbeitsbaum pruefen;
   Nutzerarbeit nicht rueckgaengig machen. Keine Branchwechsel, Commits oder Pushes ohne Auftrag.
2. Den gewuenschten Abschnitt und die dafuer noetigen Quellen lokal abgleichen.
   Verifizierte Fakten und offene Punkte unterscheiden.
3. Nur den beauftragten Abschnitt knapp ausarbeiten. Entscheidungen und Ergebnisse priorisieren;
   Details in vorhandener Dokumentation lassen oder nur bei Bedarf in den Anhang nehmen.
4. Nach LaTeX-Aenderungen kompilieren, Seitenumfang, Referenzen, Formeln und Layout kontrollieren.
   Keine langen Leerseiten oder einzelnen Absatzreste durch unguenstige Platzierung akzeptieren.
5. Die gepruefte PDF aktualisieren. Kurz berichten, was geaendert und was nicht geprueft wurde.

LaTeX-Stil der Datei beibehalten, einschliesslich ASCII-Umlautbefehlen.
Mit `-cd` bauen, damit relative Grafikpfade korrekt aufgeloest werden. Beispiel ab Repo-Wurzel
fuer PowerShell; Hilfsdateien bleiben im temporaeren Ordner:

```powershell
latexmk -cd -pdf -quiet -interaction=nonstopmode -halt-on-error -outdir="$env:TEMP" -jobname=awp2-abschlussbericht-review reports/abschlussbericht/abschlussbericht.tex
```

Nach erfolgreichem Build und Layoutkontrolle:

```powershell
Copy-Item -LiteralPath "$env:TEMP/awp2-abschlussbericht-review.pdf" -Destination reports/abschlussbericht/abschlussbericht.pdf
```

Nur Markdown am Handoff geaendert: kein PDF-Neubau erforderlich.

## Startauftrag fuer eine neue Session

> Lies zuerst reports/abschlussbericht/HANDOFF.md und AGENTS.md. Schreibe anschliessend
> Kapitel [Nummer/Titel] in reports/abschlussbericht/abschlussbericht.tex auf Basis der
> vorhandenen Dokumentation, Python-Implementierung und relevanten Notebooks.
> Halte den Bericht klar und knapp, insgesamt 10 +/- 2 Seiten. BAcc und Macro-F1 stehen
> im Mittelpunkt. Behalte die kompakte Pipeline und Split-Tabelle in Kapitel 3 bei.
> Aendere andere Kapitel nicht ungefragt und pruefe danach Build und PDF-Layout.