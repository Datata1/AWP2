# AutoML: umfassende Suche auf 80/20

## Status

**Breites Screening und fokussierter CV-Lauf abgeschlossen, Gewinner einmal validiert.**
Ausgangspunkt ist der Branch `refactor/preprocessing-modularization`; die Vorbereitung liegt
auf `exp/automl-exhaustive-80-20`.

Der fokussierte Lauf schloss alle **384 Kandidaten ohne Fehler** ab. Der per CV gewählte
Gewinner ist eine kombinierte RBF-SVM (`C=100`, `gamma=scale`) mit Vorverarbeitung 5:
CV-BAcc kombiniert **0,86682 ± 0,01744** (MLflow `a08557a27a84465eb1a0667b312ef39e`).
Die einmalige Validierung `automl_final` (MLflow `c99e3915704a4131bf6da0a795fb50ee`)
erreicht BAcc Crop **0,9325**, Stage **0,9204**, kombiniert **0,8832**; Macro-F1 Crop
0,9170, Stage 0,9107, kombiniert 0,8576; Samples-F1 0,9217 und keine ungültigen Paare.
[Vollständige Einstellungen und Einordnung](kombiniert.md#rbf-svm-aus-dem-fokussierten-automl-lauf).
Die Suche ist eingefroren. Die Befehle unten dokumentieren den Ablauf, sie sind keine
Aufforderung zur Wiederholung der Holdout-Auswertung.

Nach der LPP-Stabilisierung sind 200 hierarchische und 150 kombinierte Kandidaten erfolgreich:

| Screening | Kandidaten | Beste CV-BAcc kombiniert | Fold-Std | MLflow des besten Kandidaten |
| --- | --- | --- | --- | --- |
| Hierarchisch, SVM → LDA | 200 | 0,84554 | 0,01536 | `123cbdba465549389ac7e6226bec8df3` |
| Kombiniert, Extra Trees | 150 | 0,84562 | 0,02164 | `bd0211fdd51c47ce9da61ea90d01f6a7` |

Der Unterschied beträgt nur 0,00008 beziehungsweise 0,008 Prozentpunkte. Daraus folgt kein
belastbarer Vorteil eines Ansatzes; die Screening-Budgets waren zudem unterschiedlich verteilt.
Der frühere Lauf vor der LPP-Stabilisierung hatte 146 erfolgreiche und 54 fehlgeschlagene
Kandidaten. Er bleibt ein separater Suchstand.

Die Suche ist ein Vergleich bestehender Modellierungsansätze, kein weiterer Klassifikator.
Ein vollständiger Lauf kann weit über das verbleibende Zeitbudget bis zur Abgabe hinausgehen.
Deshalb zuerst die Vorschau und einen begrenzten technischen Probelauf verwenden.

## Versuchsprotokoll

- Gemeinsame, reproduzierbare 80/20-Artefakte; kein eigener Split. Die Software lehnt
  abweichende Artefakte ab, anstatt sie automatisch zu überschreiben.
- Auswahl ausschließlich nach mittlerer kombinierter Balanced Accuracy auf den fünf
  Trainingsfolds. Alle Transformationen werden innerhalb der Pipeline je Fold angepasst.
- Alle Kandidaten werden mit `awp2.experiment.tune()` und in MLflow erfasst.
- Nur der globale CV-Gewinner wird mit `run()` einmal auf den 20 % Validierung bewertet.
  Gleichstände entscheidet die stabile Kandidaten-ID, nicht ein Holdout-Score.
- Eine umfangreiche Suche kann die CV-Schätzung optimistisch machen. Das vielfach zuvor
  betrachtete gemeinsame Holdout ist außerdem kein unangetasteter abschließender Testdatensatz.
- Jeder Hyperparameterkandidat ist ein Checkpoint. SQLite speichert erfolgreiche und
  fehlgeschlagene Kandidaten; Fehler werden nicht als gültige Scores gewertet.
- Hashes von Quellcode, Lockfile und Datenartefakten sowie Git-Commit und Suchkonfiguration
  verhindern das Vermischen verschiedener Versuche beim Fortsetzen.

## Suchraum

### Begrenztes Folgeprofil: `focused`

Das breite Screening wird nicht unbegrenzt fortgesetzt. Das aus den **CV-Ergebnissen**, nicht
aus dem Holdout, abgeleitete Profil prüft acht Modellkonfigurationen auf denselben sechs
Vorverarbeitungen und jeweils acht Hyperparametereinstellungen:

**8 × 6 × 8 = 384 Kandidaten = 1.920 CV-Pipeline-Fits.**

| Modellkonfiguration | Variierte Parameter; übrige Werte fest | Kandidaten |
| --- | --- | --- |
| Kombiniert: Extra Trees | `max_features` 0,5/1,0 × `min_samples_leaf` 1/2/5/10; 600 Bäume, keine Tiefengrenze | 48 |
| Kombiniert: Random Forest | `max_features` sqrt/0,5 × `min_samples_leaf` 1/2/5/10; 600 Bäume, keine Tiefengrenze | 48 |
| Kombiniert: RBF-SVM | `C` 30/100/300/1000 × `gamma` scale/0,001 | 48 |
| Kombiniert: HGB | Lernrate 0,03/0,1 × Blattanzahl 31/63 × L2 1/10; 300 Iterationen | 48 |
| Hierarchisch: SVM → LDA | Kultur-`C` 30/100/300/1000 × Shrinkage 0,01/auto; Kultur-`gamma` 0,1 | 48 |
| Hierarchisch: SVM → Extra Trees | Kultur-`C` 100/1000 × Stadium-Tiefe 10/keine × Stadium-Blattminimum 1/2; Kultur-`gamma` 0,1, Stadium 300 Bäume und alle Features | 48 |
| Hierarchisch: Extra Trees → HGB | Kultur-Featureanteil 0,5/1,0 × Stadium-Iterationen 100/300 × Stadium-L2 1/10; Kultur 600 Bäume, Tiefe 30, Blattminimum 2; Stadium Lernrate 0,03 und 63 Blätter | 48 |
| Hierarchisch: SVM → SVM | Kultur-`C` 30/100 × Stadium-`C` 100/300 × Stadium-`gamma` 0,003/0,01; Kultur-`gamma` 0,01 | 48 |

SVM → SVM ist eine im Projekt etablierte Vergleichskonfiguration, obwohl das breite
Screening für dieses Paar keinen Spitzenwert gefunden hat. Damit hängt die Vorauswahl
nicht ausschließlich von den zufällig im Screening besuchten Parametern ab.

Die sechs Vorverarbeitungen verwenden alle Bänder und AEZ mit zyklischem Monat:

| Nr. | Vorverarbeitung | Zweck |
| --- | --- | --- |
| 1 | Standardisierung, sonst keine Zusatzmerkmale | Einfache Vergleichsbasis |
| 2 | Wie 1, zusätzlich PCA mit 20 Komponenten | Etablierte spektrale Reduktion |
| 3 | Standardisierung, Savitzky-Golay 11/3, LPP 40 mit 20 Nachbarn, parallele PC_1/PC_2 | Exakte Vorverarbeitung des kombinierten Screening-Gewinners |
| 4 | Wie 3, aber ohne LPP | Kontrollierter Vergleich des LPP-Beitrags |
| 5 | MinMax, Savitzky-Golay 11/3, Albedo, Indizes, AEZ×Monat, Interpolationsgrenze 10 nm | Exakte Vorverarbeitung des hierarchischen Screening-Gewinners |
| 6 | Wie 5, aber ohne Vegetationsindizes | Kontrollierter Vergleich des Index-Beitrags |

Für 1–4 gilt die Interpolationsgrenze 15 nm. SNV ist in allen sechs Varianten aus.
Komponentenzahlen, Metadaten und Glättung werden nicht weiter frei gekreuzt.
Die Raster stehen in `AUTOML_FOCUSED_PREPROCESSING` und `AUTOML_FOCUSED_MODEL_GRIDS`.
Beide Screening-Gewinner sind einschließlich ihrer Parameter exakt enthalten.

Die Reihenfolge wechselt zwischen den Modellkonfigurationen, sodass bereits die ersten acht
Kandidaten alle acht abdecken. Das gleiche Kandidatenbudget bedeutet keine gleiche Laufzeit,
keine identische Optimierungschance und keinen statistisch unabhängigen Ansatzvergleich.
Die Anpassung auf Basis derselben CV-Folds kann CV-Scores optimistisch machen.

```powershell
uv run python -m awp2.automl plan --profile focused
uv run python -m awp2.automl search --profile focused
uv run python -m awp2.automl report --profile focused
```

Der Standard-Ausgabeordner ist `reports/automl-focused-80-20`. **Anders als beim breiten
Profil** kann dieser vollständige, endliche Suchlauf ohne `--max-tasks` gestartet werden;
er endet nach 384 Versuchen. Die Laufzeit hängt von Rechner und Modell ab. Ein Abbruch lässt sich
mit demselben Befehl fortsetzen. Optional begrenzt `--max-tasks` zusätzliche Versuche je Aufruf.

Die alten Datenbanken bleiben unverändert. Durch die Profil-Erweiterung hat sich der
Quellcode-Fingerprint geändert: Alte Screening-Stände können mit diesem Code nicht weiter
trainiert oder validiert werden; `report --output <alter-Ordner>` bleibt lesbar. Keine alten
Scores in den neuen Suchstand kopieren und keine Fingerprints manuell überschreiben.

Abdeckung und Fehler wurden nach Abschluss geprüft und die Auswahl vor der einmaligen
Bewertung mit `evaluate --profile focused` festgelegt. Diese Bewertung ist abgeschlossen;
nicht erneut ausführen oder die Sperrdatei löschen. Für einen anderen, frühzeitig beendeten
Versuch wäre zusätzlich `--allow-partial` nötig; er müsste als unvollständig dokumentiert werden.

### Breites Screening: `broad`

Die endlichen Raster stehen in `awp2.config` (`AUTOML_*`); die Kandidatenerzeugung in
`awp2.automl`. **Vollständig** bedeutet alle gültigen Kombinationen genau dieses Rasters,
nicht alle möglichen reellen Hyperparameterwerte oder Architekturen.

| Dimension | Werte |
| --- | --- |
| Modellfamilien | SVM, Random Forest, Extra Trees, HGB, logistische Regression, kNN, sklearn-MLP, ausgeglichene LDA, PLS-DA, Gaussian Naive Bayes |
| Ansatz | Kombinierte Klasse; hartes hierarchisches Routing mit allen unterstützten Kultur-/Stadium-Modellpaaren |
| Metadaten | Ohne; numerischer/zyklischer Monat jeweils mit/ohne AEZ×Monat; vier reine Metadaten-Baselines |
| Spektralregionen | Alle 127 nichtleeren Teilmengen der sieben Projektregionen; optional nur alle Bänder zusammen |
| Skalierung | Keine, Standard, MinMax, Mittelwertzentrierung |
| SNV / Albedo | Jeweils aus/an |
| Glättung | Keine; Savitzky-Golay (Fenster, Grad): (5,2), (7,2), (11,2), (11,3) |
| Zusätzliche Features | Vegetationsindex-Katalog und parallele PC_1/PC_2 jeweils aus/an |
| Reduktion | Keine; PCA, ICA, NMF, OSP, VSRP mit 10/20/40/80 Komponenten; LPP zusätzlich mit 5/10/20 Nachbarn; DBN (64,20), (128,40), (128,64,20) |
| Interpolationsgrenze | 10/15/20 nm |
| Modellparameter | Vollständiges kartesisches Produkt der Werte in `AUTOML_MODEL_GRIDS`, für hierarchische Modelle unabhängig je Stufe |

Pydantic entfernt unzulässige Vorverarbeitungskombinationen. Datenabhängig unmögliche
Konfigurationen, etwa zu viele Komponenten für eine schmale Spektralregion, werden beim
CV-Lauf als fehlgeschlagen dokumentiert. Die Vorschau zählt auch diese Versuche.
Die Zahl der CV-Fits bezeichnet Pipeline-Fits; ein hierarchischer Fit trainiert mehrere Modelle.

### LPP-Stabilisierung und getrennte Suchstände

In den ersten 200 Versuchen scheiterten 54 von 75 LPP-Kandidaten, während alle 125
Kandidaten ohne LPP erfolgreich waren. Für 49 Fehler ist ein singuläres verallgemeinertes
Eigenwertproblem im Traceback belegt; fünf weitere speicherten nur einen nicht endlichen
CV-Score. Deren genaue Fold-Ursache ist damit nicht belegt.

LPP löst das Problem im numerisch unabhängigen Teilraum der gradgewichteten Trainingsmatrix
(siehe [Vorverarbeitungsentscheidungen](../../daten/pipeline-entscheidungen.md)). Synthetische
Regressionstests decken abhängige und konstante Bandspalten, verschiedene Skalierungen und
die Übereinstimmung mit der Vollrang-Lösung ab. Anfragen oberhalb des numerischen Rangs
werden weiterhin als ungültig abgelehnt; eine fehlerfreie Suche ist dadurch nicht garantiert.

Die bisherigen Dateien unter `reports/automl-80-20` bleiben unverändert. Wegen des
Quellcode-Fingerprints muss die korrigierte Suche einen **neuen Ordner** verwenden:

```powershell
uv run python -m awp2.automl search --full-spectrum-only --max-tasks 100 --output reports/automl-80-20-lpp-fixed
uv run python -m awp2.automl report --output reports/automl-80-20-lpp-fixed
```

Dies beginnt eine neue Suche mit derselben deterministischen Kandidatenreihenfolge, nicht
bei Kandidat 201. Die bisher gespeicherten Scores werden nicht übernommen oder überschrieben.
Auch `evaluate` für den alten Suchstand ist mit verändertem Quellcode gesperrt; für dessen
Reproduktion ist der damalige Code-Stand erforderlich. Noch keinen Holdout-Lauf starten.

Klassenungleichgewicht wird durch `class_weight`, gleiche LDA-Priors bzw. ausgeglichene
Sample-Gewichte für MLP/GNB berücksichtigt; bei kNN durch die Auswahlmetrik und optional
Distanzgewichte. Hierarchische MLP/GNB-Paare mit kNN/LDA/PLS-DA sind ausgeschlossen,
weil diese Partner keine Sample-Gewichte unterstützen. Sämtliche aufgenommenen Wrapper
sagen nur im jeweiligen Training beobachtete Kultur-/Stadium-Paare vorher.

**Nicht enthalten:** CNN/LSTM, Multi-Task-Netze, Ensemble-Suche und weiches Routing;
kategorischer Embedding-Kontext; der veraltete `scale`-Schalter; beliebige einzelne
Band-/Indexteilmengen. Diese benötigen eigene, zur Eingabestruktur passende Suchräume.

## Ausführen

Alle Befehle im Repository-Hauptverzeichnis, beispielsweise in PowerShell. Bei eingeschränkten
Läufen dieselben Modell-/Ansatz-/Regionsoptionen bei `plan`, `search` und `evaluate` angeben.
`--output` trennt unabhängige Versuche; die Dateien werden nicht automatisch gelöscht.

### Vorbereitung und Vorschau

```powershell
uv sync
uv run python -m unittest discover -s tests -p test_automl.py
uv run python -m awp2.automl plan
```

Die Tests verwenden Mocks für Training, MLflow und Datenzugriff. `plan` erzeugt nur den
Suchraum und lädt keine Datensätze; es zeigt Kandidatenzahl und Zahl der CV-Pipeline-Fits.
Auch ein riesiger Suchraum wird ohne Materialisierung aller Hyperparameterkombinationen geplant.

Nur falls die gemeinsamen Artefakte fehlen oder nicht 80/20 verwenden:

```powershell
uv run python -m awp2.data.artifacts --validation-size 0.2
```

Dies überschreibt die gemeinsamen abgeleiteten Split-Artefakte, nicht die Rohdaten.
Nicht während anderer Experimente ausführen.

### Technischer Probelauf

```powershell
uv run python -m awp2.automl search --max-tasks 2 --output reports/automl-pilot
uv run python -m awp2.automl report --output reports/automl-pilot
```

Das sind zwei **Kandidaten**, also zehn CV-Pipeline-Fits, keine finale Holdout-Bewertung.
Der Probelauf prüft die Ausführung, liefert aber keine belastbare Gesamtlaufzeitprognose.
Bei Fehlern zuerst den CSV-Export prüfen. Tracking-Probleme stoppen die Suche, damit
Ergebnisse nicht unbemerkt ohne MLflow fortgeführt werden.

### Vollständige Suche

```powershell
uv run python -m awp2.automl search
uv run python -m awp2.automl report
```

Mit Strg+C unterbrechen und denselben Suchbefehl erneut ausführen. Abgeschlossene Kandidaten
werden übersprungen; nur ein gerade laufender Kandidat kann erneut berechnet werden.
Ein harter Prozessabbruch kann eine `search-*.lock` hinterlassen. Sie erst entfernen, wenn
der zugehörige Prozess sicher beendet ist. Quellcode und Umgebung während des Versuchs
nicht ändern; bei Änderungen einen neuen Ausgabeordner verwenden.

Die Kandidatenreihenfolge ist eine deterministische Permutation des gesamten Suchraums.
Ein begrenzter Lauf bleibt deshalb nicht nur am Anfang des Rasters hängen, garantiert aber
keine gleichmäßige Abdeckung oder einen Mindestumfang je Modellfamilie.

### Laufbudget statt Vollständigkeit

```powershell
uv run python -m awp2.automl search --max-tasks 1000
uv run python -m awp2.automl report
```

`--max-tasks` begrenzt zusätzliche Kandidaten **pro Aufruf**, nicht Sekunden oder die
Gesamtzahl gespeicherter Versuche. Die Suche ist damit ausdrücklich nicht vollständig.
Ein einzelner Fit kann sehr lange dauern; es gibt keinen harten Fit-Timeout.

Alternativ den Suchraum bewusst kleiner festlegen und einen separaten Ordner verwenden:

```powershell
uv run python -m awp2.automl plan --models svm et hgb --approaches combined --full-spectrum-only
uv run python -m awp2.automl search --models svm et hgb --approaches combined --full-spectrum-only --output reports/automl-classical
```

Auch dieser reduzierte Suchraum kann groß sein. Für unabhängige Worker gibt es `--shards N`
und `--shard 0` bis `N-1`, mit identischen restlichen Optionen und demselben Ausgabeordner
auf demselben Rechner. Shard-Anzahl während aktiver Worker nicht ändern. Bäume verwenden
bereits alle CPU-Kerne; mehrere Worker können den Rechner überlasten. Vor der Auswertung
alle Worker beenden. Kein gemeinsamer SQLite-Ordner auf Netzlaufwerken.

### Gewinner einfrieren und einmal validieren

Nach vollständiger Suche:

```powershell
uv run python -m awp2.automl evaluate
uv run python -m awp2.automl report
```

Falls das Abgabebudget vorher erreicht ist: Worker stoppen und die **unvollständige** Suche
explizit einfrieren:

```powershell
uv run python -m awp2.automl evaluate --allow-partial
uv run python -m awp2.automl report
```

Die Datei `validation.json` sperrt weitere Suche und erneute Holdout-Auswertung, auch wenn
die erste Auswertung abbricht. In diesem Fall zunächst MLflow und den Fehler prüfen und
die Sperre nicht blind löschen. Ein fehlgeschlagenes Training ohne ausgewertete Holdout-
Vorhersagen muss von einer bereits erfolgten Bewertung unterschieden werden.

## Ergebnisdokumentation

Im jeweiligen Ausgabeordner entstehen:

- `search.sqlite`: Manifest, vollständige Konfigurationen, CV-Scores, Laufzeiten,
  Fehlermeldungen und MLflow-IDs.
- `candidates.csv`: Alle abgeschlossenen Kandidaten mit Konfiguration als JSON.
- `report.md`: Top-20 nach CV, Gewinnerkonfiguration und gegebenenfalls alle finalen
  Kultur-/Stadium-Metriken.
- `validation.json`: Eingefrorener Gewinner, Versuchsabdeckung und einmaliges Holdout-Ergebnis.

Den Bericht nach dem Lauf prüfen und die wesentlichen Ergebnisse hier ergänzen:

1. Anzahl erfolgreicher/fehlgeschlagener/geplanter Kandidaten und tatsächlich getesteter
   Familien; vollständige oder vorzeitig beendete Suche klar benennen.
2. Gewinnerfamilie, Vorverarbeitung, Parameter, CV-Mittelwert und Fold-Streuung.
3. Holdout: BAcc und Macro-F1 für Crop und Stage sowie kombinierte Metriken und Samples-F1;
   nur mit früheren Läufen auf demselben 80/20-Split vergleichen.
4. Eine Zeile mit tatsächlichen Scores und MLflow-ID im [Experimentlog](../experimente.md),
   Details auf der zutreffenden Seite [Kombiniert](kombiniert.md) oder
   [Hierarchisch](hierarchisch.md), anschließend die [Ansatzübersicht](index.md) aktualisieren.
5. Konfusionsmatrizen aus dem finalen MLflow-Lauf sichten; Fehlermuster und Grenzen der
   großen Suche im Abschlussbericht festhalten. Keine Platzhalterwerte als Resultate übernehmen.

Ergebnisse liegen lokal unter `reports/`; keine Daten, Modelle oder große Suchdatenbanken
committen. Nur die ausgewählte Ergebniszusammenfassung in die deutsche Projektdokumentation
übernehmen.

## Sicherung und Wiederverwendung

Stand 26.10.2026: Die eingefrorene Pipeline aus Run `c99e3915704a4131bf6da0a795fb50ee`
ist unabhängig von der lokalen MLflow-Datenbank exportiert:

| Lokal gespeicherte Datei | Zweck |
| --- | --- |
| `models/automl_final/` | Entpacktes, eigenständiges Modellpaket mit Anleitung |
| `models/automl_final.zip` | Transport- und Sicherungsarchiv |
| `models/automl_final.zip.sha256` | SHA-256-Prüfsumme des Archivs |

Das Paket enthält die vollständige **trainierte Vorverarbeitung und SVM**, den tatsächlichen
`awp2`-Quellcode inklusive uncommittierter Änderungen, `pyproject.toml`, `uv.lock`,
Python-Version **3.12.13**, Paketversionen, Git-HEAD und Änderungsstatus. Datei-Prüfsummen
identifizieren den gesicherten Stand eindeutig; Git-HEAD allein genügt bei diesem Arbeitsbaum
nicht. Gewinnerkonfiguration, Suchmanifest und Holdout-Ergebnis liegen unter `provenance/`.
Der vollständige Holdout mit Referenzvorhersagen liegt unter `reference/` und dient nur dem
Wiederherstellungstest, nicht weiterem Tuning. Das Paket gehört nicht ins öffentliche Git.

**Geprüft:** Das ZIP wurde außerhalb des Repositories entpackt und eine neue Umgebung mit
107 Paketen aus der mitgelieferten Lockdatei aufgebaut (`--frozen --offline`, ohne Dev-/Docs-
Gruppen; Pakete aus dem lokalen uv-Cache). Der mitgelieferte Code lädt das Modell ohne
Zugriff auf die Original-MLflow-Datenbank. Alle **1.118 Vorhersagen und acht Metriken** stimmen
exakt überein. Auch der separate `predict`-Befehl wurde mit Parquet-Eingabe und -Ausgabe
geprüft, einschließlich der Zeilenindizes. Es gab kein neues Training oder Tracking eines
Bewertungslaufs. `verification.json` im Paket protokolliert den isolierten Prozesstest beim
Export; der zusätzliche Test in frischer Umgebung wurde anschließend auf dem ZIP ausgeführt.

Nach dem Entpacken im Ordner `automl_final/`:

```powershell
uv sync --project project --frozen --no-default-groups
uv run --project project --no-sync python model_bundle.py verify .
uv run --project project --no-sync python model_bundle.py predict . `
  --input neue_features.parquet --output vorhersagen.parquet
```

Die Feature-Namen und Datentypen stehen in `bundle.json/input_schema`. Erwartet werden die
Bandspalten sowie AEZ und Month, ohne Zielspalten und ohne selbst gefittete Vorverarbeitung.
Ein vorhandenes Ausgabe-Parquet wird nicht überschrieben. Für einen weiteren Export dient
`uv run python tools/model_bundle.py export --output models/anderer-paketname`; bestehende
Pakete, ZIPs und Prüfsummendateien werden nicht überschrieben.

**Noch außerhalb dieses Rechners sichern:** ZIP und SHA-256-Datei gemeinsam auf einen zweiten
Speicherort kopieren und dort die Prüfsumme vergleichen. Die lokale Kopie ist kein Schutz
gegen Geräteausfall. Ohne uv-Cache ist zur Installation Internetzugang nötig; eine
Python-Installation ist nicht im Archiv enthalten. Nur vertrauenswürdige Modellpakete laden:
Cloudpickle kann Code ausführen, und Prüfsummen beweisen keine vertrauenswürdige Herkunft.