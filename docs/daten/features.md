# Mögliche Merkmale für die Modellierung

Diese Seite sammelt Merkmalsmengen, die sich aus der EDA, der Projektbeschreibung und der
fachlichen Einordnung hyperspektraler Vegetationssignaturen ergeben. Sie ist ein Arbeitsplan,
nicht eine Liste bereits ausgewählter Eingaben: Ein Merkmal bleibt nur dann in einem Ansatz,
wenn es im gemeinsamen Trainingsanteil per Cross-Validation einen nachvollziehbaren Nutzen
zeigt.

## Grundregel für jeden Vergleich

Eine Zeile beschreibt eine Reflexionskurve einer landwirtschaftlichen Fläche über die
Wellenlänge sowie `AEZ` und `Month`. Die Bandspalten sind keine Zeitreihe. Das Modell darf daher
die Reihenfolge der Wellenlängen nutzen, aber keine zeitliche Abfolge innerhalb eines Spektrums
annehmen.

Vorverarbeitung und Merkmalsbildung werden immer in `PreprocessingConfig` und einer
sklearn-`Pipeline` umgesetzt. Im jeweiligen Trainingsfold werden vollständig leere Bänder
erkannt, einzelne Lücken bei Stützbändern mit höchstens 15 nm Abstand interpoliert und
bei größeren Abständen mit dem nächstgelegenen Messwert gefüllt; erst danach werden neue
Merkmale berechnet. Der 30-%-Holdout
bleibt bis zur einmaligen Bewertung des per Cross-Validation ausgewählten Ansatzes unangetastet.
Details zum Ablauf stehen in der [Pipeline](pipeline.md) und in der
[ML-Aufgabe](../domaene/ml-aufgabe.md#data-leakage).

## Ausgangspunkt: verfügbare Rohinformationen

| Merkmal | Status und Begründung | Rolle im Vergleich |
| --- | --- | --- |
| Verfügbare Spektralbänder | **Feste Referenz.** 131 von 198 Bändern enthalten mindestens eine Messung. Mittelwerte unterscheiden sich nach Kultur und teils nach Stadium; einzelne Bereiche überlappen aber. | Rohspektren sind die Baseline für jede weitere Merkmalsmenge. Vollständig leere Bänder sind keine Eingaben. |
| `AEZ` | **Kontextmerkmal.** Die Crop-Anteile unterscheiden sich deutlich zwischen den Zonen. | One-Hot-kodiert und stets gegen die Variante ohne Metadaten vergleichen. |
| `Month` | **Kontextmerkmal.** Innerhalb einer Kultur konzentrieren sich Stadien oft auf wenige Monate. | Standardmäßig durch die zyklischen Merkmale `Month_sin` und `Month_cos` mit Jahresperiode 12 ersetzen; die numerische Variante bleibt über `use_cyclic_month=False` vergleichbar. |

Die EDA beschreibt die vollständigen Befunde zu Datenqualität, Spektren und Metadaten in
[Explorative Datenanalyse](eda.md). `AEZ` und `Month` können zulässige Information sein, aber
auch ein fragiles Shortcut-Signal für Ort oder Stichprobenzeit liefern. Ein guter Score mit ihnen
belegt deshalb keine robuste spektrale Klassifikation.

## Bereits untersuchte Merkmalsmengen

| Merkmalsmenge | Beobachtung im EDA-Random-Forest | Konsequenz |
| --- | --- | --- |
| Rohspektren | CV Balanced Accuracy kombiniert: 0,570. | Bleibt die unverzichtbare Referenz. |
| Rohspektren + One-Hot-`AEZ` + numerischer `Month` | CV Balanced Accuracy kombiniert: 0,750. | Bisher stärkster beobachteter Zugewinn, aber mit Risiko einer Orts- oder Kalenderabhängigkeit; Monatskodierung wird separat verglichen. |
| Rohspektren + NDVI, NDRE, PRI, NDWI | CV Balanced Accuracy kombiniert: 0,590. | Für diesen Random Forest kein klarer Zusatznutzen gegenüber Rohspektren. Nicht blind übernehmen; für andere Modelle oder eine kompakte Merkmalsmenge separat prüfen. |

Die Werte sind in der [Random-Forest-Baseline](../modelle/ansaetze/baseline.md) dokumentiert. Sie gelten für den dort getesteten Random Forest und dessen
gemeinsamen Split, nicht als allgemeine Rangfolge aller denkbaren Merkmale.

## Kandidaten für weitere, getrennte Vergleiche

| Kandidat | Daten- und Domänenbezug | Konkrete, falsifizierbare Prüfung | Wichtige Grenze |
| --- | --- | --- | --- |
| EVI | EVI kombiniert NIR, Rot und Blau; die passenden verfügbaren Bänder liegen nahe bei `X854`, `X671` und `X468`. | Rohspektren plus die vorab festgelegten Indizes einschließlich EVI gegen Rohspektren vergleichen. | Die Reflektanzwerte müssen vor der EVI-Berechnung von Prozent auf Anteile umgerechnet werden. |
| Red-Edge-Steigung | Zwischen ungefähr 680 und 750 nm steigt die Vegetationsreflektanz häufig stark an. Die EDA sieht Stage-Unterschiede am Übergang zu NIR. | Wenige vorab definierte Differenzen oder Steigungen nur zwischen kontinuierlich verfügbaren Bändern in diesem Bereich als zusätzliche Spalten prüfen. | Der Verlauf kann auch von Bodenanteil, Aufnahmebedingungen und Kultur beeinflusst sein. |
| Vorab definierte Bandverhältnisse | Die Projektbeschreibung nennt Band-Ratios als Feature Engineering. Verhältnisse können spektrale Kontraste verdichten. | Eine kleine, fachlich begründete Liste, etwa NIR/Rot oder NIR/SWIR, gegen Rohbänder vergleichen. | Nicht alle möglichen Bandpaare durchsuchen: Das würde zufällige Treffer und Overfitting begünstigen. |
| Regionsmerkmale für VIS, Red Edge, NIR und SWIR | Diese Bereiche stehen für unterschiedliche Teile des Vegetationsspektrums. Mittelwert, Spannweite oder lokale Steigung können sie kompakt zusammenfassen. | Eine wenige Kennzahlen umfassende, vorab festgelegte Merkmalsmenge gegen Rohspektren und gegen Indizes vergleichen. | Regionen müssen nur verfügbare, physikalisch zusammenhängende Bänder enthalten; große Lücken werden nicht überbrückt. |
| PCA | Benachbarte Bänder sind stark redundant; zwei Komponenten erklären in der EDA mehr als 90 % der Spektralvarianz. | Rohbänder gegen vorab festgelegte PCA-Komponentenzahlen vergleichen, besonders für skalierungsempfindliche Modelle. | PCA ist keine fachlich interpretierbare Bandwahl und wird pro Trainingsfold gelernt. |
| Überwachte Bandauswahl | Die EDA zeigt univariate Klassenunterschiede an einzelnen Wellenlängen. | Eine kleine Anzahl von Bändern ausschließlich innerhalb jedes Trainingsfolds auswählen und gegen Rohbänder vergleichen. | Ein hoher EDA-F-Wert ist kein Leistungsnachweis; Auswahl auf allen Daten wäre Leakage. |

Die fachliche Motivation für sichtbares Licht, Red Edge, NIR, SWIR und die Vegetationsindizes
ist in [Spektrale Signatur von Vegetation](../domaene/vegetation.md) mit Quellen erläutert. Die
Projektbeschreibung nennt außerdem Vegetationsindizes, Bandverhältnisse, Gradienten, PCA und
Feature Selection ausdrücklich als mögliche Varianten.

## Nicht als Modellmerkmal verwechseln

- Die Kennung `id` ist nur eine Identifikation und darf nie in die Eingaben gelangen.
- `Crop`, `Stage` und das Paar `Crop|Stage` sind Zielwerte, keine Features. Gültige
  Crop-Stage-Paare werden durch die Modellierungsstrategie oder Nachverarbeitung behandelt.
- Eine Glättung ist eine Vorverarbeitung, kein zusätzliches Merkmal. Sie bleibt eine eigene,
  vorab festgelegte Pipeline-Variante und wird nicht standardmäßig angewendet.
- Eine Confusion Matrix, Feature Importance oder ein Validierungsscore dient der Diagnose. Sie
  darf nicht genutzt werden, um anschließend auf demselben Holdout weitere Merkmale auszuwählen.

## Empfohlene Reihenfolge

1. Rohspektren ohne Metadaten als Referenz beibehalten.
2. Rohspektren mit `AEZ` und `Month` berichten, aber die Kontextabhängigkeit transparent
   machen.
3. Die standardmäßige zyklische Monatskodierung gegen `use_cyclic_month=False` vergleichen.
4. Danach genau eine kleine spektrale Hypothese wählen: EVI **oder** Red-Edge-Steigungen.
5. Erst wenn diese Varianten verglichen sind, PCA oder trainingsfold-basierte Bandauswahl für
   weitere Modelle prüfen.

Jede Variante folgt demselben Ablauf: `tune()` sucht Parameter und Merkmalskonfigurationen
innerhalb der Cross-Validation auf dem Trainingsanteil; `run()` bewertet ausschließlich den
ausgewählten Kandidaten einmal auf dem Holdout. Berichtet werden Balanced Accuracy und
Macro-F1 getrennt für `Crop` und `Stage`, zusätzlich Samples-F1 und ungültige Crop-Stage-Paare.