# Explorative Datenanalyse (EDA)

Ergebnisse der EDA auf `train.csv` (und `test.csv` zum Vergleich). Jedes EDA-Issue füllt seinen
Abschnitt mit kurzen Stichpunkten und den wichtigsten Plots. Die Analysen selbst liegen in den
Notebooks unter `notebooks/`.

!!! info "So arbeiten wir hier"
    - Plots mit `awp2.plots` erzeugen, z. B. `plot_spectra(df, by="Crop")`.
    - Plots für diese Seite mit `save_doc_figure(fig, "<name>")` speichern (landen in
      `docs/daten/img/`) und einbinden: `![Beschreibung](img/<name>.png)`.
    - Nur beschreiben und markieren – Daten werden erst in der Data Preparation verändert.

## EDA-Fahrplan: Fragen zuerst klären

Eine Zeile ist ein Spektrum einer landwirtschaftlichen Fläche: 198 Reflektanzwerte entlang der
Wellenlänge sowie `AEZ` und `Month`; nur `train.csv` enthält `Crop` und `Stage`. Die EDA prüft,
welche beobachtbaren Muster und Risiken diese Felder enthalten. Sie wählt noch keine Merkmale
und bewertet keine Modelle. Insbesondere beweist ein sichtbarer Unterschied zwischen Gruppen
noch nicht, dass er auf unbekannten Daten vorhersagekräftig ist.

Wir laden die Daten ausschließlich über `awp2.data.load_train()` und `load_test()`. Bandspalten
werden über `band_columns()` und ihre Wellenlängen über `wavelengths()` bestimmt. Die folgenden
Schritte sind in dieser Reihenfolge auszuführen; das Ergebnis jedes Schritts wird in den
passenden Abschnitt weiter unten eingetragen.

| Priorität | Konkrete Frage und Datenbezug | Analyse und festzuhaltendes Ergebnis | Konsequenz oder Grenze |
| --- | --- | --- | --- |
| 1 | Haben Train und Test dieselbe technisch nutzbare Eingabestruktur? (`AEZ`, `Month`, alle Bandspalten) | Form, Datentypen, IDs, Wertebereiche und Kategorien vergleichen; je Band Anzahl fehlender Werte, konstante Werte sowie Min/Median/Max dokumentieren. Fehlende Bänder als Wellenlängenbereiche darstellen. | Vollständig leere Bänder sind keine Modellinformation. Einzelne Lücken und auffällige Werte sind zunächst Befunde, keine Cleaning-Entscheidung. |
| 2 | Treten Lücken oder auffällige Spektren systematisch in bestimmten Gruppen auf? (fehlende Werte, `AEZ`, `Month`, `Crop`, `Stage`) | Für teilweise fehlende Bänder die betroffenen Zeilen nach Metadaten und Labels auszählen. Spektren mit robusten Kennzahlen und Einzelplots prüfen, besonders an Bandgrenzen sowie um `X912`–`X925`. | Klärt, ob fehlende Werte zufällig wirken oder mit einer beobachteten Gruppe zusammenhängen. Die Ursache (Sensor, Atmosphäre oder Export) bleibt ohne Erstellungsdokumentation offen. |
| 3 | Gibt es identische oder fast identische Beobachtungen, die einen zufälligen Split verfälschen könnten? (`AEZ`, `Month`, verfügbare Bänder, Labels) | Exakte Duplikate und Gruppen sehr ähnlicher Spektren mit einer vorher festgelegten Distanz messen; für jede Gruppe prüfen, ob Labels, Metadaten und die Zugehörigkeit zu Train/Test übereinstimmen. | Liefert ein Leakage-Risiko und mögliche Gruppen für eine spätere Split-Regel. Ohne Koordinaten oder Feld-IDs beweist Ähnlichkeit keine gemeinsame Fläche. |
| 4 | Welche Labels und Crop-Stage-Paare sind selten oder fehlen? (`Crop`, `Stage`) | Häufigkeiten, Anteile und die Kreuztabelle `Crop` x `Stage` erstellen; kleinste Paarhäufigkeit gegen 70/30-Split und fünf CV-Folds prüfen. | Begründet Stratifizierung nach `Crop|Stage`, klassenfaire Kennzahlen und eine vorsichtige Interpretation seltener Paare. Fehlende Paare sind ein Stichprobenbefund, keine Aussage über landwirtschaftliche Unmöglichkeit. |
| 5 | Können `AEZ` oder `Month` die Labels stark vorstrukturieren? (`AEZ` x `Crop`, `Crop` x `Month` x `Stage`) | Kreuztabellen und normierte Anteile je Kultur zeigen; innerhalb jeder Kultur die Monatsverteilung je Stadium vergleichen. Zusätzlich die Verteilung von `AEZ` und `Month` zwischen Train und Test gegenüberstellen. | Macht legitime Kontextinformation und mögliches Shortcut-Lernen sichtbar. Es klärt nicht, ob `Month` ein tatsächlicher Aufnahmemonat ist oder ob Metadaten in der finalen Vorhersage verfügbar sein werden. |
| 6 | Unterscheiden sich die Spektren nach Kultur und innerhalb einer Kultur nach Stadium? (verfügbare Bänder, `Crop`, `Stage`) | Median und Interquartilsbereich je Kultur über der Wellenlänge plotten. Danach denselben Plot je `Stage` innerhalb jeder Kultur erstellen und die Gruppengröße sichtbar ausweisen. VIS, Red Edge, NIR und SWIR getrennt vergleichen. | Prüft die fachlichen Hypothesen zu Chlorophyll, Blattstruktur und Wasserstatus an den Daten. Überlappende Bereiche, ungleiche Gruppengrößen und Mischpixel bleiben Alternativerklärungen. |
| 7 | Gibt es wenige Spektralbereiche mit auffälligen Klassenunterschieden oder stark redundante Bänder? (geordnete Bandspalten, Labels) | Nach dem Qualitätscheck Korrelation benachbarter Bänder, Klassenunterschiede pro Band und eine PCA mit erklärter Varianz und 2D-Projektion erstellen. Leere Bänder bleiben ausgeschlossen. | Erzeugt Hypothesen für Bandauswahl, Glättung oder Dimensionsreduktion. Jede spätere Auswahl oder PCA muss ausschließlich innerhalb der Trainings-Pipeline gelernt werden. |
| 8 | Verhalten sich unbeschriftete Testdaten in den Eingaben anders als Train? (`AEZ`, `Month`, nutzbare Bänder) | Für Metadaten und ausgewählte robuste Spektralkennwerte Verteilungen, Quantile und auffällige Randbereiche von Train und Test vergleichen. | Zeigt mögliche Verteilungsverschiebungen. Ohne Testlabels lässt sich weder die spätere Modellgüte noch eine Labelverteilung im Test bestimmen. |
| 9 | Verdichten vorab definierte Vegetationsindizes die sichtbaren Unterschiede? (nächstgelegene verfügbare Bänder für NDVI, EVI, NDRE, PRI und eine eindeutig benannte NDWI-Variante) | Erst nach Schritt 1–2 die Berechenbarkeit und Verteilungen je Kultur und Stadium prüfen; Formel, Bänder und Umgang mit fehlenden Werten protokollieren. | Dies ist nur eine Feature-Engineering-Hypothese aus der [Vegetationslehre](../domaene/vegetation.md#vegetationsindizes), kein Leistungsnachweis. Ein späterer Vergleich erfolgt innerhalb der Validierungspipeline. |

### Bezug zu den offenen Fragen

| Offene Frage | Was die EDA dazu beitragen kann | Was Daten allein nicht klären können |
| --- | --- | --- |
| [#1: Rolle von `test.csv`](../projekt/offene-fragen.md) | Schritt 1 und 8 prüfen, ob die Eingaben kompatibel wirken. | Ob dies der finale Bewertungsdatensatz ist, kann nur die Aufgabenbetreuung beantworten. |
| [#3: „70/30 Clusterplot-Split“](../projekt/offene-fragen.md) | Schritt 3 zeigt Duplikat- und Ähnlichkeitsgruppen als mögliche Leakage-Gefahr. | Ohne Definition von „Clusterplot“, Koordinaten oder Feld-IDs lässt sich keine räumliche Split-Regel ableiten. |
| [#5: Quelle, Regionen, Jahre und AEZ](../projekt/offene-fragen.md) | Schritt 5 beschreibt die tatsächlich vorkommenden AEZ- und Monatsgruppen. | Codes, Regionen, Jahre und die genaue Challenge-Auswahl benötigen die Originalquelle. |
| [#7: Kriterien der Stage-Labels](../projekt/offene-fragen.md) | Schritt 5–6 prüfen Monats- und Spektralmuster innerhalb jeder Kultur. | Diese Muster definieren keine BBCH-, V/R- oder Zadoks-Zuordnung. |
| [#8: Fehlende Crop-Stage-Kombinationen](../projekt/offene-fragen.md) | Schritt 4 dokumentiert exakt, welche Kombinationen in der Stichprobe fehlen. | Die Ursache kann Selektion, Anbausaison oder Labelbildung sein und ist nicht aus der Kreuztabelle ableitbar. |
| [#9: Bedeutung von `Month`](../projekt/offene-fragen.md) | Schritt 5 quantifiziert den Zusammenhang von Monat und Labels. | Ob `Month` der tatsächliche Aufnahmemonat oder eine nachträgliche Kategorie ist, muss die Quelle bestätigen. |
| [#10: Wiederholte Feldbeobachtungen](../projekt/offene-fragen.md) | Schritt 3 kann exakte und sehr ähnliche Spektren als Verdachtsfälle markieren. | Feldwiederholungen und räumliche Nähe sind ohne räumliche oder Bild-Metadaten nicht nachweisbar. |

Die Fragen #2, #4 und #6 betreffen Termine, die Aufgabeninterpretation beziehungsweise das
Abgabeformat; sie sind nicht mit einer Analyse von `train.csv` oder `test.csv` beantwortbar.

![Mittleres Spektrum aller Trainingsdaten](img/spektrum_ueberblick.png)

## Zusammenfassung

!!! todo "#20 – Synthese & Cleaning-Empfehlung"
    - Die wichtigsten Befunde in 5–10 Stichpunkten, fachlich eingeordnet (→ [Domäne](../domaene/index.md))
    - Textbaustein „Data Understanding" für den M1-Kurzreport

## Datenqualität

!!! todo "#15"
    - Fehlende Werte je Band (Muster, train vs. test), Zeilen mit einzelnen NaNs
    - Exakte und Beinahe-Duplikate, Wertebereiche, Verteilungen je Band
    - Vergleich train vs. test (AEZ, Month, Spektren)

## Labels & Metadaten

!!! todo "#16"
    - Verteilung Crop und Stage, Kreuztabelle Crop × Stage
    - AEZ × Crop, Month × Stage je Kultur
    - Folgerungen für Stratifizierung und Klassenungleichgewicht

## Spektrale Signaturen je Klasse

!!! todo "#17"
    - Mittlere Spektren je Kultur, je Stadium innerhalb jeder Kultur, je AEZ
    - Trennkraft je Band (z. B. ANOVA-F), Bereiche mit den größten Unterschieden

## Korrelation & Dimensionalität

!!! todo "#18"
    - Korrelationsmatrix der Bänder
    - PCA: erklärte Varianz, 2D-Projektion nach Crop/Stage
    - Hinweise für die Bandreduktion (M3)

## Ausreißer & Auffälligkeiten

!!! todo "#19"
    - Auffällige Spektren je Klasse, Spitzen/Sprünge, verrauschte Bänder
    - Überlappung X912–X925
    - Gruppen fast identischer Spektren (gleiches Feld?) als Leakage-Risiko

## Cleaning-Empfehlungen

!!! todo "#20"
    | Thema | Befund | Empfehlung | Begründung |
    | --- | --- | --- | --- |
    | Leere Bänder | | | |
    | Einzelne NaNs | | | |
    | Duplikate | | | |
    | Ausreißer | | | |
    | Glättung / Rauschen | | | |
    | Split-Strategie | | | |
