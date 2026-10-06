# Glossar

Alphabetisch, je 1–2 Sätze, mit Link zum Abschnitt, in dem der Begriff ausführlich erklärt ist.
Jede:r ergänzt die Begriffe aus der eigenen Seite.

| Begriff | Erklärung | Mehr |
| --- | --- | --- |
| AEZ (agroökologische Zone) | Gruppe von Gebieten mit ähnlichem Klima, Böden und ähnlicher Vegetationsperiode; sie beschreibt keine genaue Position. | [Daten](../daten/index.md#herkunft-aez) |
| Atmosphärische Korrektur | Rechenschritt, der Einflüsse der Atmosphäre aus einem Satellitensignal möglichst entfernt. | [Fernerkundung](fernerkundung.md#atmosphare-toa-und-oberflachenreflektanz) |
| Atmosphärisches Wasserabsorptionsband | Wellenlängenbereich, in dem Wasserdampf in der Atmosphäre einen großen Teil des Lichts absorbiert; um 1.400 und 1.900 nm ist die Oberflächenreflektanz besonders unsicher. | [Vegetation](vegetation.md#wasserabsorptionsbander) |
| Balanced Accuracy | Mittelwert der Recalls aller Klassen. Dadurch zählt eine seltene Klasse wie Reis genauso stark wie eine häufige Klasse wie Mais; Zufallsniveau ist 1/Anzahl Klassen. | [ML-Aufgabe](ml-aufgabe.md#balanced-accuracy-hauptmetrik) |
| BBCH-Skala | Standardisierte Dezimalskala zur Beschreibung beobachtbarer Entwicklungsstadien von Pflanzen. | [Kulturpflanzen](kulturpflanzen.md#entwicklungsstadien-als-gemeinsame-sprache) |
| Blattflächenindex (LAI) | Verhältnis der gesamten Blattfläche zur Bodenfläche. Er beeinflusst, wie stark der Boden unter einem Pflanzenbestand im Pixel noch sichtbar ist. | [Vegetation](vegetation.md#red-edge-und-nahes-infrarot) |
| CAI (Cellulose Absorption Index) | Index für die Absorption trockener Pflanzenbestandteile um 2.100 nm. Der Standard-CAI ist hier wegen des leeren Bands `X2002` nicht direkt berechenbar. | [Vegetation](vegetation.md#indizes-fur-photosynthese-wasser-und-trockene-bestandteile) |
| Chlorophyll | Grüner Blattfarbstoff, der vor allem blaues und rotes Licht absorbiert und damit die Reflexion grüner Pflanzen prägt. | [Vegetation](vegetation.md#sichtbares-licht-die-pflanze-erscheint-grun) |
| Confusion Matrix | Tabelle, die wahre und vorhergesagte Klassen gegenüberstellt. Die Diagonale enthält korrekte Vorhersagen, die übrigen Felder zeigen Verwechslungen. | [ML-Aufgabe](ml-aufgabe.md#bewertungsmetriken) |
| Cross-Validation (CV) | Verfahren zum Vergleichen von Varianten: Der Trainingsanteil wird mehrfach in Lern- und Prüf-Folds geteilt, die Scores werden gemittelt. | [ML-Aufgabe](ml-aufgabe.md#validierung) |
| Data Leakage | Informationen aus Validierung oder Test gelangen beim Lernen in das Modell und machen den Score künstlich zu gut. | [ML-Aufgabe](ml-aufgabe.md#data-leakage) |
| Entwicklungsstadium (Stage) | Abschnitt im saisonalen Entwicklungsverlauf einer Kulturpflanze. | [Domäne](index.md#bedeutung-des-entwicklungsstadiums) |
| EVI (Enhanced Vegetation Index) | Vegetationsindex aus NIR-, Rot- und Blau-Reflektanz, der bestimmte Boden- und Atmosphäreneinflüsse robuster behandeln soll. | [Vegetation](vegetation.md#indizes-fur-grune-vegetation) |
| F1-Score | Harmonisches Mittel von Precision und Recall. Er ist nur hoch, wenn ein Modell eine Klasse sowohl zuverlässig vorhersagt als auch möglichst vollständig findet. | [ML-Aufgabe](ml-aufgabe.md#bewertungsmetriken) |
| Fernerkundung | Gewinnung von Informationen aus der Entfernung, etwa mit Sensoren auf Satelliten. | [Fernerkundung](fernerkundung.md#grundlagen-der-optischen-fernerkundung) |
| Hierarchische Klassifikation | Mehrstufige Vorhersage, hier zuerst `Crop` und anschließend `Stage` innerhalb der vorhergesagten Kultur. | [ML-Aufgabe](ml-aufgabe.md#mogliche-formulierungen) |
| Holdout | Beschrifteter Datenteil, der vor der Modellarbeit beiseitegelegt und nur einmal zur abschließenden Bewertung genutzt wird. Bei uns dasselbe wie die 30-%-Validierung. | [ML-Aufgabe](ml-aufgabe.md#validierung) |
| Hyperspektral | Viele schmale, aufeinanderfolgende Spektralbänder erfassen ein detailliertes Spektrum. | [Fernerkundung](fernerkundung.md#multispektral-vs-hyperspektral) |
| Klassenungleichgewicht | Manche Klassen haben sehr viel weniger Beobachtungen als andere. Bei uns betrifft das insbesondere Reis und `Harvest`. | [ML-Aufgabe](ml-aufgabe.md#klassenungleichgewicht) |
| Klassengewichte | Gewichte beim Training, die Fehler bei seltenen Klassen stärker berücksichtigen. | [ML-Aufgabe](ml-aufgabe.md#mogliche-gegenmanahmen) |
| Kulturart (Crop) | Landwirtschaftlich angebaute Pflanzenart, zum Beispiel Mais oder Soja. | [Domäne](index.md#projektziel) |
| Macro-F1 | Ungewichteter Mittelwert der F1-Scores aller Klassen. Bestraft im Gegensatz zur Balanced Accuracy auch, wenn eine Klasse zu oft vorhergesagt wird. | [ML-Aufgabe](ml-aufgabe.md#macro-f1) |
| Macro-/Micro-/Weighted-Mittelung | Arten, Klassenwerte zusammenzufassen: Macro gewichtet Klassen gleich, Micro einzelne Vorhersagen gleich und Weighted Klassen nach ihrer Häufigkeit. | [ML-Aufgabe](ml-aufgabe.md#mittelungen-und-samples-f1) |
| Mischpixel | Pixel, das mehrere Oberflächen wie Kultur, Boden oder Feldrand zugleich enthält. | [Fernerkundung](fernerkundung.md#raumliche-auflosung-mischpixel) |
| Multi-Label-Klassifikation | Jede Beobachtung kann mehrere unabhängige Labels gleichzeitig tragen. Unsere Daten haben stattdessen genau ein `Crop`- und ein `Stage`-Label. | [ML-Aufgabe](ml-aufgabe.md#was-soll-vorhergesagt-werden) |
| Multiklassen-Klassifikation | Eine Beobachtung wird genau einer von mehreren möglichen Klassen zugeordnet, etwa einer von fünf Kulturarten. | [ML-Aufgabe](ml-aufgabe.md#was-soll-vorhergesagt-werden) |
| Multispektral | Wenige, vergleichsweise breite Spektralbänder erfassen ein grobes Spektrum. | [Fernerkundung](fernerkundung.md#multispektral-vs-hyperspektral) |
| Multi-Output-Klassifikation | Ein Modell sagt für dieselbe Beobachtung mehrere Zielspalten vorher, bei uns `Crop` und `Stage`. | [ML-Aufgabe](ml-aufgabe.md#was-soll-vorhergesagt-werden) |
| NDRE (Normalized Difference Red Edge) | Verhältnisindex aus NIR und Red Edge; er nutzt den steilen Übergang zwischen rotem Licht und NIR. | [Vegetation](vegetation.md#indizes-fur-grune-vegetation) |
| NDVI (Normalized Difference Vegetation Index) | Verhältnisindex aus NIR und rotem Licht, der häufig mit grüner Vegetation verbunden ist. Bei dichtem Bestand kann er sättigen. | [Vegetation](vegetation.md#indizes-fur-grune-vegetation) |
| NDWI / NDII | Namen für NIR-SWIR-Verhältnisindizes, die je nach genauer Formel Hinweise auf Wasser in einer Vegetationsdecke geben. | [Vegetation](vegetation.md#indizes-fur-photosynthese-wasser-und-trockene-bestandteile) |
| Oberflächenreflektanz | Geschätzte Reflektanz der Erdoberfläche, bei der Atmosphäreinflüsse möglichst korrigiert sind. | [Fernerkundung](fernerkundung.md#atmosphare-toa-und-oberflachenreflektanz) |
| Passiver optischer Sensor | Sensor, der kein eigenes Licht aussendet, sondern reflektiertes Sonnenlicht misst. | [Fernerkundung](fernerkundung.md#was-misst-ein-passiver-optischer-sensor) |
| Phänologie | Zeitliche Abfolge wiederkehrender Entwicklungsphasen einer Pflanze, etwa vom Auflaufen bis zur Ernte. | [Kulturpflanzen](kulturpflanzen.md#was-ist-phanologie) |
| Pipeline | Verkettung von Vorverarbeitung und Modell, die jeden lernbaren Schritt nur auf dem jeweiligen Trainingsanteil anpasst. | [ML-Aufgabe](ml-aufgabe.md#data-leakage) |
| Precision | Anteil korrekter Vorhersagen unter allen Vorhersagen einer Klasse; „Wenn das Modell Reis sagt, wie oft stimmt das?“ | [ML-Aufgabe](ml-aufgabe.md#bewertungsmetriken) |
| PRI (Photochemical Reflectance Index) | Schmalbandiger Index aus sichtbaren Bändern um 531 und 570 nm, der für Änderungen der photosynthetischen Lichtnutzung entwickelt wurde. | [Vegetation](vegetation.md#indizes-fur-photosynthese-wasser-und-trockene-bestandteile) |
| Räumliche Autokorrelation | Nahe Flächen können ähnliche Spektren haben. Ein zufälliger Split kann dann zu optimistische Validierungsscores liefern. | [ML-Aufgabe](ml-aufgabe.md#raumliche-ahnlichkeit-als-risiko) |
| Räumliche Auflösung | Größe der Fläche am Boden, die ein Bildpixel repräsentiert. | [Fernerkundung](fernerkundung.md#raumliche-auflosung-mischpixel) |
| Recall | Anteil der gefundenen Beobachtungen einer Klasse; „Welchen Anteil der echten Reis-Spektren erkennt das Modell?“ | [ML-Aufgabe](ml-aufgabe.md#bewertungsmetriken) |
| Red Edge | Steiler Anstieg der Vegetationsreflexion zwischen ungefähr 680 und 750 nm, vom roten Licht zum NIR. | [Vegetation](vegetation.md#red-edge-und-nahes-infrarot) |
| Reflektanz | Anteil des einfallenden Lichts, den eine Oberfläche bei einer Wellenlänge zurückreflektiert. | [Fernerkundung](fernerkundung.md#reflektanz-und-die-bandspalten) |
| Samples-F1 | F1-Score pro Beobachtung über beide Zielwerte. Bei uns zählen beide korrekt als 1, nur einer korrekt als 0,5 und beide falsch als 0; nicht klassenfair. | [ML-Aufgabe](ml-aufgabe.md#samples-f1) |
| Seneszenz | Alterungsphase von Pflanzenteilen, in der unter anderem die photosynthetische Aktivität abnimmt. | [Vegetation](vegetation.md#veranderung-uber-die-saison) |
| Spektralband | Kleiner Wellenlängenbereich, für den ein Sensor einen Reflektanzwert speichert. | [Fernerkundung](fernerkundung.md#reflektanz-und-die-bandspalten) |
| Spektrale Signatur | Verlauf der Reflektanz einer Oberfläche über die Wellenlängen; sie kann Hinweise auf Material- und Vegetationseigenschaften geben. | [Vegetation](vegetation.md#die-typische-reflexionskurve) |
| Spektrum | Folge der Reflektanzwerte einer Beobachtung über viele Wellenlängen. | [Fernerkundung](fernerkundung.md#reflektanz-und-die-bandspalten) |
| Schmalband-Hyperspektralindex | Vegetationsindex, der eng benachbarte Bänder eines hyperspektralen Sensors gezielt kombiniert; keine einzelne festgelegte Formel. | [Vegetation](vegetation.md#schmalband-hyperspektralindizes) |
| Stratifizierung | Aufteilung, die die Anteile der Klassen möglichst in jedem Teil erhält. Hier wird nach dem kombinierten Crop-Stage-Label stratifiziert. | [ML-Aufgabe](ml-aufgabe.md#validierung) |
| SWIR / NIR / VIS | Kurzwelliges Infrarot, nahes Infrarot und sichtbares Licht: drei Wellenlängenbereiche. | [Fernerkundung](fernerkundung.md#wellenlangenbereiche) |
| TOA-Reflektanz | Reflektanzsignal, das nach dem Weg durch die Atmosphäre am Satelliten ankommt. | [Fernerkundung](fernerkundung.md#atmosphare-toa-und-oberflachenreflektanz) |
| Validierung | Ein bis zur abschließenden Bewertung unberührter Anteil der beschrifteten Daten, der die Leistung auf unbekannten Beobachtungen schätzt. | [ML-Aufgabe](ml-aufgabe.md#validierung) |
| Vegetationsindex | Kennzahl, die mehrere Reflektanzwerte kombiniert, um eine bestimmte Vegetationseigenschaft kompakt zu beschreiben. | [Vegetation](vegetation.md#vegetationsindizes) |
| Wellenlänge | Eigenschaft des Lichts, die seinen Bereich im Spektrum bestimmt; sie wird hier in Nanometern angegeben. | [Fernerkundung](fernerkundung.md#reflektanz-und-die-bandspalten) |
