# Glossar

Alphabetisch, je 1–2 Sätze, mit Link zum Abschnitt, in dem der Begriff ausführlich erklärt ist.
Jede:r ergänzt die Begriffe aus der eigenen Seite.

| Begriff | Erklärung | Mehr |
| --- | --- | --- |
| AEZ (agroökologische Zone) | Gruppe von Gebieten mit ähnlichem Klima, Böden und ähnlicher Vegetationsperiode; sie beschreibt keine genaue Position. | [Daten](../daten/index.md#herkunft-aez) |
| Atmosphärische Korrektur | Rechenschritt, der Einflüsse der Atmosphäre aus einem Satellitensignal möglichst entfernt. | [Fernerkundung](fernerkundung.md#atmosphare-toa-und-oberflachenreflektanz) |
| Atmosphärisches Wasserabsorptionsband | Wellenlängenbereich, in dem Wasserdampf in der Atmosphäre einen großen Teil des Lichts absorbiert; um 1.400 und 1.900 nm ist die Oberflächenreflektanz besonders unsicher. | [Vegetation](vegetation.md#wasserabsorptionsbander) |
| Balanced Accuracy | | [ML-Aufgabe](ml-aufgabe.md#bewertungsmetriken) |
| BBCH-Skala | Standardisierte Dezimalskala zur Beschreibung beobachtbarer Entwicklungsstadien von Pflanzen. | [Kulturpflanzen](kulturpflanzen.md#entwicklungsstadien-als-gemeinsame-sprache) |
| Blattflächenindex (LAI) | Verhältnis der gesamten Blattfläche zur Bodenfläche. Er beeinflusst, wie stark der Boden unter einem Pflanzenbestand im Pixel noch sichtbar ist. | [Vegetation](vegetation.md#red-edge-und-nahes-infrarot) |
| CAI (Cellulose Absorption Index) | Index für die Absorption trockener Pflanzenbestandteile um 2.100 nm. Der Standard-CAI ist hier wegen des leeren Bands `X2002` nicht direkt berechenbar. | [Vegetation](vegetation.md#indizes-fur-photosynthese-wasser-und-trockene-bestandteile) |
| Chlorophyll | Grüner Blattfarbstoff, der vor allem blaues und rotes Licht absorbiert und damit die Reflexion grüner Pflanzen prägt. | [Vegetation](vegetation.md#sichtbares-licht-die-pflanze-erscheint-grun) |
| Data Leakage | | [ML-Aufgabe](ml-aufgabe.md#data-leakage) |
| Entwicklungsstadium (Stage) | Abschnitt im saisonalen Entwicklungsverlauf einer Kulturpflanze. | [Domäne](index.md#bedeutung-des-entwicklungsstadiums) |
| EVI (Enhanced Vegetation Index) | Vegetationsindex aus NIR-, Rot- und Blau-Reflektanz, der bestimmte Boden- und Atmosphäreneinflüsse robuster behandeln soll. | [Vegetation](vegetation.md#indizes-fur-grune-vegetation) |
| Fernerkundung | Gewinnung von Informationen aus der Entfernung, etwa mit Sensoren auf Satelliten. | [Fernerkundung](fernerkundung.md#grundlagen-der-optischen-fernerkundung) |
| Hyperspektral | Viele schmale, aufeinanderfolgende Spektralbänder erfassen ein detailliertes Spektrum. | [Fernerkundung](fernerkundung.md#multispektral-vs-hyperspektral) |
| Kulturart (Crop) | Landwirtschaftlich angebaute Pflanzenart, zum Beispiel Mais oder Soja. | [Domäne](index.md#projektziel) |
| Mischpixel | Pixel, das mehrere Oberflächen wie Kultur, Boden oder Feldrand zugleich enthält. | [Fernerkundung](fernerkundung.md#raumliche-auflosung-mischpixel) |
| Multispektral | Wenige, vergleichsweise breite Spektralbänder erfassen ein grobes Spektrum. | [Fernerkundung](fernerkundung.md#multispektral-vs-hyperspektral) |
| NDRE (Normalized Difference Red Edge) | Verhältnisindex aus NIR und Red Edge; er nutzt den steilen Übergang zwischen rotem Licht und NIR. | [Vegetation](vegetation.md#indizes-fur-grune-vegetation) |
| NDVI (Normalized Difference Vegetation Index) | Verhältnisindex aus NIR und rotem Licht, der häufig mit grüner Vegetation verbunden ist. Bei dichtem Bestand kann er sättigen. | [Vegetation](vegetation.md#indizes-fur-grune-vegetation) |
| NDWI / NDII | Namen für NIR-SWIR-Verhältnisindizes, die je nach genauer Formel Hinweise auf Wasser in einer Vegetationsdecke geben. | [Vegetation](vegetation.md#indizes-fur-photosynthese-wasser-und-trockene-bestandteile) |
| Oberflächenreflektanz | Geschätzte Reflektanz der Erdoberfläche, bei der Atmosphäreinflüsse möglichst korrigiert sind. | [Fernerkundung](fernerkundung.md#atmosphare-toa-und-oberflachenreflektanz) |
| Passiver optischer Sensor | Sensor, der kein eigenes Licht aussendet, sondern reflektiertes Sonnenlicht misst. | [Fernerkundung](fernerkundung.md#was-misst-ein-passiver-optischer-sensor) |
| Phänologie | Zeitliche Abfolge wiederkehrender Entwicklungsphasen einer Pflanze, etwa vom Auflaufen bis zur Ernte. | [Kulturpflanzen](kulturpflanzen.md#was-ist-phanologie) |
| PRI (Photochemical Reflectance Index) | Schmalbandiger Index aus sichtbaren Bändern um 531 und 570 nm, der für Änderungen der photosynthetischen Lichtnutzung entwickelt wurde. | [Vegetation](vegetation.md#indizes-fur-photosynthese-wasser-und-trockene-bestandteile) |
| Räumliche Auflösung | Größe der Fläche am Boden, die ein Bildpixel repräsentiert. | [Fernerkundung](fernerkundung.md#raumliche-auflosung-mischpixel) |
| Red Edge | Steiler Anstieg der Vegetationsreflexion zwischen ungefähr 680 und 750 nm, vom roten Licht zum NIR. | [Vegetation](vegetation.md#red-edge-und-nahes-infrarot) |
| Reflektanz | Anteil des einfallenden Lichts, den eine Oberfläche bei einer Wellenlänge zurückreflektiert. | [Fernerkundung](fernerkundung.md#reflektanz-und-die-bandspalten) |
| Seneszenz | Alterungsphase von Pflanzenteilen, in der unter anderem die photosynthetische Aktivität abnimmt. | [Vegetation](vegetation.md#veranderung-uber-die-saison) |
| Spektralband | Kleiner Wellenlängenbereich, für den ein Sensor einen Reflektanzwert speichert. | [Fernerkundung](fernerkundung.md#reflektanz-und-die-bandspalten) |
| Spektrale Signatur | Verlauf der Reflektanz einer Oberfläche über die Wellenlängen; sie kann Hinweise auf Material- und Vegetationseigenschaften geben. | [Vegetation](vegetation.md#die-typische-reflexionskurve) |
| Spektrum | Folge der Reflektanzwerte einer Beobachtung über viele Wellenlängen. | [Fernerkundung](fernerkundung.md#reflektanz-und-die-bandspalten) |
| Schmalband-Hyperspektralindex | Vegetationsindex, der eng benachbarte Bänder eines hyperspektralen Sensors gezielt kombiniert; keine einzelne festgelegte Formel. | [Vegetation](vegetation.md#schmalband-hyperspektralindizes) |
| SWIR / NIR / VIS | Kurzwelliges Infrarot, nahes Infrarot und sichtbares Licht: drei Wellenlängenbereiche. | [Fernerkundung](fernerkundung.md#wellenlangenbereiche) |
| TOA-Reflektanz | Reflektanzsignal, das nach dem Weg durch die Atmosphäre am Satelliten ankommt. | [Fernerkundung](fernerkundung.md#atmosphare-toa-und-oberflachenreflektanz) |
| Vegetationsindex | Kennzahl, die mehrere Reflektanzwerte kombiniert, um eine bestimmte Vegetationseigenschaft kompakt zu beschreiben. | [Vegetation](vegetation.md#vegetationsindizes) |
| Wellenlänge | Eigenschaft des Lichts, die seinen Bereich im Spektrum bestimmt; sie wird hier in Nanometern angegeben. | [Fernerkundung](fernerkundung.md#reflektanz-und-die-bandspalten) |
