# Daten

Wie der `data/`-Ordner funktioniert, steht in `data/README.md`. Laden immer über:

```python
from awp2.data import load_train, load_test, band_columns, wavelengths

train = load_train()  # validiert per pandera-Schema, Index = id
```

## Quelle

!!! info "Herkunft der Challenge-Dateien"
  Die Labels, sieben AEZ, EO-1-Hyperion-Daten und die Gesamtzahl von knapp 7.000
  Spektren stimmen mit **GHISACONUS V001** (*Global Hyperspectral Imaging
  Spectral-library of Agricultural crops for Conterminous United States*) überein.
  Die Challenge-Dateien sind daher mit hoher Sicherheit ein abgeleiteter Export dieses
  Datensatzes.[^1][^2]

- 220 Bänder im Bereich 0,4–2,5 µm (VNIR bis SWIR), ca. 10 nm spektrale Auflösung
- 30 m räumliche Auflösung, 16-Tage-Wiederholzyklus
- 99 Hyperion-Bilder aus den Jahren 2008–2015, sieben AEZ der zusammenhängenden USA
- Die fünf Kulturarten und sechs Stadien stimmen mit GHISACONUS überein.

!!! info "Hinweis aus den Kickoff-Folien"
  Die Karte auf Folie 7 ordnet das Untersuchungsgebiet Nordamerika zu. Das passt zum
  offiziellen GHISACONUS-Produkt für die zusammenhängenden USA. Die Folie erklärt, dass
  AEZ Gebiete mit ähnlichem Klima, Böden und Vegetationsperiode gruppieren.[^3]

## Datengrundlage
Für die Aufgabe liegen hyperspektrale Beobachtungen landwirtschaftlicher Flächen vor, die dem Sensor EO-1 Hyperion zugeordnet sind. Der Trainingsdatensatz enthält 5.591 Zeilen, der Testdatensatz 1.397 Zeilen. Jede Zeile enthält eine Beobachtung mit einer spektralen Reflexionskurve sowie Kontextinformationen zur agroökologischen Zone (AEZ) und zum Aufnahmemonat (Month).

Das Spektrum besteht aus 198 geordneten Reflexionswerten (X427 bis X2395). Die Zahl im Spaltennamen bezeichnet die Wellenlänge in Nanometern. Damit beschreibt eine Zeile, wie stark die beobachtete Fläche Licht über den Wellenlängenbereich von ungefähr 427 bis 2.395 nm reflektiert.

Im Trainingsdatensatz sind zusätzlich die Zielvariablen `Crop` und `Stage` enthalten. Crop umfasst die fünf Kulturarten Mais, Soja, Winterweizen, Baumwolle und Reis. Stage beschreibt sechs allgemeine Entwicklungsstadien. Der Testdatensatz enthält dieselben Eingabemerkmale, aber keine dieser Zielvariablen.

## Herkunft & AEZ

Das veröffentlichte GHISACONUS enthält rund 7.000 Spektren einzelner 30-m-Pixel inklusive
Koordinaten, Bildinformationen, AEZ, Kulturart und Entwicklungsstadium. Die Challenge-Dateien
enthalten zusammen 6.988 Zeilen, aber keine Koordinaten oder Bildinformationen. Damit ist die
Herkunft gut belegt, während die genaue Auswahl der Zeilen und die Umformung auf 198
Bandspalten ohne die Erstellungsdokumentation der Challenge offen bleiben.

Die sechs Stage-Labels entsprechen den GHISACONUS-Gruppen von sehr früher Vegetation bis zur
Ernte. Die sieben beobachteten AEZ-Codes in den Challenge-Dateien passen zur Anzahl der Zonen
im Originalprodukt. Die Bedeutung der einzelnen Codes und die genaue Vorverarbeitung vor dem
Challenge-Export sind weiterhin zu klären.

[^1]: Thenkabail & Aneece (2019), siehe [Quellen](../domaene/quellen.md).
[^2]: Aneece & Thenkabail (2018), siehe [Quellen](../domaene/quellen.md).
[^3]: Domänenprojekt 2 (2026), siehe [Quellen](../domaene/quellen.md).

## Dateien in `data/raw/`

| Datei | Zeilen | Inhalt |
| --- | --- | --- |
| `train.csv` | 5591 | Features + Labels `Crop`, `Stage` |
| `test.csv` | 1397 | Nur Features, keine Labels |

!!! question "Offen"
    Ist `test.csv` bereits der finale Validierungsdatensatz für die Bewertung, oder kommt dieser
    später separat? (Laut Aufgabenstellung erhalten wir ihn „gegen Ende des Projekts".)

## Spalten

| Spalte | Typ | Beschreibung |
| --- | --- | --- |
| *(erste, unbenannt)* | int | Nur Zeilenindex – wird beim Laden verworfen |
| `id` | str | Eindeutige ID (`train_0`, `test_0`, …) |
| `AEZ` | int | Agroökologische Zone (2, 5–10) |
| `Month` | int | Aufnahmemonat (5–10) |
| `Crop` | str | `corn`, `soybean`, `winter_wheat`, `cotton`, `rice` |
| `Stage` | str | `Emerge_VEarly`, `Early_Mid`, `Critical`, `Late`, `Mature_Senesc`, `Harvest` |
| `X427` … `X2395` | float | Reflexion (%) je Band, Zahl = Wellenlänge in nm (198 Bänder) |

## Erste Befunde

- **67 von 198 Bändern sind komplett leer** (in train und test identisch), v. a. die
  Wasserabsorptionsbereiche um ~1400 nm und ~1900 nm sowie die Ränder → 131 nutzbare Bänder.
- Vereinzelte fehlende Werte in 6 Bändern (max. 24 Zeilen in `X2063`).
- 2 Zeilen mit identischem Spektrum.
- Reflexionswerte zwischen 0,25 und 93,2, keine negativen Werte.
- **Starkes Klassenungleichgewicht**: corn 2097, soybean 1669, winter_wheat 1073, cotton 659,
  rice 93. Auch die Stages sind unbalanciert (Harvest nur 180).
- **Nicht jede Crop-Stage-Kombination existiert** (z. B. rice nur `Early_Mid`/`Late`,
  winter_wheat ohne `Early_Mid`/`Harvest`, cotton ohne `Late`) → spricht für hierarchische
  bzw. kombinierte Ansätze.
