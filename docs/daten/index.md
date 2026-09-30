# Daten

Wie der `data/`-Ordner funktioniert, steht in `data/README.md`. Laden immer über:

```python
from awp2.data import load_train, load_test, band_columns, wavelengths

train = load_train()  # validiert per pandera-Schema, Index = id
```

## Quelle

Hyperspektral-Signaturen landwirtschaftlicher Kulturpflanzen, aufgenommen vom
**EO-1 Hyperion**-Sensor (NASA, außer Betrieb seit 2017):

- 220 Bänder im Bereich 0,4–2,5 µm (VNIR bis SWIR), ca. 10 nm spektrale Auflösung
- 30 m räumliche Auflösung, 16-Tage-Wiederholzyklus
- Harmonisierter Datensatz aus weltweiten Quellen, verschiedene agroökologische Zonen (AEZ)

!!! info "Hinweis aus den Kickoff-Folien"
  Die Karte auf Folie 7 ordnet das Untersuchungsgebiet Nordamerika zu. Die Folie erklärt,
  dass AEZ Gebiete mit ähnlichem Klima, Böden und Vegetationsperiode gruppieren.[^1] Ob die
  Angabe „weltweite Quellen" die Herkunft der gesamten Bibliothek oder auch der hier
  verwendeten Beobachtungen beschreibt, bleibt bis zur Klärung der Originalquelle offen.

## Datengrundlage
Für die Aufgabe liegen hyperspektrale Beobachtungen landwirtschaftlicher Flächen vor, die dem Sensor EO-1 Hyperion zugeordnet sind. Der Trainingsdatensatz enthält 5.591 Zeilen, der Testdatensatz 1.397 Zeilen. Jede Zeile enthält eine Beobachtung mit einer spektralen Reflexionskurve sowie Kontextinformationen zur agroökologischen Zone (AEZ) und zum Aufnahmemonat (Month).

Das Spektrum besteht aus 198 geordneten Reflexionswerten (X427 bis X2395). Die Zahl im Spaltennamen bezeichnet die Wellenlänge in Nanometern. Damit beschreibt eine Zeile, wie stark die beobachtete Fläche Licht über den Wellenlängenbereich von ungefähr 427 bis 2.395 nm reflektiert.

Im Trainingsdatensatz sind zusätzlich die Zielvariablen `Crop` und `Stage` enthalten. Crop umfasst die fünf Kulturarten Mais, Soja, Winterweizen, Baumwolle und Reis. Stage beschreibt sechs allgemeine Entwicklungsstadien. Der Testdatensatz enthält dieselben Eingabemerkmale, aber keine dieser Zielvariablen.

## Herkunft & AEZ

!!! todo "Leitfragen"
    - Stammt der Datensatz aus der USGS **GHISA** (Global Hyperspectral Imaging Spectral-library
      of Agricultural crops, Thenkabail / Aneece)? Originalpublikation finden und lesen.
    - Was ist eine Zeile: ein einzelnes Pixel, ein gemitteltes Feldspektrum? Aus welchen Jahren
      und Regionen stammen die Aufnahmen?
    - Welche Vorverarbeitung haben die Autor:innen schon gemacht (atmosphärische Korrektur,
      Bandauswahl, Glättung)?
    - Wie definiert die Quelle die Stadien-Labels? (→ [Kulturpflanzen](../domaene/kulturpflanzen.md))
    - **AEZ:** Welche Einteilung wird verwendet (FAO/GAEZ, USGS)? Was bedeuten die Zonen
      2 und 5–10 (Klima, Region)? Warum könnte die AEZ bei der Klassifikation helfen?
    - Bekannte Schwächen oder Einschränkungen laut Quelle?

  [^1]: Domänenprojekt 2 (2026), siehe [Quellen](../domaene/quellen.md).

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
