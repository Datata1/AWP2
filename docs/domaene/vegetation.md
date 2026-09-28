# Spektrale Signatur von Vegetation

## Die typische Reflexionskurve

!!! todo "Leitfragen"
    - Wie sieht die typische Reflexionskurve gesunder Vegetation aus? (Skizze oder später
      Plot aus der EDA)
    - Was passiert bei ~550 nm (Grünpeak), ~670 nm (Chlorophyll-Absorption),
      680–750 nm (**Red Edge**), im NIR-Plateau und im SWIR?
    - Welche Pflanzeneigenschaften stecken dahinter: Chlorophyll, Zellstruktur,
      Blattflächenindex (LAI), Wassergehalt, Trockenmasse, Lignin/Zellulose?

## Wasserabsorptionsbänder

!!! todo "Leitfragen"
    - Warum absorbiert die Atmosphäre bei ~1400 nm und ~1900 nm fast das ganze Licht?
    - Passt das zu den 67 komplett leeren Bändern in unserem Datensatz
      (`X1326`–`X1508`, `X1770`–`X2052`, Ränder)? Siehe [Daten](../daten/index.md).

## Veränderung über die Saison

!!! todo "Leitfragen"
    - Wie verändert sich das Spektrum vom Auflaufen über die volle Entwicklung bis zur
      Seneszenz und Ernte?
    - Welchen Einfluss hat der sichtbare Boden bei jungen Pflanzen oder nach der Ernte?
    - Welche Spektralbereiche sollten demnach die Stadien am besten trennen?

## Vegetationsindizes

!!! todo "Leitfragen"
    - Was ist ein Vegetationsindex, und warum nutzt man Verhältnisse statt Rohwerte?
    - Kurz je Index (Formel, welche Bänder bei Hyperion, was er misst):
      NDVI, EVI, NDRE / Red-Edge-Position, PRI, NDWI bzw. NDII, CAI
    - Was sind Schmalband-Hyperspektralindizes (HVIs)?
    - Welche Indizes könnten für Kulturart bzw. Entwicklungsstadium nützlich sein?
      (Nur Ideen – das Feature Engineering kommt später.)
