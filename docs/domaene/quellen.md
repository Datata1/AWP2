# Quellen

Alle genutzten Quellen, alphabetisch nach Autor:in. In den Seiten per Fußnote zitieren.
Format: Autor:innen (Jahr): *Titel*. Verlag/Journal. Link. – eine Zeile, wofür wir sie nutzen.

## Datensatz & Sensor

!!! todo
    - Originalpublikation des Datensatzes (vermutlich USGS GHISA, Thenkabail / Aneece)
    - Offizielle Beschreibung EO-1 Hyperion (USGS / NASA)

- Domänenprojekt 2 (2026): *Domaeneprojekt Kickoff Teil Aufgabenbeschreibung*, Folie 7.
    Lokales Projektmaterial unter `data/assets/`. - Karte zum Untersuchungsgebiet Nordamerika
    sowie Erklärung der AEZ als Gruppierung nach Klima, Böden und Vegetationsperiode.

## Fernerkundung & Vegetation

!!! todo
    - Literatur zu Hyperspektral-Fernerkundung und Vegetationsindizes

- Daughtry, C. S. T., Hunt, E. R. & McMurtrey, J. E. (2004): *Assessing crop residue cover
    using shortwave infrared reflectance*. Remote Sensing of Environment, 90(1), 126-134.
    https://doi.org/10.1016/j.rse.2003.10.023. - Cellulose Absorption Index und trockene
    Pflanzenbestandteile.

- Gamon, J. A., Peñuelas, J. & Field, C. B. (1992): *A narrow-waveband spectral index that
    tracks diurnal changes in photosynthetic efficiency*. Remote Sensing of Environment,
    41(1), 35-44. https://doi.org/10.1016/0034-4257(92)90085-3. - Photochemical Reflectance
    Index (PRI).

- Gao, B.-C. (1996): *NDWI - A normalized difference water index for remote sensing of
    vegetation liquid water from space*. Remote Sensing of Environment, 58(3), 257-266.
    https://doi.org/10.1016/S0034-4257(96)00067-3. - NIR-SWIR-Wasserindex für Vegetation.

- Huete, A., Didan, K., Miura, T., Rodriguez, E. P., Gao, X. & Ferreira, L. G. (2002):
    *Overview of the radiometric and biophysical performance of the MODIS vegetation indices*.
    Remote Sensing of Environment, 83(1-2), 195-213.
    https://doi.org/10.1016/S0034-4257(02)00096-2. - Enhanced Vegetation Index (EVI).

- Knipling, E. B. (1970): *Physical and physiological basis for the reflectance of visible
    and near-infrared radiation from vegetation*. Remote Sensing of Environment, 1, 155-159.
    https://doi.org/10.1016/0034-4257(70)90021-9. - Zusammenhang von Blattmerkmalen mit
    der Reflexion im sichtbaren und nahen Infrarot.

- Mulla, D. J. (2013): *Twenty five years of remote sensing in precision agriculture: Key
    advances and remaining knowledge gaps*. Biosystems Engineering, 114(4), 358-371.
    https://doi.org/10.1016/j.biosystemseng.2012.08.009. - Wiederholte Fernerkundung für
    landwirtschaftliches Monitoring und mögliche Entscheidungsunterstützung.

- NASA Earthdata (o. J.): *Earth Observation Data Basics*. NASA.
    https://earthdata.nasa.gov/learn/backgrounders/remote-sensing. - Fernerkundung erfasst
    reflektierte oder emittierte Energie mit Instrumenten auf Satelliten oder Flugzeugen.

- NASA Earthdata (o. J.): *Resolution*. NASA.
    https://earthdata.nasa.gov/learn/earth-observation-data-basics/remote-sensing-resolution.
    - Spektrale, räumliche, zeitliche und radiometrische Auflösung sowie der Unterschied
    zwischen multi- und hyperspektralen Sensoren.

- NASA (2025): *Earth Observing-1*. NASA Earth Observatory.
    https://science.nasa.gov/earth/earth-observatory/earth-observing-1. - Start, Orbit und
    Zweck der EO-1-Mission sowie räumliche und spektrale Kenndaten von Hyperion.

- Tucker, C. J. (1979): *Red and photographic infrared linear combinations for monitoring
    vegetation*. Remote Sensing of Environment, 8(2), 127-150.
    https://doi.org/10.1016/0034-4257(79)90013-0. - Normalized Difference Vegetation Index
    (NDVI).

- U.S. Geological Survey (o. J.): *Landsat Collection 2 Surface Reflectance*. USGS.
    https://www.usgs.gov/landsat-missions/landsat-collection-2-surface-reflectance. - Begriff
    der Oberflächenreflektanz sowie Einfluss und Korrektur atmosphärischer Effekte.

## Kulturpflanzen & Phänologie

!!! todo
    - kulturspezifische Stadien-Skalen für Mais, Soja und Winterweizen

- Allen, R. G., Pereira, L. S., Raes, D. & Smith, M. (1998): *Crop evapotranspiration -
    Guidelines for computing crop water requirements*. FAO Irrigation and Drainage Paper 56.
    https://www.fao.org/4/x0490e/x0490e0a.htm. - Veränderungen von Bodenbedeckung,
    Pflanzenhöhe und Blattfläche während der Kulturentwicklung.

- Meier, U. (Hrsg.) (2018): *Growth stages of mono- and dicotyledonous plants: BBCH
  Monograph*. Julius Kühn-Institut.
  https://doi.org/10.5073/20180906-074619. - Standardisierte Skala für beobachtbare
  Entwicklungsstadien.

## Machine Learning & Metriken

!!! todo
    - Literatur zu hierarchischer Klassifikation und räumlicher Kreuzvalidierung

- scikit-learn Developers (2026a): *Metrics and scoring: quantifying the quality of
    predictions*. scikit-learn User Guide.
    https://scikit-learn.org/stable/modules/model_evaluation.html. - Confusion Matrix,
    Precision, Recall, F1, Mittelungen und Balanced Accuracy.

- scikit-learn Developers (2026b): *Cross-validation: evaluating estimator performance*.
    scikit-learn User Guide. https://scikit-learn.org/stable/modules/cross_validation.html.
    - Train-/Validierungsaufteilung, stratifizierte und gruppierte Cross-Validation.

- scikit-learn Developers (2026c): *Pipelines and composite estimators*. scikit-learn User
    Guide. https://scikit-learn.org/stable/modules/compose.html. - Pipelines als Schutz vor
    Leakage bei Vorverarbeitung und Parameterwahl.
