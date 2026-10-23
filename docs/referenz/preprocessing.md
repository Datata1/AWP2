# awp2.preprocessing

Die öffentliche API bleibt unter `awp2.preprocessing` erreichbar. Die Implementierung ist
nach Verantwortung gegliedert; Notebooks importieren weiterhin aus dem Paket und müssen
die internen Module nicht kennen.

| Modul | Verantwortung |
| --- | --- |
| `options` | `PreprocessingConfig` und Validierung der Optionen |
| `bands` | Bandprüfung, Interpolation, Glättung, SNV und Albedo |
| `indices` | Wellenlängenzuordnung, Vegetationsindizes und REIP |
| `decomposition` | PCA-Merkmale und weitere Dimensionsreduktionen |
| `metadata` | Monats- und AEZ-Monats-Kodierung |
| `pipeline` | Aufbau und Verknüpfung der unabhängigen Transformer-Zweige |
| `reporting` | Textbeschreibungen der konkreten Pipeline für die Experimentprotokollierung |

Neue Schritte erhalten eine Option in `PreprocessingConfig` und einen Transformer im
passenden Fachmodul. Die Einbindung erfolgt in `pipeline`, die Beschreibung in `reporting`.
Die [Optionsübersicht](../daten/pipeline.md#ablauf-der-vorverarbeitung) dokumentiert den
Ablauf und die Einschränkungen. Öffentliche Transformer werden in
`__init__.py` explizit exportiert.

::: awp2.preprocessing
