# Ansätze

Die Aufgabe lässt offen, wie wir Kultur und Stadium modellieren; die Wahl muss **begründet**
und mit Alternativen verglichen werden (siehe [Aufgabenstellung](../../projekt/index.md)).
Begriffe: [ML-Aufgabe](../../domaene/ml-aufgabe.md#problemformulierung).

- Ein **Ansatz** legt fest, *wie* Crop und Stage gemeinsam vorhergesagt werden (getrennt,
  kombiniert, hierarchisch …) – eine Seite je Ansatz.
- Eine **Architektur** ist das Modell darin (Random Forest, SVM, 1D-CNN …) – ein Unterkapitel
  je Architektur auf der Seite des Ansatzes.

## Überblick

| Ansatz | Status | Architekturen | Bestes Experiment | Entscheidung |
| --- | --- | --- | --- | --- |
| [Baseline](baseline.md) | verglichen | Dummy, Random Forest | `rf_baseline` (BAcc kombiniert 0.743) | Referenz für alle weiteren Ansätze |
| [Getrennte Modelle](getrennt.md) | offen | – | | |
| [Kombinierte Klasse](kombiniert.md) | offen | – | | |
| [Hierarchisch](hierarchisch.md) | offen | – | | |
| [Multi-Task](multi-task.md) | offen | – | | |

Status: *offen* → *geplant* → *in Arbeit* → *verglichen* → *gewählt* / *verworfen*.
Quer zu allen Ansätzen: [Datenrepräsentation](datenrepraesentation.md) (tabellarisch oder
sequenziell).

## Aufbau einer Ansatz-Seite

```markdown
# <Ansatz>

## Idee
Was der Ansatz modelliert und warum er für unsere Aufgabe in Frage kommt.

## Vor- und Nachteile
| Vorteil | Nachteil |

## Architekturen

### <Architektur, z. B. Random Forest>
- Einstellungen (per `tune()` gefunden), Vorverarbeitung
- Ergebnis: BAcc Crop / Stage / kombiniert, CV-Score – Notebook und MLflow-Run verlinken
- Erkenntnisse (2–5 Stichpunkte)

## Fazit
Status, Entscheidung, Begründung.
```

Wie ein Ansatz entsteht und bewertet wird: [Einen Ansatz entwickeln](../ansatz-entwickeln.md).
