# Ansätze

Die Aufgabe lässt offen, wie wir Kultur und Stadium modellieren; die Wahl muss **begründet**
und mit Alternativen verglichen werden (siehe [Aufgabenstellung](../../projekt/index.md)).
Begriffe: [ML-Aufgabe](../../domaene/ml-aufgabe.md#problemformulierung).

- Ein **Ansatz** legt fest, *wie* Crop und Stage gemeinsam vorhergesagt werden (getrennt,
  kombiniert, hierarchisch …) – eine Seite je Ansatz.
- Ein **Klassifikator** ist das Modell darin (Random Forest, SVM, 1D-CNN …) – ein Unterkapitel
  je Klassifikator auf der Seite des Ansatzes. Zweistufige Ansätze (getrennt, hierarchisch)
  können je Stufe einen anderen nutzen; dann heißt das Unterkapitel nach der Kombination, z. B.
  „Random Forest → SVM“ (Kultur → Stadium).

## Überblick

| Ansatz | Status | Klassifikatoren | Bestes Experiment | Entscheidung |
| --- | --- | --- | --- | --- |
| [Baseline](baseline.md) | verglichen | Dummy, Random Forest | `rf_baseline` (BAcc kombiniert 0.743) | Referenz für alle weiteren Ansätze |
| [Getrennte Modelle](getrennt.md) | offen | – | | |
| [Kombinierte Klasse](kombiniert.md) | offen | – | | |
| [Hierarchisch](hierarchisch.md) | offen | – | | |
| [Multi-Task](multi-task.md) | offen | – | | |

Status: *offen* → *geplant* → *in Arbeit* → *verglichen* → *gewählt* / *verworfen*.

## Ansatz × Klassifikator

Beste BAcc kombiniert auf der Validierung je Kombination – ein Klick auf die Zahl führt zum
Unterkapitel. Eine neue Zeile, sobald ein Klassifikator in irgendeinem Ansatz getestet wurde.

| Klassifikator | Baseline | Getrennt | Kombiniert | Hierarchisch | Multi-Task |
| --- | --- | --- | --- | --- | --- |
| Dummy | [0.044](baseline.md#dummy) | | | | |
| Random Forest | [**0.743**](baseline.md#random-forest) | | | | |

So sieht man auf einen Blick, ob ein Klassifikator in allen Ansätzen gut ist oder ob ein Ansatz
mit jedem Klassifikator besser abschneidet.
Quer zu allen Ansätzen: [Datenrepräsentation](datenrepraesentation.md) (tabellarisch oder
sequenziell).

## Aufbau einer Ansatz-Seite

```markdown
# <Ansatz>

## Idee
Was der Ansatz modelliert und warum er für unsere Aufgabe in Frage kommt.

## Vor- und Nachteile
| Vorteil | Nachteil |

## Klassifikatoren

### <Klassifikator, z. B. Random Forest – bei zwei Stufen: Random Forest → SVM>
- Einstellungen (per `tune()` gefunden), Vorverarbeitung
- Ergebnis: BAcc Crop / Stage / kombiniert, CV-Score – Notebook und MLflow-Run verlinken
- Erkenntnisse (2–5 Stichpunkte)

## Fazit
Status, Entscheidung, Begründung.
```

Wie ein Ansatz entsteht und bewertet wird: [Einen Ansatz entwickeln](../ansatz-entwickeln.md).
