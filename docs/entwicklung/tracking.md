# Experiment-Tracking

Jeder Lauf über `awp2.experiment.run()` wird automatisch in **MLflow** protokolliert:
Parameter von Vorverarbeitung und Modell, alle Metriken, die Confusion Matrices, Git-Commit,
Branch und Autor:in. So lassen sich viele Modelle vergleichen, ohne Ergebnisse von Hand zu notieren.

## Nutzung

```python
from awp2.experiment import CombinedLabelClassifier, run

result = run(CombinedLabelClassifier(model), "svm_combined")   # wird getrackt
result.run_id                                                  # MLflow-Run-ID
run(model, "schneller_test", track=False)                      # nicht tracken
run(model, "finales_modell", log_model=True)                   # Modell mitspeichern
```

```bash
make mlflow        # UI auf http://127.0.0.1:5000
```

In der UI: Experiment **awp2** öffnen → Runs anhaken → **Compare** zeigt Parameter und Metriken
nebeneinander, inklusive Plots. Nach `bacc_combined` sortieren, um das beste Modell zu finden.

| Was | Wo |
| --- | --- |
| Runs (Parameter, Metriken, Tags) | `mlflow.db` im Projekt-Root |
| Artefakte (Plots, Modelle) | `mlruns/` im Projekt-Root |
| Gespeichertes Modell laden | `mlflow.sklearn.load_model("runs:/<run_id>/model")` |

Beides ist **gitignored** und liegt nur lokal. Tag `git.dirty = True` heißt: Der Lauf hatte
uncommittete Änderungen in `src/` – für nachvollziehbare Ergebnisse vorher committen.

## Im Team vergleichen

Jede:r sieht in MLflow nur die **eigenen** Runs. Ergebnisse, die das Team kennen soll, kommen
deshalb zusätzlich ins [Experiment-Log](../modelle/experimente.md) – mit der MLflow-Run-ID, damit
sich der Lauf wiederfinden lässt. `/experiment` erledigt das automatisch.

## Entscheidung: MLflow lokal

| Tool | Stärken | Schwächen |
| --- | --- | --- |
| **MLflow lokal** ✅ | Industriestandard, Open Source, UI zum Vergleichen, kein externer Dienst | Runs nur lokal sichtbar |
| MLflow + DagsHub | Gemeinsamer gehosteter Server | Externer Dienst, Accounts |
| Weights & Biases | Sehr gute Team-Dashboards | Cloud-Dienst, Account je Person |
| DVC Experiments | Runs in git, kein Server | Mehr Konzepte, schwächere UI |
| Nur Markdown-Log | Kein Tool | Kein interaktiver Vergleich |

Begründung: Wir wollten MLflow als verbreitetes Werkzeug kennenlernen und keinen externen Dienst
einbinden. Der Nachteil (keine gemeinsame Ansicht) wird durch das Experiment-Log im Repo
ausgeglichen. Das Tracking steckt komplett in `src/awp2/tracking.py` – ein Wechsel auf einen
gemeinsamen Server (`MLFLOW_TRACKING_URI` setzen) oder ein anderes Tool betrifft nur diese Datei.
