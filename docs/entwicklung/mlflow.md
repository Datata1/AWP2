# MLflow – Überblick

Diese Seite erklärt, **was** MLflow ist, **warum** wir es nutzen und **welche Teile** davon.
Wie man damit arbeitet, steht unter [Experimente ausführen](tracking.md).

## Was ist MLflow?

MLflow ist eine Open-Source-Plattform für den Lebenszyklus von Machine-Learning-Modellen
(ursprünglich von Databricks, heute Linux Foundation). Sie ist ein Baukasten aus mehreren,
unabhängig nutzbaren Teilen:

| Bestandteil | Wofür |
| --- | --- |
| **Tracking** | Jeden Trainingslauf mit Parametern, Metriken, Tags und Dateien aufzeichnen und in einer Web-Oberfläche vergleichen |
| **Models** | Trainierte Modelle in einem einheitlichen Format speichern und wieder laden – mit Umgebung und Signatur |
| **Model Registry** | Versionen von Modellen verwalten und freigeben (z. B. „Staging“ → „Production“) |
| **Evaluate** | Modelle mit vorgefertigten Metriken und Plots automatisch bewerten |
| **Projects** | Trainingscode mit Umgebung paketieren, damit andere ihn reproduzierbar ausführen |
| **Deployments / Serving** | Modelle als REST-Dienst bereitstellen |
| **Tracing / GenAI** | Aufrufe von LLM-Anwendungen nachverfolgen |

Technisch besteht MLflow aus einer Python-Bibliothek, einem **Tracking-Store** (Datenbank für
Läufe), einem **Artefakt-Store** (Ordner für Dateien) und der **Web-Oberfläche**.

## Wie hilft es uns im Projekt?

Wir probieren in wenigen Wochen viele Modelle, Vorverarbeitungen und Einstellungen aus – und
müssen am Ende begründen, warum wir uns für einen Ansatz entschieden haben.

| Ohne MLflow | Mit MLflow |
| --- | --- |
| Scores von Hand in Tabellen kopieren | Jeder Lauf mit `run()` wird automatisch gespeichert |
| „Welche Einstellungen hatte der gute Lauf von gestern?“ | Alle Parameter liegen beim Lauf |
| „Mit welchem Code- und Datenstand war das?“ | Git-Commit, Branch, Autor:in und Datenversion als Tags |
| Plots suchen oder neu erzeugen | Confusion Matrices hängen am Lauf |
| Modell neu trainieren, um es wieder zu nutzen | Gespeichertes Modell jederzeit ladbar |
| Läufe mühsam nebeneinanderlegen | In der Oberfläche anhaken → **Compare** |

Für den Abschlussbericht heißt das: Jede Zahl lässt sich auf Code, Daten und Einstellungen
zurückführen.

## Begriffe

| Begriff | Bedeutung | Bei uns |
| --- | --- | --- |
| **Experiment** | Sammelmappe für Läufe, die dieselbe Frage beantworten | `crop-stage` für die Hauptaufgabe; eigene Experimente nur für andere Fragen (z. B. die Bandstudie in M3) |
| **Run** (Lauf) | Ein Training mit Bewertung | Bewertungslauf: ein Aufruf von `run()`, z. B. `rf_tuned`; Tuning-Lauf: ein Aufruf von `tune()` mit einem **Unterlauf** (nested run) je ausprobierter Einstellung |
| **Beschreibung** | Freitext am Lauf | Was probiert wurde und warum (`description`) |
| **Parameter** | Einstellungen eines Laufs | `preprocessing.scale`, `balance_samples`, `model.max_depth` … |
| **Metrik** | Gemessene Zahl | `bacc_combined`, `bacc_crop`, `f1_samples` … |
| **Tag** | Zusatzinformation | `approach` (Ansatz), `run.kind` (`evaluation`, `tuning`, `tuning-candidate`), `tuning_run`, `git.commit`, `git.branch`, `git.dirty`, `author` |
| **Dataset** | Welche Daten ein Lauf nutzte – nur Name, Hash, Schema und Quelle, nicht die Daten | `training` und `validation` aus `data/processed/split.csv` |
| **Logged Model** | Gespeichertes Modell eines Laufs, mit seinen Metriken | Jeder Lauf speichert sein Modell (Tab *Models*) |
| **Artefakt** | Datei am Lauf | `confusion_matrices.png` |
| **System-Metriken** | CPU-, Speicher-, Festplatten-Verlauf während des Laufs | Nur mit `system_metrics=True` |
| **Tracking-Store** | Datenbank der Läufe | `mlflow.db` im Projekt-Root (nicht in git) |
| **Artefakt-Store** | Ablage der Dateien | `mlruns/` im Projekt-Root (nicht in git) |

**Ein Experiment je Frage, nicht je Ansatz:** Die Kernfrage „welcher Ansatz ist am besten?“
vergleicht Läufe *verschiedener* Ansätze – das geht innerhalb eines Experiments am einfachsten.
Der Ansatz steht deshalb als Tag `approach` an jedem Lauf; in der Oberfläche lässt sich danach
filtern (`tags.approach = 'hierarchical'`) und gruppieren (*Group by*).

## Was wir nutzen – und was nicht

| Bestandteil | Nutzen wir? | Warum |
| --- | --- | --- |
| **Tracking** | ✅ ja | Kern: Läufe vergleichbar und nachvollziehbar machen – mit Beschreibung, Datasets und optional System-Metriken |
| **Models** (Logged Models) | ✅ ja | Jeder Lauf speichert sein Modell; es lässt sich laden, z. B. für die Vorhersagen der Abgabe |
| Tracking-**Server** | ❌ nein | Wir speichern lokal; kein Dienst, den jemand betreiben muss |
| **Model Registry** | ❌ nein | Freigabe-Workflow für den Betrieb – wir liefern eine Vorhersage-CSV, kein Modell in Produktion |
| **Evaluate** | ❌ nein | Unsere Metriken sind durch die Bewertung vorgegeben und stecken in `awp2.evaluation` |
| **Autologging** | ❌ nein | Loggt sehr viel Unwichtiges; wir loggen gezielt, was wir vergleichen |
| **Projects** | ❌ nein | Reproduzierbarkeit lösen wir mit `uv` und `make data` |
| **Serving / Deployments** | ❌ nein | Kein Betrieb eines Modells vorgesehen |
| **Tracing / GenAI** | ❌ nein | Für LLM-Anwendungen gedacht, nicht für unsere Klassifikatoren |

## Die Oberfläche

Oben links schaltet MLflow zwischen **GenAI** und **Model training** um. Standardmäßig steht es
auf *GenAI* – für LLM-Anwendungen. **Einmal auf „Model training“ klicken**, der Browser merkt
sich die Wahl. Danach:

| Wo | Was du siehst |
| --- | --- |
| Experiment `crop-stage` → **Runs** | Alle Läufe mit Datasets, Dauer; Spalten für Metriken/Parameter wählbar, *Group by* `approach` |
| Lauf → **Overview** | Beschreibung, Parameter, Metriken, Tags, Datasets, verknüpftes Modell |
| Lauf → **Model metrics** | Metriken als Diagramm |
| Lauf → **System metrics** | CPU/Speicher – nur bei `system_metrics=True` |
| Lauf → **Artifacts** | `confusion_matrices.png` (anklicken für die Vorschau) |
| Experiment → **Models** | Alle gespeicherten Modelle mit ihren Metriken |

**Bewusst leer:** *Traces*, *Sessions*, *Judges*, *Prompts*, *Evaluation*, *Model Registry*,
*AI Gateway* und der *MLflow Assistant* – das sind Funktionen für LLM-Anwendungen oder den
Betrieb, die wir nicht nutzen.

Im Code spricht nur `awp2.tracking` mit MLflow. Alle anderen Module wissen nichts davon – ein
Wechsel des Werkzeugs betrifft nur diese eine Datei.

## Warum MLflow – und nicht etwas anderes?

| Werkzeug | Stärken | Schwächen |
| --- | --- | --- |
| **MLflow lokal** ✅ | Weit verbreitet, Open Source, Oberfläche zum Vergleichen, kein externer Dienst | Läufe nur auf dem eigenen Rechner sichtbar |
| MLflow + DagsHub | Gemeinsamer, gehosteter Server | Externer Dienst, Accounts für alle |
| Weights & Biases | Sehr gute Team-Dashboards | Cloud-Dienst, Account je Person |
| DVC Experiments | Läufe in git, kein Server | Mehr Konzepte, schwächere Oberfläche |
| Nur Markdown-Log | Kein Werkzeug nötig | Kein interaktiver Vergleich, alles von Hand |

Entscheidung (#47): **MLflow lokal**. Wir wollten ein verbreitetes Werkzeug kennenlernen und
keinen externen Dienst einbinden. Der Nachteil – keine gemeinsame Ansicht – gleicht das
[Experiment-Log](../modelle/experimente.md) im Repo aus. Soll es später doch ein gemeinsamer
Server sein, reicht es, `MLFLOW_TRACKING_URI` zu setzen.
