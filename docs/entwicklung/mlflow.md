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
| **Experiment** | Sammelmappe für zusammengehörige Läufe | Eines: `awp2` |
| **Run** (Lauf) | Ein Training mit Bewertung | Ein Aufruf von `run()`, z. B. `rf_baseline` |
| **Parameter** | Einstellungen eines Laufs | `preprocessing.scale`, `balance_samples`, `model.max_depth` … |
| **Metrik** | Gemessene Zahl | `bacc_combined`, `bacc_crop`, `f1_samples` … |
| **Tag** | Zusatzinformation | `git.commit`, `git.branch`, `git.dirty`, `author`, `data.version` |
| **Artefakt** | Datei am Lauf | `confusion_matrices.png`, optional das Modell |
| **Tracking-Store** | Datenbank der Läufe | `mlflow.db` im Projekt-Root (nicht in git) |
| **Artefakt-Store** | Ablage der Dateien | `mlruns/` im Projekt-Root (nicht in git) |

## Was wir nutzen – und was nicht

| Bestandteil | Nutzen wir? | Warum |
| --- | --- | --- |
| **Tracking** | ✅ ja | Kern: Läufe vergleichbar und nachvollziehbar machen |
| **Models** | ✅ teilweise | Nur zum Speichern und Wiederladen einzelner Modelle (`log_model=True`) |
| Tracking-**Server** | ❌ nein | Wir speichern lokal; kein Dienst, den jemand betreiben muss |
| **Model Registry** | ❌ nein | Wir bringen kein Modell in Produktion – abgegeben wird eine Vorhersage-CSV |
| **Evaluate** | ❌ nein | Unsere Metriken sind durch die Bewertung vorgegeben und stecken in `awp2.evaluation` |
| **Autologging** | ❌ nein | Loggt sehr viel Unwichtiges; wir loggen gezielt, was wir vergleichen |
| **Projects** | ❌ nein | Reproduzierbarkeit lösen wir mit `uv` und `make data` |
| **Serving / Deployments** | ❌ nein | Kein Betrieb eines Modells vorgesehen |
| **Tracing / GenAI** | ❌ nein | Für LLM-Anwendungen gedacht, nicht für unsere Klassifikatoren |

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
