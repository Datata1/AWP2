# Domäne

Hier halten wir fest, was wir über die Domäne wissen müssen, bevor wir mit den Daten arbeiten:
Fernerkundung, Vegetation, Kulturpflanzen und die ML-Aufgabe. Grundlage für den
M1-Kurzreport „Problemverständnis" und den Abschlussbericht.

!!! info "So füllen wir die Seiten"
    Jeder Abschnitt enthält Leitfragen als TODO. Beantworten, Leitfragen dann löschen.
    Kurz und mit eigenen Worten; Quellen als Fußnote (`[^1]`) und in [Quellen](quellen.md)
    eintragen; neue Fachbegriffe ins [Glossar](glossar.md).

## Leseleitfaden

| Seite | Inhalt |
| --- | --- |
| [Fernerkundung](fernerkundung.md) | Wie Satelliten Reflexion messen, der Sensor EO-1 Hyperion |
| [Vegetation](vegetation.md) | Spektrale Signatur von Pflanzen, Vegetationsindizes |
| [Kulturpflanzen](kulturpflanzen.md) | Die fünf Kulturen und ihre Entwicklungsstadien |
| [ML-Aufgabe](ml-aufgabe.md) | Problemformulierung, Bewertungsmetriken, Validierung |
| [Glossar](glossar.md) | Alle Fachbegriffe kurz erklärt |
| [Quellen](quellen.md) | Literatur |

Zum Datensatz selbst (Herkunft, Spalten, AEZ): [Daten](../daten/index.md).

## Projektziel & Nutzen

### Projektziel

Ziel dieses Projekts ist es, anhand hyperspektraler Satellitenmessungen automatisch zu
erkennen, welche Kultur auf einer beobachteten Fläche wächst und in welchem
Entwicklungsstadium sie sich befindet. Dazu erhält ein Modell die Reflexionswerte über viele
Wellenlängen sowie Kontextinformationen zur agroökologischen Zone und zum Aufnahmemonat.
Es soll daraus die Kulturart (`Crop`) und das Entwicklungsstadium (`Stage`) ableiten.

### Möglicher Nutzen und Nutzergruppen

Satellitendaten ermöglichen wiederholte Beobachtungen großer oder schwer zugänglicher
Gebiete und können Feldbegehungen ergänzen.[^1] Mögliche Nutzergruppen solcher Informationen
sind landwirtschaftliche Betriebe, Beratungsdienste, Behörden und Agrarstatistik. Das Projekt
prüft jedoch nur die Vorhersagequalität im vorliegenden Datensatz; ob die Ergebnisse für eine
bestimmte Anwendung ausreichen, ist nicht Teil der Untersuchung.

### Bedeutung des Entwicklungsstadiums

Das Entwicklungsstadium ordnet eine Kultur in ihren saisonalen Entwicklungsverlauf ein.
Zusammen mit der Kulturart kann es eine Grundlage für Fragen zur Bewässerung, Düngung,
Pflanzenschutz, Ernteplanung und Ertragsprognose bilden.[^1] Welche dieser Entscheidungen
durch die Daten tatsächlich verbessert wird, untersuchen wir in diesem Projekt nicht.

### Bewertung und Bedeutung von Fehlern

Unabhängig vom gewählten Modellansatz bewerten wir die Vorhersagequalität für Kulturart und
Entwicklungsstadium jeweils separat. Die Balanced Accuracy ist die primäre Metrik; ergänzend
berichten wir Macro-F1 und Samples-F1. Balanced Accuracy und Macro-F1 behandeln alle Klassen
gleich gewichtet. Der Samples-F1 berücksichtigt beide Labels je Beobachtung gemeinsam.

Die Aufgabenstellung legt nicht fest, ob bestimmte Fehler schwerer wiegen als andere. Deshalb
behandeln wir eine falsche Kulturart und ein falsches Stadium gleich. Auch benachbarte Stadien
erhalten in Balanced Accuracy und Macro-F1 keine besondere Teilbewertung: Ein vorhergesagtes
Label ist dort entweder richtig oder falsch.

## Zusammenfassung

In diesem Projekt soll ein Modell aus hyperspektralen Satellitenmessungen erkennen, welche
Kultur auf einer Fläche wächst und in welchem Entwicklungsstadium sie sich befindet. Ein
Spektrum beschreibt dafür, wie stark die Fläche Licht bei vielen verschiedenen Wellenlängen
reflektiert. Zusätzlich stehen die agroökologische Zone und der Aufnahmemonat als
Kontextinformationen zur Verfügung. Solche Satellitendaten können Feldbegehungen durch
wiederholte Beobachtungen großer oder schwer zugänglicher Gebiete ergänzen. Kulturart und
Entwicklungsstadium zusammen beschreiben, was auf einer Fläche wächst und wo sich die Kultur
in ihrem saisonalen Entwicklungsverlauf befindet. Welche Modellansätze wir vergleichen,
entscheiden wir später. Die Vorhersagequalität bewerten wir für beide Zielgrößen. Da die
Klassen unterschiedlich häufig vorkommen, ist die Balanced Accuracy die primäre Metrik.
Macro-F1 und Samples-F1 ergänzen sie. Die Untersuchung bewertet die Vorhersagequalität im
Datensatz, nicht den praktischen Nutzen für eine konkrete Entscheidung.

[^1]: Mulla, D. J. (2013): *Twenty five years of remote sensing in precision agriculture:
Key advances and remaining knowledge gaps*. Biosystems Engineering, 114(4), 358-371.
https://doi.org/10.1016/j.biosystemseng.2012.08.009.