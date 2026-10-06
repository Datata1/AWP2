# ML-Aufgabe, Metriken & Validierung

## Problemformulierung

### Was soll vorhergesagt werden?

Jede Zeile unseres Trainingsdatensatzes beschreibt ein Spektrum einer landwirtschaftlichen
Fläche. Aus 198 Reflexionsbändern sowie `AEZ` und `Month` sollen zwei Zielwerte vorhergesagt
werden:

- `Crop`: genau eine von fünf Kulturarten,
- `Stage`: genau eines von sechs Entwicklungsstadien.

Für jedes Ziel einzeln ist das eine **Multiklassen-Klassifikation**: Das Modell wählt genau
eine Kategorie aus mehreren möglichen Kategorien. Zusammen ist es eine **Multi-Output-
Klassifikation**, weil dasselbe Eingangsspektrum zwei verschiedene Zielspalten besitzt.

Von **Multi-Label-Klassifikation** spricht man üblicherweise, wenn eine Beobachtung beliebig
viele Labels gleichzeitig tragen kann, etwa „Wald“ und „Wasser“ in einem Luftbild. Unsere
Zielwerte sind strukturierter: Jede Beobachtung hat immer genau eine Kulturart und genau ein
Stadium. Die Aufgabenstellung verwendet den Begriff dennoch bei Samples-F1, weil diese Metrik
beide vorhergesagten Werte pro Beobachtung gemeinsam betrachtet.

### Mögliche Formulierungen

Die Aufgabe verlangt den begründeten Vergleich von Modellierungsstrategien, nicht eine
voreilige Festlegung.[^1]

| Ansatz | Idee | Vorteil | Risiko oder Nachteil |
| --- | --- | --- | --- |
| Getrennt | Ein Modell sagt `Crop`, ein anderes `Stage` vorher. | Einfach zu trainieren und je Ziel gut auswertbar. | Kann unmögliche Paare wie `rice|Harvest` erzeugen. |
| Kombiniert | Jedes beobachtete Crop-Stage-Paar ist eine Klasse, etwa `corn|Late`. | Gültige Paare werden direkt gelernt. | Seltene Paare wie `cotton|Harvest` haben sehr wenig Daten; die Klassenanzahl steigt. |
| Hierarchisch | Zuerst wird `Crop`, danach `Stage` innerhalb dieser Kultur vorhergesagt. | Spiegelt die Abhängigkeit der Zielwerte wider und kann ungültige Paare verhindern. | Ein Fehler bei `Crop` kann die zweite Vorhersage mit falsch leiten. |
| Multi-Task | Ein gemeinsames Modell lernt eine Spektralrepräsentation und hat getrennte Ausgänge für `Crop` und `Stage`. | Kann gemeinsame Information ausnutzen. | Höhere technische Komplexität; der Nutzen muss bei unserer Datenmenge erst gezeigt werden. |

Im Trainingsdatensatz fehlen mehrere Paare, etwa `rice|Harvest`. Diese Abwesenheit ist eine
feste Einschränkung der vorhandenen Stichprobe, aber noch kein Beweis, dass solche Paare
fachlich unmöglich sind. Alle Ansätze müssen daher ausweisen, wie sie ungültige Vorhersagen
erkennen oder verhindern.

### Zwei Sichten auf dasselbe Spektrum

Bei einer **tabellarischen** Darstellung ist jede Bandspalte ein eigenes Merkmal. Modelle wie
Random Forest, Gradient Boosting, SVM oder MLP erhalten damit eine Zeile mit 198 Werten. Sie
können Kombinationen von Bändern lernen, kennen die Nachbarschaft auf der Wellenlängenachse
aber nicht automatisch.

Bei einer **sequenziellen** Darstellung bleibt die Reihenfolge der Bänder erhalten. Ein
1D-CNN kann zum Beispiel lokale Muster wie den Anstieg an der Red Edge erkennen. RNN, GRU,
LSTM oder Transformer sind weitere mögliche Architekturen. Die Sequenzachse ist dabei die
**Wellenlänge**, nicht die Zeit: Die Werte `X427` bis `X2395` stammen aus einer Beobachtung,
nicht aus aufeinanderfolgenden Monaten.

Die tabellarische Sicht ist ein sinnvoller, gut nachvollziehbarer Start. Sequenzmodelle sind
eine spätere Vergleichsfrage, denn sie haben mehr Modellfreiheit und müssen ihren Zusatznutzen
gegenüber derselben Validierung erst belegen.

## Klassenungleichgewicht

Eine Klasse ist **unausgewogen**, wenn sie deutlich weniger Beobachtungen als andere Klassen
hat. Bei `Crop` stehen beispielsweise 2.097 Mais-Beobachtungen nur 93 Reis-Beobachtungen
gegenüber. Bei `Stage` ist `Harvest` mit 180 Beobachtungen ebenfalls selten. Ein Modell kann
seinen Durchschnittsscore verbessern, indem es häufige Klassen bevorzugt, und die seltenen
Klassen dabei fast ignorieren.

Das Problem wird bei kombinierten Crop-Stage-Paaren noch stärker: `cotton|Harvest` kommt im
gesamten Training nur 11-mal vor. Solche Beispiele reichen möglicherweise nicht aus, um ein
komplexes Klassenmuster verlässlich zu lernen.

### Mögliche Gegenmaßnahmen

| Maßnahme | Idee | Wichtig für dieses Projekt |
| --- | --- | --- |
| Klassengewichte | Fehler bei seltenen Klassen erhalten beim Training mehr Gewicht. | Für kombinierte Crop-Stage-Paare ist dies der bevorzugte erste Vergleich; der Projektcode stellt dafür ausgeglichene Gewichte bereit. |
| Oversampling | Seltene Trainingsbeobachtungen werden wiederholt ausgewählt. | Nur innerhalb des jeweiligen Trainingsanteils; sonst können Duplikate in die Validierung gelangen. |
| Undersampling | Häufige Klassen werden teilweise weggelassen. | Verringert die Dominanz von Mais, verwirft aber möglicherweise nützliche Spektren. |
| Synthetisches Oversampling, z. B. SMOTE | Neue Punkte zwischen seltenen Trainingspunkten werden konstruiert. | Bei hochdimensionalen Spektren können synthetische Kurven fachlich unplausibel sein; Crop-Stage-Paare müssen gültig bleiben. Daher nur eine begründete, später validierte Hypothese. |
| Klassenfaire Metriken | Jede Klasse beeinflusst die Bewertung gleich stark. | Balanced Accuracy und Macro-F1 sind bereits verpflichtend. |

Keine Maßnahme ersetzt eine ausreichende Zahl echter Beobachtungen. Für Reis oder seltene
Crop-Stage-Paare müssen wir daher auch die Unsicherheit und Fehleranalyse offen berichten.

## Bewertungsmetriken

Eine **Confusion Matrix** zählt, welche wahre Klasse als welche Klasse vorhergesagt wurde.
In unseren Diagrammen stehen die wahren Klassen in den Zeilen und die Vorhersagen in den
Spalten. Die Diagonale enthält korrekte Vorhersagen; außerhalb der Diagonale stehen die
Verwechslungen.

Für eine betrachtete Klasse, etwa `rice`, bedeuten die vier Grundbegriffe:

| Begriff | Bedeutung |
| --- | --- |
| True Positive (TP) | Ein Reis-Spektrum wird korrekt als Reis erkannt. |
| False Positive (FP) | Ein anderes Spektrum wird fälschlich als Reis vorhergesagt. |
| False Negative (FN) | Ein Reis-Spektrum wird als andere Kultur übersehen. |
| True Negative (TN) | Eine andere Kultur wird korrekt nicht als Reis vorhergesagt. |

**Precision** fragt: „Wenn das Modell Reis sagt, wie oft stimmt das?“
**Recall** fragt: „Welchen Anteil der echten Reis-Spektren findet das Modell?“
**F1** ist das harmonische Mittel aus Precision und Recall und fällt niedrig aus, wenn einer
der beiden Werte niedrig ist.[^2]

$$
\mathrm{Precision} = \frac{TP}{TP + FP}
\qquad
\mathrm{Recall} = \frac{TP}{TP + FN}
\qquad
F1 = 2 \cdot \frac{\mathrm{Precision} \cdot \mathrm{Recall}}
{\mathrm{Precision} + \mathrm{Recall}}
$$

### Warum nicht nur Accuracy?

Die gewöhnliche **Accuracy** ist der Anteil aller korrekten Vorhersagen. Bei ungleichen Klassen
kann sie täuschen: Angenommen, es gibt 90 Mais-, 5 Reis- und 5 Baumwoll-Beobachtungen. Ein
Modell, das immer Mais sagt, erreicht $90/100 = 0{,}90$ Accuracy. Sein Recall beträgt für Mais
1, für Reis 0 und für Baumwolle 0.

Die **Balanced Accuracy** bildet den Mittelwert dieser Klassen-Recalls. Im Beispiel ist sie
$(1 + 0 + 0)/3 = 0{,}33$. Sie macht damit sichtbar, dass das Modell zwei Klassen vollständig
ignoriert. Deshalb ist sie die primäre Projektmetrik.[^2]

### Mittelungen und Samples-F1

| Mittelung | Was wird gleich gewichtet? | Bedeutung bei uns |
| --- | --- | --- |
| Macro | Jede Klasse | Seltene Klassen wie Reis oder `Harvest` zählen genauso stark wie häufige Klassen. |
| Micro | Jede einzelne Vorhersage | Häufige Klassen beeinflussen das Ergebnis stärker. Bei einer einfachen Multiklassenaufgabe entspricht Micro-F1 der Accuracy. |
| Weighted | Jeder Klassenwert nach seiner Häufigkeit | Gibt einen Gesamtwert, kann schlechte Ergebnisse seltener Klassen aber verdecken. |
| Samples | Jede Beobachtung | Betrachtet `Crop` und `Stage` gemeinsam pro Zeile. |

Bei genau zwei Zielwerten ist Samples-F1 besonders anschaulich: Sind beide Werte korrekt,
erhält die Beobachtung 1; ist nur `Crop` oder nur `Stage` korrekt, erhält sie 0,5; sind beide
falsch, erhält sie 0. Der Samples-F1 ist der Mittelwert dieser Werte über alle Beobachtungen.

### Was die Kennzahlen aussagen und wo ihre Grenzen liegen

Alle Werte liegen zwischen 0 und 1. Jede Kennzahl beantwortet eine andere Frage – erst
zusammen ergeben sie ein ehrliches Bild. **Kombiniert** heißt jeweils: Das Paar
`Crop|Stage` gilt als eine Klasse (23 Paare in der Validierung) und zählt nur als richtig,
wenn **beide** Werte stimmen.

#### Balanced Accuracy (Hauptmetrik)

- **Entstehung:** Für jede wahre Klasse $k$ den Recall bestimmen, dann ungewichtet mitteln:
  $\mathrm{BAcc} = \frac{1}{K}\sum_{k=1}^{K} \mathrm{Recall}_k$.
- **Aussage:** „Welchen Anteil jeder Klasse erkennt das Modell im Durchschnitt?“ Reis zählt so
  viel wie Mais. Das **Zufallsniveau** ist $1/K$: 0,20 für 5 Kulturen, 0,17 für 6 Stadien,
  rund 0,04 für die Paare – ein Wert ist erst im Vergleich dazu einzuordnen.
- **Grenzen:**
    - Sie misst nur Recall, keine Precision. Sagt ein Modell zu oft Reis, sinkt nur der
      Recall der anderen Klassen etwas – Fehlalarme fallen kaum auf.
    - Alle Fehler wiegen gleich: Ein Nachbarstadium (`Early_Mid` statt `Late`) kostet so viel
      wie ein völlig falsches.
    - Seltene Klassen machen sie unruhig: In der Validierung gibt es 27 Reis-Zeilen – ein Fehler
      mehr kostet 3,7 Prozentpunkte Reis-Recall und damit 0,7 Punkte BAcc Crop. Bei
      `cotton|Harvest` (3 Zeilen) kostet ein Fehler 33 Punkte Recall dieser Klasse.

#### Macro-F1

- **Entstehung:** Für jede Klasse F1 aus Precision und Recall bilden, dann ungewichtet mitteln:
  $\mathrm{Macro\text{-}F1} = \frac{1}{K}\sum_{k=1}^{K} F1_k$.
- **Aussage:** Ergänzt die Balanced Accuracy um die Precision: Ein hoher Wert heißt, das Modell
  findet jede Klasse **und** vergibt sie nicht zu oft. Liegt Macro-F1 deutlich unter der
  Balanced Accuracy, sagt das Modell einzelne Klassen zu häufig vorher.
- **Grenzen:**
    - Wie bei der Balanced Accuracy wiegen alle Fehler gleich, und seltene Klassen streuen stark.
    - Eine Klasse, die nie vorhergesagt wird, hat F1 = 0 und zieht den Mittelwert deutlich
      herunter.
    - Kombiniert: Ein vorhergesagtes Paar, das in der Validierung nicht vorkommt (z. B. ein
      unmögliches Paar), zählt als zusätzliche Klasse mit F1 = 0. Ungültige Paare senken
      Macro-F1 kombiniert daher stärker als die Balanced Accuracy.

#### Samples-F1

- **Entstehung:** Je Beobachtung der Anteil richtiger Zielwerte (0, 0,5 oder 1), gemittelt über
  alle Beobachtungen. Bei genau einem `Crop`- und einem `Stage`-Label ist das der Mittelwert aus
  der gewöhnlichen Accuracy von `Crop` und von `Stage`.
- **Aussage:** „Wie viele der vorhergesagten Werte stimmen insgesamt?“ – anschaulich und
  verlangt von der Aufgabenstellung.
- **Grenzen:**
    - **Nicht klassenfair:** Häufige Klassen dominieren, genau wie bei der Accuracy. Die
      Dummy-Baseline, die immer Mais und `Critical` sagt, erreicht schon 0,33, obwohl sie nichts
      gelernt hat.
    - Sie verrät nicht, ob `Crop` oder `Stage` falsch war, und belohnt halb richtige Paare.

#### Für alle Kennzahlen

- Sie sind **Schätzungen auf einer Stichprobe** (1.677 Validierungszeilen). Die Streuung über
  die CV-Folds (bei der Baseline ±0,02) zeigt, wie groß Unterschiede mindestens sein müssen,
  um mehr als Zufall zu sein.
- Sie gelten für die Verteilung unserer Daten. Wie gut ein Modell auf anderen Regionen, Jahren
  oder Feldern funktioniert, sagen sie nicht – siehe
  [Räumliche Ähnlichkeit](#raumliche-ahnlichkeit-als-risiko).
- Welche Klassen verwechselt werden, zeigen erst Confusion Matrix und Recall pro Klasse.

### Was die Projektfunktion berechnet

`awp2.evaluation.evaluate()` gibt für `Crop`, `Stage` und das kombinierte
Crop-Stage-Label jeweils Balanced Accuracy und Macro-F1 zurück. Zusätzlich berechnet sie
Samples-F1 sowie, falls gültige Paare übergeben werden, den Anteil vorhergesagter
Crop-Stage-Kombinationen, die im Training nicht vorkamen. Confusion Matrices und Recall pro
Klasse ergänzen diese Kennzahlen für die Fehleranalyse.

Formeln der offiziellen Bewertung: [Bewertung & Abgabe](../projekt/bewertung.md).

## Validierung

**Validierung** beantwortet die Frage, wie gut ein Modell auf unbekannten Beobachtungen
funktioniert. Eine Vorhersage auf Daten, die das Modell beim Training bereits gesehen hat,
ist dafür kein aussagekräftiger Test.[^3]

| Datenteil | Rolle im Projekt |
| --- | --- |
| Training | Auf diesem Teil werden Modell und Vorverarbeitung gelernt. |
| Cross-Validation-Folds | Der Trainingsanteil wird fünfmal intern geteilt, um Hyperparameter und Varianten zu vergleichen. |
| Validierung | Bleibt während des Tunings unangetastet und bewertet die gewählte Variante genau einmal. |
| Unbeschriftete Testdaten | Enthalten keine Zielwerte und dürfen keine Modellentscheidung beeinflussen. Ob `test.csv` bereits dem späteren Challenge-Datensatz entspricht, ist noch offen. |

Ein **Holdout** ist ein Teil der beschrifteten Daten, der vor jeder Modellarbeit beiseitegelegt
und erst ganz am Ende einmal zur Bewertung genutzt wird – das Modell hat ihn nie gesehen, und
keine Entscheidung wurde anhand seiner Ergebnisse getroffen. Nur dann ist sein Score eine
ehrliche Schätzung für neue Daten. Bei uns sind „Holdout“ und „Validierung“ derselbe
30-%-Anteil; die unbeschrifteten Testdaten sind etwas anderes, weil sie keine Zielwerte haben.

Unser gemeinsamer Split reserviert 30 % der beschrifteten Daten als Validierung und stratifiziert
nach dem kombinierten Crop-Stage-Label. Dadurch sind seltene, aber vorhandene Paare möglichst
in Training und Validierung vertreten. Im 70%-Trainingsanteil sucht `tune()` per fünffacher
Cross-Validation nach Hyperparametern; `run()` bewertet die ausgewählte Konfiguration einmal
auf den 30 % Validierung.[^3]

### Räumliche Ähnlichkeit als Risiko

**Räumliche Autokorrelation** bedeutet, dass nahe beieinanderliegende Flächen häufig ähnliche
Spektren besitzen. Liegen Pixel desselben Feldes oder Bildausschnitts auf beiden Seiten eines
zufälligen Splits, kann ein Modell eher die lokale Aufnahme wiedererkennen als auf neue Felder
zu generalisieren. Der Validierungsscore wäre dann zu optimistisch.

Für unseren Datensatz fehlen derzeit Koordinaten und Feld-IDs. Wir können räumliche
Autokorrelation daher nicht direkt prüfen und nicht mit einem räumlichen Split kontrollieren.
Exakte Duplikate werden vor dem Split entfernt. Falls Feld- oder Bildgruppen verfügbar werden,
verwenden wir eine gruppierte, zugleich stratifizierte Cross-Validation, damit keine Gruppe
zwischen Training und Validierung geteilt wird.[^3]

Was mit dem in der Aufgabenbeschreibung genannten „70/30 Clusterplot-Split“ gemeint ist,
ist noch ungeklärt. Bis zur Antwort in den [offenen Fragen](../projekt/offene-fragen.md)
behandeln wir ihn nicht als festgelegte räumliche Validierungsregel.

## Data Leakage

**Data Leakage** liegt vor, wenn beim Lernen Informationen einfließen, die bei einer echten
Vorhersage nicht verfügbar wären. Der Validierungsscore wirkt dann besser, als das Modell auf
neuen Daten tatsächlich wäre.

Typische Risiken für dieses Projekt sind:

- einen Imputer, Skalierer, Glätter, PCA oder eine Bandauswahl auf allen 5.591 Zeilen zu
  lernen, bevor der Split gezogen wird,
- Duplikate oder sehr nahe Wiederholungen in Training und Validierung zu belassen,
- Indizes oder Spektralbereiche anhand des Validierungs- oder unbeschrifteten Testsets
  auszuwählen,
- nach einem Validierungsscore weiter zu tunen, bis derselbe Validierungsanteil gut aussieht,
- `AEZ` oder `Month` zu verwenden, obwohl diese Informationen bei der späteren Vorhersage
  nicht verfügbar oder nicht erwünscht wären.

Der Schutz folgt einer festen Reihenfolge: Zuerst werden die Daten geteilt. Danach liegen alle
lernbaren Schritte in einer sklearn-`Pipeline`, sodass sie in jedem Fold ausschließlich auf
dem jeweiligen Trainingsanteil angepasst werden. `tune()` nutzt nur den Trainingsanteil;
`run()` schaut für die finale Bewertung einmal auf die Validierung. Die Funktion
`prepare_dataset()` entfernt außerdem exakte Duplikate, bevor der gemeinsame Split erzeugt
wird.[^4]

Eine hohe Leistung mit `AEZ` oder `Month` ist nicht automatisch Leakage. Sind die Metadaten
bei der späteren Vorhersage verlässlich verfügbar, können sie legitime Kontextinformation sein.
Sie können aber ein fragiles Abkürzungssignal sein. Deshalb vergleichen wir Modelle mit und
ohne diese Metadaten transparent und dokumentieren die Entscheidung.

[^1]: Domänenprojekt 2 (2026), siehe [Quellen](quellen.md).
[^2]: scikit-learn Developers (2026a), siehe [Quellen](quellen.md).
[^3]: scikit-learn Developers (2026b), siehe [Quellen](quellen.md).
[^4]: scikit-learn Developers (2026c), siehe [Quellen](quellen.md).
