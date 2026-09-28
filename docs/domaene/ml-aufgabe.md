# ML-Aufgabe, Metriken & Validierung

## Problemformulierung

!!! todo "Leitfragen"
    - Was ist Multiklassen-, Multi-Label-, Multi-Output- und hierarchische Klassifikation?
    - Welche Formulierungen sind für uns möglich (hierarchisch, kombinierte Klasse, zwei
      getrennte Modelle, Multi-Task) – mit Vor- und Nachteilen, noch ohne Entscheidung.
    - Tabellarische vs. sequenzielle Sicht auf das Spektrum: was heißt das jeweils?

## Klassenungleichgewicht

!!! todo "Leitfragen"
    - Warum ist Ungleichgewicht ein Problem (Reis: 93 Zeilen, Mais: 2097)?
    - Welche Gegenmittel gibt es (Class Weights, Over-/Undersampling, SMOTE, passende Metriken)?

## Bewertungsmetriken

!!! todo "Leitfragen"
    - Confusion Matrix, TP/FP/FN/TN, Precision, Recall, F1 – je eine Zeile Erklärung.
    - Warum ist Accuracy bei Ungleichgewicht irreführend? Was macht **Balanced Accuracy**
      anders?
    - Macro- vs. Micro- vs. Weighted-Mittelung – was betont welche?
    - Was ist **Samples-F1** und wie ergibt er sich bei genau zwei Labels (Kultur + Stadium)?
    - Kleines Rechenbeispiel mit 3 Klassen, bei dem Accuracy und Balanced Accuracy stark
      auseinanderliegen.
    - Was berechnet `awp2.evaluation.evaluate()` genau (Crop, Stage, kombiniert)?

Formeln der offiziellen Bewertung: [Bewertung & Abgabe](../projekt/bewertung.md).

## Validierung

!!! todo "Leitfragen"
    - Train / Validation / Test: wofür ist welcher Teil da? Was ist unser „Test" (ungelabelte
      Daten der Dozenten)?
    - Stratifizierter Split und Cross-Validation – wann was?
    - **Räumliche Autokorrelation:** Spektren vom selben Feld oder Bild sind sich ähnlich.
      Warum führt ein zufälliger Split dann zu zu optimistischen Scores?
    - Was könnte der „70/30 Clusterplot-Split" aus der Aufgabenstellung sein?
      (→ [Offene Fragen](../projekt/offene-fragen.md))

## Data Leakage

!!! todo "Leitfragen"
    - Was ist Data Leakage? Typische Beispiele (Skalierung/PCA vor dem Split, Duplikate in
      Train und Validation, Tuning auf dem Testset).
    - Wie verhindern wir es (sklearn `Pipeline`, Split zuerst, Duplikate prüfen)?
