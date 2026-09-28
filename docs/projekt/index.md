# Aufgabenstellung

Quelle: `data/assets/Beschreibung_Domaenen_Projekt_2_ Vorabversion_2026_27.docx` (Vorabversion,
Änderungen möglich) und die Kickoff-Folien.

## Ziel

Aus satellitengestützten Hyperspektraldaten (EO-1 Hyperion) sollen

1. die **Kulturpflanze** (Crop) und
2. deren **Entwicklungsstadium** (Growing Stage)

klassifiziert werden. Format: **Challenge** – alle Teams bearbeiten dieselbe Aufgabe.

## Mögliche Modellierungsansätze

Der Ansatz ist frei wählbar, muss aber **begründet** werden (Vor-/Nachteile). Der Vergleich
mehrerer Strategien ist ausdrücklich Teil der Aufgabe.

| Ansatz | Idee |
| --- | --- |
| Hierarchisch | Erst Crop, dann Stage innerhalb der vorhergesagten Crop-Klasse |
| Gemeinsam | Crop + Stage zu einer Klasse kombinieren (`corn_Late`, …), klassische Multiklassen-Aufgabe |
| Separat | Zwei unabhängige Modelle für Crop und Stage |
| Weitere | z. B. Multi-Task-Learning, eigene hierarchische Ansätze |

**Datenrepräsentation:**

- *Tabellarisch*: jedes Band ist ein Feature → Random Forest, Gradient Boosting, SVM, MLP
- *Sequenziell*: das Spektrum als geordnete Sequenz (entlang der Wellenlänge, **keine** Zeitreihe)
  → 1D-CNN, RNN/LSTM/GRU, Transformer

## Vorgehen laut Aufgabenstellung

1. **EDA**: Klassenverteilung, spektrale Signaturen je Crop/Stage, Ausreißer, Überschneidungen
2. **Data Preparation**: Bereinigung, Skalierung, Bandauswahl/Glättung, Feature Engineering
   (Vegetationsindizes, Band-Ratios, Gradienten), Dimensionsreduktion (PCA, Feature Selection)
3. **Modelling** und Vergleich der Strategien
4. **Evaluation** mit Metriken, die bei unbalancierten Klassen aussagekräftig sind

Literatur zu Hyperspectral Remote Sensing / Crop Phenology ist ausdrücklich erwünscht. Alle
Annahmen und Entscheidungen reproduzierbar dokumentieren.

## Bonus

Feature-Importance / Explainability (z. B. SHAP, Permutation Importance, Integrated Gradients):
Welche Wellenlängen sind für welche Crops/Stages entscheidend – fachlich begründet.
