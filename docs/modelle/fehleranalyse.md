# Fehleranalyse

Welche Klassen verwechselt werden, warum, und weshalb Baseline, kombinierter und hierarchischer
Ansatz unterschiedlich abschneiden (#32). Notebook: `notebooks/10_jd_fehleranalyse.ipynb`,
Bausteine: `error_breakdown()`, `recall_by_class()`, `compare_predictions()`
([Einen Ansatz entwickeln](ansatz-entwickeln.md#5-ergebnis-ansehen)).

!!! note "Validierung nur zur Analyse"
    Die Analyse nutzt die Vorhersagen auf der gemeinsamen Validierung, entscheidet aber nichts.
    Ideen daraus (siehe [Folgerungen](#folgerungen)) werden wie jede Hypothese per CV mit
    `tune()` geprüft. Die Muster nach Monat und AEZ sind zusätzlich auf Out-of-fold-Vorhersagen
    des Trainingsteils gegengeprüft.

## Verglichene Modelle

Je Ansatz das beste bewertete Modell, mit den protokollierten Einstellungen neu trainiert und auf
der gemeinsamen Validierung vorhergesagt – ohne neue Läufe und ohne weiteres Einstellen.

| Ansatz | Lauf | Vorverarbeitung | BAcc Crop | BAcc Stage | BAcc kombiniert |
| --- | --- | --- | --- | --- | --- |
| [Baseline](ansaetze/baseline.md) | `rf_baseline`, Random Forest | unskaliert, Monat numerisch | 0.888 | 0.866 | 0.750 |
| [Kombiniert](ansaetze/kombiniert.md) | `combined_svm_expanded`, RBF-SVM | `scale=True`, Monat numerisch | 0.931 | 0.906 | **0.866** |
| [Hierarchisch](ansaetze/hierarchisch.md) | `hierarchical_svm`, RBF-SVM → RBF-SVM | Spektren standardisiert, Monat zyklisch | 0.920 | 0.902 | 0.844 |

`combined_svm_expanded` und `hierarchical_svm` reproduzieren ihre protokollierten Werte (bis auf
Rundung). `rf_baseline` nicht: Das gespeicherte M1-Modell ist mit der heutigen Pipeline nicht mehr
ladbar (Interpolation geändert in `585fbf2`); nachtrainiert ergibt sich 0.750 statt 0.743 und
0,1 % ungültige Paare.

## Wo entstehen die Fehler?

![Fehlerzerlegung](img/fehleranalyse_zerlegung.png)

| | Baseline | Kombiniert | Hierarchisch |
| --- | --- | --- | --- |
| beides richtig | 0.815 | 0.877 | 0.871 |
| Kultur falsch | 0.118 | 0.084 | 0.086 |
| nur Stadium falsch | 0.067 | 0.039 | 0.043 |
| Stadium-BAcc bei richtiger Kultur | 0.922 | 0.957 | 0.950 |

- **Rund zwei Drittel der Fehler sind Kulturfehler** – in allen drei Ansätzen. Die Kultur ist
  der Engpass, nicht das Stadium.
- **Baseline → SVM:** Die SVMs senken die reinen Stadiumfehler um rund 40 % (6,7 % → 3,9–4,3 %)
  und die Kulturfehler um gut ein Viertel (11,8 % → 8,4–8,6 %).
- **Hierarchisch:** Mit der wahren Kultur wäre BAcc kombiniert 0.939 statt 0.844
  (Oracle-Kultur, Stadium-BAcc 0.947) – die Kulturfehler pflanzen sich ungebremst ins Stadium
  fort. Die Stadium-BAcc bei richtiger Kultur ist beim kombinierten Ansatz allerdings sogar etwas
  höher (0.957 vs. 0.950); die Stadium-Modelle je Kultur sind also nicht besser als die
  kombinierte SVM.

## Unterscheiden sich die Ansätze wirklich?

McNemar-Test auf kombinierte Treffer (nicht BAcc – seltene Klassen fallen kaum ins Gewicht):

| Vergleich | nur erster richtig | nur zweiter richtig | p |
| --- | --- | --- | --- |
| Baseline vs. Kombiniert | 51 | 155 | 2·10⁻¹³ |
| Baseline vs. Hierarchisch | 47 | 140 | 6·10⁻¹² |
| Kombiniert vs. Hierarchisch | 44 | 33 | 0.25 |

- Beide SVM-Ansätze schlagen die Baseline klar.
- **Kombiniert vs. hierarchisch: kein signifikanter Unterschied** bei den Treffern (p = 0.25).
  Sie machen weitgehend dieselben Fehler: 106 Zeilen haben in beiden die falsche Kultur (140 bzw.
  145 insgesamt), 89 sogar in allen drei Ansätzen. Diese „harten“ Zeilen begrenzen jeden Ansatz.

## Woher kommt der BAcc-Abstand zwischen kombiniert und hierarchisch?

Die Balanced Accuracy gewichtet jede der 23 Klassen gleich. Bei ähnlich vielen Treffern entsteht
der Abstand von 0.022 über die Verteilung auf die Klassen.

![Recall je Klasse](img/fehleranalyse_recall.png)

| Klasse | Zeilen | Kombiniert | Hierarchisch | Differenz |
| --- | --- | --- | --- | --- |
| cotton\|Emerge_VEarly | 52 | 0.769 | 0.654 | −0.115 |
| winter_wheat\|Late | 9 | 0.889 | 0.778 | −0.111 |
| soybean\|Harvest | 20 | 0.900 | 0.800 | −0.100 |
| corn\|Early_Mid | 64 | 0.906 | 0.812 | −0.094 |
| cotton\|Critical | 47 | 0.809 | 0.723 | −0.085 |
| winter_wheat\|Emerge_VEarly | 182 | 0.863 | 0.907 | +0.044 |
| soybean\|Emerge_VEarly | 32 | 0.562 | 0.656 | +0.094 |

- Der hierarchische Ansatz verliert vor allem bei **kleinen Klassen** (9–64 Zeilen) und gewinnt
  bei der großen Klasse Winterweizen/Emerge (182). Das gleicht sich bei den Treffern aus, in der
  BAcc aber nicht. Mehrere Klassen beruhen auf weniger als 20 Zeilen – einzelne Differenzen sind
  unsicher.
- **Vermutung (nicht getestet):** Die eigene Kultur-SVM zieht die Grenzen zwischen den Kulturen
  anders als die kombinierte SVM, die je Kultur und Stadium eigene Bereiche lernt; Grenzfälle
  kleiner Klassen fallen so eher auf die falsche Seite. Der Vergleich ist allerdings durch die
  Vorverarbeitung verzerrt (siehe Tabelle oben). Mit gleicher Vorverarbeitung liegen beide in der
  CV nah beieinander (`combined_svm_cv_reference` 0.845 vs. `hierarchical_svm` 0.839).

## Wann und wo liegt die Kultur falsch?

![Kulturfehler nach Monat](img/fehleranalyse_monat.png)

- **Saisonränder:** Im Mai/Juni und Oktober liegt die Kultur bei den SVMs in 15–25 % der Zeilen
  falsch, von Juli bis September nur in 3–6 %. Auf dem Trainingsteil (out-of-fold, hierarchisch)
  zeigt sich dasselbe Muster für Mai (25 %) und Oktober (13 %), für Juni schwächer (10 %).
- **Vermutung:** Am Saisonanfang ist wenig Bestand da, am Ende ist er abgereift oder geerntet –
  die Spektren verschiedener Kulturen werden sich dann ähnlicher.
- Die Klassifikatoren unterscheiden sich an den Rändern trotzdem: im Juni kombiniert 14,5 % vs.
  hierarchisch 19,6 %, im Mai umgekehrt 24,8 % vs. 20,0 %. Die Baseline fällt zusätzlich im
  August/September ab (9–11 %); dort holen die SVMs am meisten.

![Kulturfehler nach AEZ](img/fehleranalyse_aez.png)

- **Regionen:** Die Kulturfehler häufen sich in AEZ 9 (17 %, 428 Zeilen), 7 (14 %, nur 43 Zeilen)
  und 5 (12 %); in AEZ 8 und 10 sind es unter 4 %. Auf dem Trainingsteil ist das Muster gleich
  (AEZ 9: 16 %, AEZ 8/10: unter 4 %). Vermutlich bestimmt die Agrarökozone, welche Kulturen zur
  gleichen Zeit in ähnlichem Zustand sind.

## Die häufigsten Verwechslungen

Häufigste Kulturverwechslungen des hierarchischen Modells: Winterweizen → Baumwolle (26),
Mais → Soja (22), Soja → Mais (22), Mais → Winterweizen (16), Soja → Winterweizen (15).

### Winterweizen → Baumwolle

![Spektren Winterweizen und Baumwolle](img/fehleranalyse_spektren_weizen.png)

| Gruppe | Zeilen | NDVI (Median) | mittlere Reflektanz [%] |
| --- | --- | --- | --- |
| Weizen, als Baumwolle erkannt | 26 | 0.24 | 32.7 |
| Weizen, richtig erkannt (gleiche AEZ) | 283 | 0.22 | 27.2 |
| Baumwolle (gleiche AEZ) | 179 | 0.19 | 29.9 |

- 22 der 26 Fälle stammen aus **AEZ 5**, alle aus den Stadien **Mature_Senesc (15)** und
  **Emerge_VEarly (11)**, vor allem im Oktober (11) und Juni/Juli.
- Alle drei Gruppen haben einen niedrigen NDVI (0.19–0.24, wenig grüne Vegetation). Der NDVI
  erklärt den Fehler aber nicht: Die falsch erkannten Zeilen haben sogar einen etwas *höheren*
  NDVI als die richtig erkannten.
- Auffällig ist die **Helligkeit**: Die falsch erkannten Weizenspektren sind heller als beide
  Vergleichsgruppen und liegen damit näher an der Baumwolle als am übrigen Weizen.
- Das Spektrum trennt die Kulturen in diesen Stadien grundsätzlich: 261 der 283 richtig
  erkannten Weizenzeilen derselben AEZ sind in denselben Stadien. **Vermutung:** Die 26 Fälle
  sind besonders helle Felder (z. B. mehr offener oder trockenerer Boden), die in den Bereich der
  Baumwolle fallen.

### Mais ↔ Soja

![Spektren Mais und Soja](img/fehleranalyse_spektren_mais_soja.png)

| Gruppe | Zeilen | Kultur falsch (beliebig) | als andere Kultur des Paars |
| --- | --- | --- | --- |
| Mais, früh (Emerge_VEarly, Early_Mid) | 110 | 0.182 | 0.155 (Soja) |
| Mais, geschlossener Bestand (Critical, Late) | 369 | 0.019 | 0.003 (Soja) |
| Soja, früh | 171 | 0.181 | 0.064 (Mais) |
| Soja, geschlossener Bestand | 266 | 0.030 | 0.023 (Mais) |

- **Früh** ist die Kultur deutlich öfter falsch (18 %) als im geschlossenen Bestand (2–3 %).
- Die Verwechslung ist **nicht symmetrisch**: Früher Mais geht meist an Soja (15,5 %), früher
  Soja nur zu 6,4 % an Mais – die übrigen Fehler gehen an andere Kulturen.
- Spektral überlappen frühe Mais- und Sojaspektren in allen Bereichen stark (breite Streuung);
  im kurzwelligen Infrarot ab 2050 nm sind die Mittel praktisch gleich. Im geschlossenen Bestand
  ist der Abstand im nahen Infrarot ähnlich groß, die Streuung – vor allem beim Mais – aber viel
  kleiner, sodass sich die Kulturen besser trennen. **Vermutung:** Früh bestimmt der Boden
  zwischen den Reihen einen großen Teil des Signals.

## Folgerungen

Alle Punkte sind Hypothesen für weitere Experimente; ob sie helfen, entscheidet die CV.

- **Kulturfehler an den Saisonrändern sind der Hauptengpass** aller Ansätze – auf Validierung
  und Trainingsteil. Mögliche Ansätze: Kontext gezielter nutzen (Monat × AEZ), Merkmale für
  Helligkeit bzw. Bodenanteil, stärkere Gewichtung früher und später Stadien.
- **Kombiniert vs. hierarchisch** unterscheiden sich bei den Treffern nicht signifikant; der
  BAcc-Vorsprung des kombinierten Ansatzes kommt von wenigen kleinen Klassen und ist zusätzlich
  durch unterschiedliche Vorverarbeitung verzerrt. Für die Abgabe bleibt `combined_svm_expanded`
  vorn, die Begründung sollte diese Unsicherheit nennen.
- **Hierarchisch** profitiert direkt von einer besseren Kulturstufe (Oracle-Kultur 0.939), ein
  vergleichbarer Oracle-Wert für den kombinierten Ansatz fehlt aber – welcher Ansatz mehr
  gewinnen würde, ist offen.
