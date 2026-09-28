# Bewertung & Abgabe

## Metriken

Primär zählt die **Balanced Accuracy** auf dem (ungelabelten) Validierungsdatensatz, den wir
gegen Ende erhalten. Zusätzlich:

$$
\text{BAcc} = \frac{1}{K}\sum_{c=1}^{K}\frac{TP_c}{TP_c + FN_c}
\qquad
F1_{\text{macro}} = \frac{1}{K}\sum_{c=1}^{K} F1_c
\qquad
F1_{\text{samples}} = \frac{1}{n}\sum_{i=1}^{n} F1_i
$$

Samples-F1 wird pro Beobachtung berechnet (Multi-Label: Crop + Stage). Leistung jeweils für
Crops **und** Stages betrachten; Confusion Matrix und Precision/Recall pro Klasse für die
Fehleranalyse.

Den Train-/Test-Split wählen wir selbst.

## Bewertungskriterien

- **Modellleistung** (Balanced Accuracy auf Validierungsdaten)
- **Trade-off** Datenreduktion vs. Accuracy
- Bonus: Feature-Importance-Analyse
- **Fachlich**: Gesamtprozess, richtige Anwendung der Algorithmen, Preprocessing & Tuning
- **Technisch**: Struktur, Dokumentation, Verständlichkeit, Robustheit des Codes
- **Dokumentation**: begründete Entscheidungen, vollständiger Abschlussbericht
- **Kommunikation**: Statusmeetings, Meilenstein-Vorstellungen, Abschlusspräsentation
- **Organisation**: Fortschritt zu den Zwischenterminen, Aufgabenteilung

## Abgabe-Checkliste

- [ ] Code – reproduzierbar (README, Umgebung via `uv.lock`)
- [ ] Abschlussbericht (knapp & klar)
- [ ] Modellkarte (Annahmen, Grenzen)
- [ ] Interpretation der Ergebnisse
- [ ] CSV mit Vorhersagen auf dem Validierungsdatensatz
- [ ] Je Teammitglied: Dokumentation der täglichen Arbeitszeit (ILIAS-Template)
- [ ] Je Teammitglied: 0,5–1 Seite eigene Beiträge
