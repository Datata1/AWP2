# Kombinierte Klasse

## Idee

Kultur und Stadium werden zu *einer* Klasse `Crop|Stage` verbunden – 23 Klassen im Datensatz.
Ein beliebiger Klassifikator lernt damit direkt die gemeinsame Verteilung P(Crop, Stage).
Baustein und Recherche: #54, Entwurf #55.

## Vor- und Nachteile

| Vorteil | Nachteil |
| --- | --- |
| Sagt nur Paare vorher, die es gibt | Nutzt nicht, dass das Stadium von der Kultur abhängt: P(Crop, Stage) = P(Crop) · P(Stage \| Crop) wird als 23 unabhängige Klassen gelernt |
| Funktioniert mit jedem Klassifikator | Kein geteiltes Wissen zwischen z. B. Mais/Late und Soja/Late |
| Ein Modell, einfach zu vergleichen | Wenige Daten je Klasse (Baumwolle/Harvest: 11 Zeilen) |
| | Jeder Kultur-Fehler ist automatisch auch ein Stadium-Fehler |

!!! todo "Recherche (#54, Grundlagen in #11)"
    - Wird der Ansatz in der Literatur für hierarchische Labels genutzt, und mit welchem Ergebnis?
    - Wie verhält er sich gegenüber hierarchischer Klassifikation bei so seltenen Klassen?
    - Quellen in [Quellen](../../domaene/quellen.md) eintragen.

## Klassifikatoren

_Noch keine._ Neue Klassifikatoren als Unterkapitel nach der [Vorlage](index.md#aufbau-einer-ansatz-seite).

## Fazit

_Offen._
