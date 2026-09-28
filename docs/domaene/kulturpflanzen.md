# Kulturpflanzen & Phänologie

## Die fünf Kulturen

!!! todo "Leitfragen (je Kultur: Mais, Sojabohne, Winterweizen, Baumwolle, Reis)"
    - Wuchsform und Bestandsstruktur (Reihenabstand, Höhe, Blattform), C3- oder C4-Pflanze?
    - Anbausaison: Aussaat, Hauptwachstum, Ernte (in welchen Monaten)?
    - Wo wird sie typischerweise angebaut (Klima, Bewässerung, z. B. Reis geflutet)?
    - Was macht sie spektral besonders? Mit welcher Kultur könnte sie verwechselt werden?

| Kultur | Typ | Aussaat | Ernte | Besonderheit |
| --- | --- | --- | --- | --- |
| Mais | | | | |
| Sojabohne | | | | |
| Winterweizen | | | | |
| Baumwolle | | | | |
| Reis | | | | |

## Phänologie & Entwicklungsstadien

!!! todo "Leitfragen"
    - Was ist Phänologie? Was ist die **BBCH-Skala**?
    - Welche kulturspezifischen Skalen gibt es (Mais und Soja: V-/R-Stadien,
      Weizen: Zadoks)?

## Die Stadien-Labels im Datensatz

!!! todo "Leitfragen"
    - Was bedeuten die sechs Labels genau – idealerweise mit Definition aus der Originalquelle
      (siehe [Daten → Herkunft](../daten/index.md#herkunft-aez))?
    - Was ist mit **„Critical"** gemeint (z. B. Blüte / reproduktive Phase)?
    - Wie lassen sie sich je Kultur auf BBCH bzw. V/R-Stadien abbilden?

| Label | Bedeutung | Mais | Soja | Winterweizen | Baumwolle | Reis |
| --- | --- | --- | --- | --- | --- | --- |
| `Emerge_VEarly` | | | | | | |
| `Early_Mid` | | | | | | |
| `Critical` | | | | | | |
| `Late` | | | | | | |
| `Mature_Senesc` | | | | | | |
| `Harvest` | | | | | | |

## Zusammenhänge

!!! todo "Leitfragen"
    - Welche Kombinationen aus Kultur und Stadium kommen im Datensatz nicht vor – und ist das
      fachlich plausibel? (Ausgangspunkt: Kreuztabelle in [Daten](../daten/index.md))
    - Wie hängen Aufnahmemonat und Stadium zusammen? Warum verhält sich Winterweizen anders?
    - Was bedeutet das für die Modellierung (Hierarchie Kultur → Stadium)?
