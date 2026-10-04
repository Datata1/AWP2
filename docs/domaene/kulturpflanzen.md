# Kulturpflanzen & Phänologie

## Die fünf Kulturen

### Was ist eine Kulturpflanze?

Eine **Kulturpflanze** ist eine Pflanze, die Menschen gezielt anbauen und bewirtschaften, um
sie zu nutzen, zum Beispiel als Nahrungsmittel, Tierfutter oder Faserrohstoff. Eine Kulturart
ist dabei die Pflanzenart, die auf einer Fläche angebaut wird. Im Projekt ist `Crop` das Label
für diese Kulturart.

Für die Fernerkundung ist wichtig: Eine Kultur ist keine unveränderliche Oberfläche. Während
der Wachstumszeit ändern sich unter anderem Bodenbedeckung, Pflanzenhöhe und Blattfläche.[^1]
Deshalb kann dieselbe Kultur in verschiedenen Entwicklungsstadien unterschiedliche Spektren
haben. Ob sich die fünf Kulturen in unserem Datensatz tatsächlich gut trennen lassen, prüfen
wir später in der EDA und nicht allein anhand allgemeiner Pflanzenkenntnisse.

### Die fünf Crop-Labels im Datensatz

Die fünf Werte von `Crop` sind `corn` (Mais), `soybean` (Sojabohne), `winter_wheat`
(Winterweizen), `cotton` (Baumwolle) und `rice` (Reis). Die Labels benennen die Kulturart,
nicht ihr Entwicklungsstadium.

### Warum keine festen Anbaumonate?

Aussaat, Hauptwachstum und Ernte hängen nicht nur von der Kulturart ab, sondern auch von
Region, Klima, Sorte und Bewirtschaftung. Laut Karte in den Kickoff-Folien liegt das
Untersuchungsgebiet in Nordamerika. Die Werte 5 bis 10 in `Month` fallen damit in die Monate
Mai bis Oktober auf der Nordhalbkugel.[^2] Welche Regionen und Jahre genau vertreten sind,
ist jedoch noch offen. Deshalb tragen wir hier keine kulturspezifischen Anbaukalender ein.
Die Spalte `Month` ist ein wichtiger Hinweis für die EDA, aber kein Ersatz für eine bestätigte
Anbausaison.

!!! question "Noch offen für die fünf Kulturen"
  Die Originalquelle soll klären, aus welchen Regionen und Jahren die Beobachtungen stammen
  und wie die Kultur- und Stadien-Labels vergeben wurden. Erst dann können wir regionale
  Anbaukalender, typische Bewirtschaftung und kulturspezifische Spektralhypothesen sinnvoll
  mit dem Datensatz verbinden.

[^1]: Allen et al. (1998), siehe [Quellen](quellen.md).
[^2]: Domänenprojekt 2 (2026), siehe [Quellen](quellen.md).

## Phänologie & Entwicklungsstadien

### Was ist Phänologie?

**Phänologie** beschreibt die zeitliche Abfolge wiederkehrender Entwicklungsphasen einer
Pflanze. Bei einer einjährigen Kultur beginnt dieser Verlauf mit Keimung und Auflaufen, führt
über Blatt- und Bestandsentwicklung sowie die Fortpflanzungsphase zur Reife, Seneszenz und
Ernte. Wann eine Phase eintritt, hängt unter anderem von Kulturart, Sorte, Klima und
Bewirtschaftung ab.

Für die Fernerkundung ist dieser Verlauf wichtig, weil sich mit der Entwicklung Bodenbedeckung,
Pflanzenhöhe und Blattfläche verändern.[^3] Eine junge Kultur mit viel sichtbarem Boden kann
deshalb ein anderes Spektrum haben als dieselbe Kultur mit dichtem, grünem Bestand oder in
der Seneszenz. Das ist eine fachliche Erwartung, die wir später mit den Daten prüfen.

### Entwicklungsstadien als gemeinsame Sprache

Entwicklungsstadien machen beobachtbare Veränderungen vergleichbar. Die **BBCH-Skala** ist
eine standardisierte Dezimalskala, die solche Phasen mit zweistelligen Codes beschreibt.[^4]
Sie wird für viele Pflanzenarten verwendet, ersetzt aber nicht alle kulturspezifischen Skalen.

Für Mais und Soja werden beispielsweise häufig vegetative **V-Stadien** und reproduktive
**R-Stadien** unterschieden. Für Getreide wie Winterweizen ist die Zadoks-Skala verbreitet.
Diese Skalen sind detaillierter als unsere allgemeinen Datensatzlabels und lassen sich nicht
ohne eine bestätigte Definition eins zu eins übertragen.

!!! example "Allgemeiner Verlauf, keine Label-Zuordnung"
  Zu Beginn ist oft viel Boden sichtbar. Während der Entwicklung nimmt die grüne
  Bodenbedeckung zu. Gegen Ende reifen Pflanzen aus; Blätter können altern und ihre Farbe
  verändern. Diese Beschreibung erklärt den allgemeinen Verlauf, ordnet aber noch keines
  der Labels `Emerge_VEarly`, `Early_Mid` oder `Critical` exakt zu.

### Bedeutung für unser Projekt

`Stage` ist kein zweiter Name für `Crop`: Die Kulturart sagt, **was** wächst; das
Entwicklungsstadium sagt, **wo im saisonalen Verlauf** diese Kultur beobachtet wurde. Die
allgemeinen Stadien können sich zwischen Kulturen unterschiedlich äußern. Deshalb prüfen wir
später Spektren innerhalb jeder Kultur nach Stadium, statt alle Kulturen im selben Stadium als
fachlich gleich anzunehmen.

[^3]: Allen et al. (1998), siehe [Quellen](quellen.md).
[^4]: Meier (2018), siehe [Quellen](quellen.md).

## Die Stadien-Labels im Datensatz

`Stage` enthält genau sechs Kategorien: `Emerge_VEarly`, `Early_Mid`, `Critical`, `Late`,
`Mature_Senesc` und `Harvest`. Sie sind Textlabels, keine BBCH- oder Zadoks-Codes. Die
Reihenfolge, in der sie in einer Liste stehen, ist daher kein Nachweis für eine zeitliche
Abfolge oder für gleich große Zeitabschnitte.

### Verteilung im Trainingsdatensatz

Die folgende Häufigkeit ist ein gemessener Befund aus den 5.591 Zeilen von `load_train()`:

| Stage-Label | Beobachtungen |
| --- | ---: |
| `Emerge_VEarly` | 1.038 |
| `Early_Mid` | 982 |
| `Critical` | 1.541 |
| `Late` | 861 |
| `Mature_Senesc` | 989 |
| `Harvest` | 180 |

`Harvest` ist damit deutlich seltener als die anderen Stadien. Bei einer späteren Bewertung
ist das wichtig: Eine hohe Gesamtquote könnte diese seltene Klasse verdecken. Deshalb verwendet
das Projekt Balanced Accuracy, die jede Klasse gleich gewichtet.

### Was die Namen nicht belegen

Einige Namen enthalten verständliche englische Begriffe wie `Harvest` (Ernte) oder
`Mature_Senesc` (Reife und Seneszenz). Daraus lässt sich aber nicht ableiten, nach welchem
sichtbaren Merkmal, Datum oder BBCH-Code die Kategorien vergeben wurden. Besonders `Critical`
ist ohne Definition mehrdeutig: Es könnte eine empfindliche Phase meinen, muss aber nicht
Blüte oder ein bestimmtes reproduktives Stadium bezeichnen.

Auch `Month` liefert keine allgemeine Auflösung: Im Trainingsdatensatz liegt der Median für
`Emerge_VEarly` bei Monat 9, für `Early_Mid` dagegen bei Monat 7. Das widerspricht einer
einfachen gemeinsamen Kalenderreihenfolge. Unterschiede zwischen Kulturarten, Regionen,
Anbausaisons oder der Labelvergabe können dieses Muster verursachen; die Daten allein
unterscheiden diese Erklärungen nicht.

!!! question "Für eine fachliche Zuordnung fehlt noch eine Quelle"
    Die Originalquelle soll für jedes Label die beobachtbaren Kriterien und die Abbildung auf
    kulturspezifische Skalen nennen. Bis dahin behandeln wir `Stage` als sechs fest definierte
    Datenkategorien, nicht als bestätigte BBCH-, V/R- oder Zadoks-Stadien.

## Zusammenhänge

### Kultur und Stadium gehören zusammen

Nicht jede Kombination aus `Crop` und `Stage` kommt im Trainingsdatensatz vor. Reis ist nur
mit `Early_Mid` und `Late` vertreten; Baumwolle hat kein `Late`, Winterweizen kein `Early_Mid`
und kein `Harvest`. Die vollständige Kreuztabelle steht bei den [Daten](../daten/index.md).

Das ist zunächst ein Befund über die Stichprobe, keine Aussage darüber, welche Kombinationen
in der Landwirtschaft unmöglich sind. Es kann sowohl die tatsächliche Anbausaison als auch
die Auswahl der Aufnahmen widerspiegeln. Für die Klassifikation ist es dennoch eine feste
Randbedingung: Ein getrennt vorhergesagtes Paar wie `rice` und `Harvest` ist im Training nicht
beobachtet worden. Spätere Modelle müssen deshalb gültige Crop-Stage-Kombinationen beachten.

### Aufnahmemonat und Stadium

Die Monate liefern innerhalb einiger Kulturen ein plausibles zeitliches Muster. Bei Mais und
Soja liegen `Emerge_VEarly` und `Early_Mid` überwiegend zwischen Mai und Juli, während
`Harvest` nur im September und Oktober vorkommt. Für Mais erscheint `Late` ausschließlich im
Juli; `Critical` liegt zwischen Juli und September. Das zeigt eine starke Verbindung zwischen
`Month` und den Labels, beweist aber keine genaue fachliche Definition der Stadien.

Winterweizen verhält sich im Datensatz anders: `Emerge_VEarly` tritt nur im September und
Oktober auf, `Mature_Senesc` dagegen von Mai bis Juli. Eine mögliche Erklärung ist eine
Kulturperiode, die über den Jahreswechsel verläuft. Das bleibt eine Hypothese, denn weder
Jahr, Region noch die Regeln zur Labelvergabe sind bekannt. Reis ist ausschließlich im August
beobachtet; aus 93 Beobachtungen mit nur zwei Stage-Labels lässt sich kein allgemeiner
Saisonverlauf ableiten.

### Folgen für EDA und Modellierung

`Month` kann echte Kontextinformation enthalten, aber auch ein **Shortcut** sein: Ein Modell
könnte ein Stadium vor allem aus dem Aufnahmemonat lernen, statt aus dem Spektrum. Das wäre
fragil, wenn sich Regionen, Jahre oder Aufnahmemonate in zukünftigen Daten ändern. Deshalb
vergleichen wir später transparent Modelle mit und ohne Metadaten und prüfen ihre Fehler nach
Kultur, Stadium und Monat.

Die ungleichen Klassenhäufigkeiten und fehlenden Kombinationen verlangen außerdem eine
stratifizierte Aufteilung nach Crop-Stage-Paaren sowie klassenfaire Metriken. Eine gute
Vorhersage für die häufige Kombination `corn|Critical` darf die seltenen oder nicht
beobachteten Paare nicht verdecken.
