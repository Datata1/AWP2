# data/

In git versioniert sind nur die Ordnerstruktur und die Rohdaten in `raw/`.

Rohdaten im Code immer über `from awp2.data import load_train, load_test` laden – die Funktionen
prüfen das Format (pandera-Schema in `src/awp2/data/schema.py`). Pfade aus `awp2.config`, nie hardcoden.

| Ordner | Was gehört rein? | Wer schreibt? |
| --- | --- | --- |
| `raw/` | Originaldaten, genau so wie geliefert. **Nie verändern.** | Niemand, liegen im Repo |
| `interim/` | Bereinigter Datensatz (`train_clean.parquet`) | `make data` |
| `processed/` | Split-Zuordnung für alle (`split.csv`: train/val, CV-Fold) | `make data` |
| `external/` | Fremde Referenzdaten, z. B. Sentinel-2-SRF der ESA (Version per Prüfsumme fixiert) | `make external` |
| `assets/` | Aufgabenstellung, Orga-Unterlagen (aus ILIAS) – nicht in git, da das Repo öffentlich ist | Du, per Hand |

Fluss: `raw/` → Pipeline → `interim/` → Pipeline → `processed/` → Modelle

Welche Rohdateien erwartet werden: siehe Doku unter *Daten* (`docs/daten/`).
Alles in `interim/` und `processed/` muss sich jederzeit aus `raw/` neu erzeugen lassen.
