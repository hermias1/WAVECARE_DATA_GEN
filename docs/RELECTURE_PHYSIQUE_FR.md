# Relecture Physique (FR)

Ce fichier est une porte d'entrée rapide pour une relecture "physique" du repo.

## À lire en priorité

1. `docs/PHYSICS_REVIEW_GUIDE.md`
2. `docs/PHYSICS_MODEL_AND_ASSUMPTIONS.md`
3. `docs/PHYSICS_VALIDATION_CHECKLIST.md`

## Points critiques à challenger

- La grandeur simulée est basée sur `Ez` (source dipôle de Hertz + sonde ponctuelle),
  ce n'est pas un `S21` port-à-port complet d'antenne/VNA.
- Le placement des antennes doit rester hors tissu: le repo bloque désormais les
  géométries invalides avec suggestion de rayon.
- La résolution spatiale (`dx`) doit être cohérente avec `f_max` dans les tissus
  à forte permittivité.
- Les hypothèses (Debye 1 pôle, calibration par soustraction, bruit synthétique)
  doivent être explicitement acceptées pour l'usage visé.

## Commande de départ conseillée

```bash
python3 scripts/generate_scan.py \
  --phantom 071904 \
  --preset umbmid \
  --radius-cm 12 \
  --min-clearance-mm 3 \
  --tumor-mm 12
```

