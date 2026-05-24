# `data/` — fichiers locaux uniquement

**Ne jamais committer** de PDF, données patient, ou échantillons même synthétiques.
Le contenu de ce dossier est gitignoré (cf. `.gitignore`) sauf `.gitkeep`, ce `README.md`
et `study_schema_default.json` (schéma d'exemple versionné, sans données patient).

## Comment utiliser ce dossier

- Placez vos PDF de test (CT scan, bilans, comptes rendus) ici.
- Référencez-les depuis `.env` :
  ```
  ECRF_CT_TEST_PDF_PATH=./data/ct_scan_report_liver.pdf
  ECRF_TEST_PDF_PATH=./data/votre_pdf.pdf
  ```
- Les tests d'intégration (`tests/test_ct_scan_report_integration.py`) **skipent
  proprement** si le fichier est absent — la CI ne sera pas bloquée.

## Conformité

- **HDS / RGPD** : aucun fichier nominatif ne doit transiter via Git, même privé.
- Pour partager un PDF de référence avec une autre machine, utilisez un canal
  approprié (stockage chiffré, partage interne), pas le dépôt.
