# Backlog pipeline eCRF Autofill

Table de suivi des briques **non implémentées**, **partielles** ou **à renforcer** par rapport au chemin principal `run_pipeline()`.

**Légende priorité**

| Niveau | Signification |
|--------|----------------|
| **P0 — Critique** | Bloque un usage prod ou la conformité |
| **P1 — Haute** | Nécessaire pour une démo / intégration métier crédible |
| **P2 — Moyenne** | Améliore qualité, ops ou couverture fonctionnelle |
| **P3 — Basse** | Optionnel, dette technique ou exploration |

---

## Table des tâches

| Titre | Description | Impact | Priorité |
|-------|-------------|--------|----------|
| Adaptateur MA_Base → StudySchema | Générer `study_schema_<study_id>.json` depuis `data/MA_Base_example.xlsx` (mapping colonnes, familles, seuils, requêtes RAG). | Réduit la saisie manuelle du schéma et aligne le pipeline sur la base métier réelle. | **P0** |
| API HTTP pipeline | Exposer `POST /pipeline/run` (FastAPI) via `app/api/` : upload document, paramètres `patient_id` / `study_id`, retour `PipelineResult` ou job async. | Aujourd’hui usage script-only ; bloque intégration front / worker / orchestrateur externe. | **P1** |
| Pseudonymisation HDS pré-indexation | Hooks avant `upsert_chunks` pour détecter / masquer identifiants nominatifs (IPP, nom, date naissance) dans texte et métadonnées. | Risque conformité RGPD/HDS si documents réels indexés sans anonymisation. | **P0** |
| CLI réindexation | Outil CLI pour rejouer l’indexation d’un tenant (`study_id`) après changement de `embedding_version` ou de modèle dense/sparse. | Sans réindexation, changement de modèle = retrieval dégradé ou incohérent. | **P1** |
| Agrégation multi-documents patient | Fusionner plusieurs runs (labo + imagerie + courrier) en une vue patient : dédup colonnes, résolution conflits par score / temporalité. | Un run = un document ; cas réel = plusieurs sources par patient et visite. | **P1** |
| Intégration RAGFlow | Implémenter `RagflowWorkflowOrchestrator` : appels HTTP/SDK, mapping réponse → `RetrievalHit`. | Alternative retrieval externe ; utile si RAGFlow devient le moteur RAG de référence. | **P2** |
| Llama fine-tuné labo | Remplacer le stub `LlamaExtractor` (heuristiques) par inférence réelle (llama.cpp / HF / vLLM) et ajouter une stratégie `ExtractionStrategy` dédiée dans `ExtractorRegistry`. | Améliore recall labo sur bilans non structurés ; aujourd’hui le registre n’appelle jamais Llama. | **P2** |
| LangExtract pour le labo | Brancher LangExtract (ou LLM court) sur `LabDeterministicStrategy` pour analytes absents du catalogue regex. | Complète les heuristiques sur formats labo hétérogènes. | **P2** |
| Couverture `pathology_report` | Champs + stratégies d’extraction pour comptes rendus anatomopathologiques (grading, marges, biomarqueurs tissulaires). | Type document routable mais sans schéma ni extracteur dédié en V1. | **P2** |
| Couverture `treatment_lines` | Définir champs `FieldFamily.TREATMENT_LINES`, requêtes RAG et stratégie narratif / structuré pour lignes thérapeutiques (L2, L3…). | Enum et requêtes retrieval existent ; aucun champ dans `study_schema_default.json`. | **P2** |
| Validation métier eCRF | Enrichir `ValidationService` : plages cliniques, cohérence inter-champs, règles RECIST, messages d’erreur exploitables par un contrôleur. | Validation actuelle = types numériques/booléens uniquement. | **P1** |
| Workflow relecture humaine | Flag `validated`, API ou export « à revue », statuts brouillon / confirmé / rejeté avec traçabilité audit. | Pas de boucle clinique avant écriture eCRF définitive. | **P1** |
| Ingestion distante | Sources au-delà du disque local : S3, partage réseau, webhook upload, connecteur DICOM / HL7 (selon périmètre). | Limite le déploiement à des fichiers locaux ou montés manuellement. | **P2** |
| OpenTelemetry | Traces et métriques (latence parsing, embeddings, Qdrant, retrieval, LangExtract) + corrélation `document_id`. | Debug prod et SLA difficiles sans observabilité. | **P2** |
| LlamaIndex sur Qdrant | Brancher `VectorStoreIndex` + `QdrantVectorStore` avec filtres metadata ; option alternative à `QdrantHybridVectorService`. | Extra pip et stub mémoire existent ; non utilisés par `build_default_vector_service()`. | **P3** |
| Reranker activé par défaut en prod | Documenter et calibrer `ECRF_ENABLE_RERANKER=true` + seuils ; mesurer gain retrieval imagerie/labo. | Implémenté mais souvent off (CI, dev) ; gain qualité RAG non exploité. | **P2** |
| Tagging `field_family` au chunking | Option B rejetée en V1 : tagger les chunks à la création selon `document_type` / sections pour réduire le filtre OR null en Qdrant. | Améliore précision retrieval multi-familles à grande échelle. | **P3** |
| Ranking retrieval par hit (labo) | Utiliser le score de chaque `RetrievalHit` dans le blend confiance au lieu du max par famille uniquement. | Meilleure discrimination quand plusieurs chunks labo sont pertinents. | **P2** |
| Documentation pre-commit Windows | Section dédiée dans `useful_commands.md` : PATH venv, mypy, encodage UTF-8. | Friction onboarding développeurs Windows. | **P3** |
| CI : migration actions Node 24 | Mettre à jour `actions/checkout`, `actions/setup-python`, `actions/upload-artifact` avant juin 2026. | Warning GitHub Actions ; risque de rupture runner. | **P3** |

---

## Déjà implémenté (référence)

Pour éviter les doublons, le chemin suivant est **opérationnel** en V1 :

- Ingestion locale → parsing (pypdf / Docling) → chunking → routage
- Index `memory` ou Qdrant hybride (dense + BM25 + rerank optionnel)
- Extraction pilotée `StudySchema` : `lab_deterministic`, `narrative_keywords`, `imaging_langextract` (Ollama)
- Règles métier : temporalité, mapping, normalisation, score confiance, validation basique
- Export JSON + CSV mock + XLS réel (résolution patient, alias colonnes, politiques overwrite)
- Templates JSON d'évaluation imbriqués pour les documents labo et imagerie
- Évaluation offline : présence, valeur, contrat et Recall@K silver

---

*Dernière mise à jour : août 2026 — aligné sur la branche `internship-handover`.*
