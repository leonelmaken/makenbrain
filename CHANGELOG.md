# Changelog

Toutes les versions suivent le format semver `MAJOR.MINOR.PATCH`.

## 0.8.0 - Phase 2.8 Trustworthy Brain

Date: 2026-06-21

### Ajoute

- Systeme de confidence score dans `core/response_confidence.py`.
- Auto-critique deterministe dans `core/self_critic.py`.
- Routage vers sources externes suggerees dans `core/source_router.py`.
- Integration de l'evaluation de reponse dans `core/consciousness.py`.
- Champs API compatibles: `confidence`, `risk_level`, `suggested_sources`.
- Tests unitaires Phase 2.8 pour confiance, auto-critique, sources, consciousness et exposition API.

### Change

- Version applicative mise a jour vers `0.8.0`.
- README restructure autour du statut reel du cerveau: ce qui marche, ce qui est en cours, ce qui manque.
- `/chat/` et `/brain/expert-chat` exposent les metadonnees de confiance sans retirer les champs historiques.

### Notes

- Aucun test ni serveur n'a ete lance pendant la mise a jour documentaire.
- Les suggestions de sources restent declaratives: elles ne declenchent pas encore de verification reseau automatique.
- Un cycle statique global preexistant reste a traiter hors Phase 2.8: `routers.files -> core.watcher -> routers.files`.

## 0.6.0 - Phase 1 Secure Core

Date: 2026-06-20

### Ajoute

- Authentification locale par `X-API-Key` sur les endpoints sensibles.
- Configuration centralisee dans `core/config.py`.
- CORS pilote par `ALLOWED_ORIGINS`, sans wildcard.
- Rate limiting local pour endpoints critiques.
- Sandbox stricte limitee a `workspace/`, `projects/`, `uploads/`.
- Audit log JSON centralise.
- Backups, rollback, dry-run et diff avant modifications.
- Tests de securite Phase 1.

### Change

- Version FastAPI centralisee via `core/version.py`.
- Agent fichiers et ingestion passent par permissions/sandbox.
- Endpoints critiques exigent `ADMIN_API_KEY`.

### Notes

- Cette version reste local-first.
- Avant exposition reseau, ajouter une gestion multi-utilisateurs et une rotation des secrets.
