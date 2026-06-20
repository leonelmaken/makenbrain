# Changelog

Toutes les versions suivent le format semver `MAJOR.MINOR.PATCH`.

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
