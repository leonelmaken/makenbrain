# MakenBrain

**Version:** `0.6.0`  
**Release:** Phase 1 Secure Core  
**Statut:** noyau local securise, pret pour durcissement produit

MakenBrain est un cerveau numerique personnel local-first. Il combine chat avec memoire, ingestion de connaissances, recherche web, graphe de concepts, agent fichiers securise, scheduler autonome controle, audit log et connecteurs IA.

## Vision

L'objectif est de construire un cerveau numerique personnel qui ne soit pas seulement un assistant, mais une plateforme capable de:

- apprendre depuis les donnees autorisees par l'utilisateur;
- raisonner avec une memoire vectorielle et un graphe de concepts;
- analyser, creer et modifier des fichiers dans une sandbox stricte;
- faire des recherches et maintenir une veille;
- documenter toutes les actions sensibles;
- evoluer vers des integrations mail, calendrier, projets, automatisation et comptes utilisateur plus tard.

## Etat Actuel

La Phase 1 est maintenant orientee securite et stabilite.

| Domaine | Statut | Details |
|---|---:|---|
| API FastAPI | OK | routes modulaires, Swagger, version centralisee |
| Memoire vectorielle | OK | ChromaDB + embeddings Sentence Transformers |
| LLM local | OK | Ollama configurable |
| Provider cloud | OK | Groq configure si cle disponible |
| Agent fichiers | OK | sandbox, permissions, dry-run, diff, backup, rollback |
| Ingestion | OK | fichiers, dossiers, uploads, ZIP securise |
| Recherche web | OK | DuckDuckGo + ingestion |
| Graphe de concepts | OK | NetworkX + exploration |
| Scheduler | OK | protege par API key |
| Audit log | OK | journal JSON centralise |
| Auth locale | OK | `X-API-Key` sur endpoints sensibles |
| CORS | OK | `ALLOWED_ORIGINS`, pas de wildcard |
| Rate limiting | OK | limite locale sur endpoints critiques |

## Versioning

La version applicative est definie dans:

```text
core/version.py
```

Regles:

- `MAJOR`: changement incompatible ou refonte majeure.
- `MINOR`: nouvelle capacite importante compatible.
- `PATCH`: correctif ou durcissement sans nouvelle surface majeure.

Version actuelle:

```text
0.6.0 - Phase 1 Secure Core
```

Avant chaque release:

1. Mettre a jour `APP_VERSION` dans `core/version.py`.
2. Mettre a jour `CHANGELOG.md`.
3. Mettre a jour le bloc version du `README.md`.
4. Executer les tests.

## Architecture

```text
makenbrain/
├── main.py                  # Point d'entree FastAPI
├── core/
│   ├── version.py           # Version applicative
│   ├── config.py            # Configuration centralisee
│   ├── auth.py              # Auth API key
│   ├── rate_limit.py        # Rate limiting local
│   ├── sandbox.py           # Sandbox fichiers
│   ├── permissions.py       # Permissions session
│   ├── audit.py             # Audit log JSON
│   ├── backups.py           # Backup / restore
│   ├── diff.py              # Diff dry-run
│   ├── memory.py            # Memoire ChromaDB
│   ├── llm.py               # Ollama
│   └── providers.py         # Groq / Wikipedia
├── routers/
│   ├── agent.py             # Agent fichiers securise
│   ├── files.py             # Ingestion fichiers/dossiers/uploads
│   ├── memory.py            # Memoire
│   ├── brain.py             # Expertise + scheduler
│   ├── audit.py             # Lecture audit log
│   ├── chat.py              # Chat RAG
│   ├── search.py            # Recherche web
│   └── graph.py             # Graphe de concepts
├── brain_data/              # Donnees locales generees
├── workspace/               # Racine autorisee par sandbox
├── projects/                # Racine autorisee par sandbox
├── uploads/                 # Racine autorisee par sandbox
└── tests/
```

## Securite Phase 1

### Authentification

Les endpoints sensibles exigent:

```http
X-API-Key: <ADMIN_API_KEY>
```

Variable:

```env
ADMIN_API_KEY=change-me
```

Si `ADMIN_API_KEY` est absent, les endpoints proteges refusent l'acces.

### CORS

Les origines autorisees viennent de:

```env
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
```

`allow_origins=["*"]` ne doit pas etre utilise.

### Rate limiting

Variables:

```env
RATE_LIMIT_MAX_REQUESTS=20
RATE_LIMIT_WINDOW_SECONDS=60
```

Le rate limiting protege les endpoints critiques via la dependance `SECURE`.

### Sandbox fichiers

Les chemins autorises sont uniquement:

```text
workspace/
projects/
uploads/
```

Interdit:

- `../`
- chemins systeme;
- acces disque hors sandbox;
- `.git`, `.venv`, `System32`, `Program Files`.

### Dry-run et application reelle

Les operations dangereuses suivent ce contrat:

```json
{
  "dry_run": true
}
```

Genere un diff sans ecrire.

Pour appliquer vraiment:

```json
{
  "dry_run": false,
  "apply_changes": true
}
```

## Installation

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env
```

Configure ensuite `.env`:

```env
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
ADMIN_API_KEY=change-me
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
RATE_LIMIT_MAX_REQUESTS=20
RATE_LIMIT_WINDOW_SECONDS=60
```

## Lancement

```bash
ollama serve
uvicorn main:app --reload --port 8000
```

Swagger:

```text
http://localhost:8000/docs
```

## Exemples API

### Chat RAG

```bash
curl -X POST http://localhost:8000/chat/ ^
  -H "Content-Type: application/json" ^
  -d "{\"message\":\"Resume mon projet principal\",\"use_memory\":true}"
```

### Ajouter une memoire protegee

```bash
curl -X POST http://localhost:8000/memory/add ^
  -H "Content-Type: application/json" ^
  -H "X-API-Key: %ADMIN_API_KEY%" ^
  -d "{\"content\":\"MakenBrain Phase 1 est securise.\",\"source\":\"readme\",\"tags\":\"makenbrain,phase1\"}"
```

### Dry-run agent fichier

```bash
curl -X POST http://localhost:8000/agent/write ^
  -H "Content-Type: application/json" ^
  -H "X-API-Key: %ADMIN_API_KEY%" ^
  -d "{\"file_path\":\"workspace/demo.txt\",\"content\":\"hello\",\"dry_run\":true}"
```

### Application reelle

```bash
curl -X POST http://localhost:8000/agent/write ^
  -H "Content-Type: application/json" ^
  -H "X-API-Key: %ADMIN_API_KEY%" ^
  -d "{\"file_path\":\"workspace/demo.txt\",\"content\":\"hello\",\"dry_run\":false,\"apply_changes\":true}"
```

### Lire l'audit log

```bash
curl -H "X-API-Key: %ADMIN_API_KEY%" http://localhost:8000/audit/log
```

## Tests

```bash
python -m pytest tests -q
```

Les tests Phase 1 couvrent:

- sandbox;
- permissions;
- backups / restore;
- agent dry-run et apply;
- auth API key;
- CORS configure;
- rate limiting.

## Roadmap

### Phase 1 - Secure Core

- [x] Configuration centralisee
- [x] Auth API key
- [x] CORS configure
- [x] Rate limiting
- [x] Sandbox fichiers
- [x] Permissions
- [x] Audit log
- [x] Backups / rollback
- [x] Dry-run / diff
- [x] Tests securite

### Phase 2 - Productisation

- [ ] Auth multi-utilisateur
- [ ] Rotation des secrets
- [ ] UI admin securite
- [ ] Gestion fine des roles
- [ ] Observabilite plus complete

### Phase 3 - Integrations

- [ ] Emails
- [ ] Calendrier
- [ ] Comptes externes
- [ ] Workflows projet
- [ ] Automatisation avancee

## Notes De Securite

Ne publie jamais `.env`.

Les cles `GROQ_API_KEY`, `HF_API_KEY` et `ADMIN_API_KEY` doivent etre considerees comme secrets. Si elles ont ete exposees dans un historique Git ou un partage, regenere-les.

---

**MAKEN x MakenBrain** - le cerveau grandit avec toi, mais sous controle.
