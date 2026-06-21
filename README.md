# MakenBrain

**Version:** `0.8.0`  
**Release:** Phase 2.8 - Trustworthy Brain  
**Statut:** assistant personnel IA avec auth Supabase, memoire utilisateur, conscience, confiance et auto-critique  
**Derniere mise a jour:** 2026-06-21

MakenBrain est un cerveau numerique personnel construit avec FastAPI. Il combine chat securise, memoire utilisateur, memoire vectorielle, graphe de concepts, couche de conscience, agent fichiers sandboxe, recherche, scheduler autonome et fournisseurs IA locaux/cloud.

L'objectif produit est simple: donner a MAKEN un assistant intelligent, contextuel et transparent, capable de raisonner avec ses propres donnees sans inventer quand l'information manque.

## Vision

MakenBrain doit devenir un assistant personnel de production capable de:

- comprendre le contexte personnel et projet de l'utilisateur;
- repondre avec memoire, graphe et logique IA;
- evaluer sa propre confiance avant d'exposer une reponse;
- signaler les incertitudes au lieu de repondre de maniere categorique;
- proposer des sources externes quand ses donnees internes sont insuffisantes;
- agir sur les fichiers uniquement dans une sandbox controlee;
- tracer les actions sensibles et rester robuste face aux erreurs.

## Versioning Du Cerveau

La version applicative officielle est centralisee dans:

```text
core/version.py
```

Version actuelle:

```text
0.8.0 - Phase 2.8 Trustworthy Brain
```

Regles semver:

- `MAJOR`: changement incompatible, refonte d'architecture ou migration majeure.
- `MINOR`: nouvelle capacite compatible ajoutee au cerveau.
- `PATCH`: correctif, durcissement, documentation ou tests sans nouvelle surface majeure.

Avant chaque release:

1. Mettre a jour `APP_VERSION` et `APP_RELEASE_NAME` dans `core/version.py`.
2. Ajouter une entree dans `CHANGELOG.md`.
3. Mettre a jour le bloc version du `README.md`.
4. Ajouter ou mettre a jour les tests pertinents.
5. Executer les tests manuellement avant merge/release.

## Etat Actuel

### Ce Qui Marche

| Domaine | Statut | Details |
|---|---:|---|
| API FastAPI | OK | routes modulaires, Swagger, version centralisee |
| Auth Supabase | OK | Bearer token via Supabase Auth, synchronisation profil applicatif |
| UserService | OK | CRUD utilisateur via client Supabase admin |
| User memories | OK | DTO Pydantic propres + service Supabase isole par `user_id` |
| Chat securise | OK | utilisateur authentifie injecte dans le contexte |
| Memoire vectorielle | OK | ChromaDB + embeddings Sentence Transformers |
| Conscience | OK | contexte memoire utilisateur + evaluation des reponses |
| Confidence score | OK | score `0.0 -> 1.0`, `risk_level`, penalite si donnees absentes |
| Self critic | OK | detection incertitude, contradictions simples, manque de sources |
| Source router | OK | suggestions web, documentation officielle, YouTube, X |
| Brain expert chat | OK | enrichi avec `confidence`, `risk_level`, `suggested_sources` |
| Agent fichiers | OK | sandbox, permissions, dry-run, diff, backup, rollback |
| Ingestion | OK | fichiers, dossiers, uploads, ZIP securise |
| Recherche web | OK | DuckDuckGo + ingestion |
| Graphe de concepts | OK | NetworkX + exploration |
| Scheduler | OK | exploration autonome et rapports |
| Audit log | OK | journal JSON centralise |
| CORS | OK | origines configurees, pas de wildcard |
| Rate limiting | OK | limite locale sur endpoints critiques |

### En Cours

| Sujet | Statut | Notes |
|---|---:|---|
| Phase 2.8 tests | Pret | tests unitaires ajoutes, execution manuelle non lancee dans cette session |
| Streaming chat | A surveiller | envoie une evaluation finale SSE avec `confidence`, `risk_level`, `suggested_sources` |
| Documentation architecture | En cours | README mis a jour, docs plus fines a ajouter par module si besoin |
| Nettoyage imports globaux | A surveiller | cycle statique preexistant detecte: `routers.files -> core.watcher -> routers.files` |
| UX des sources suggerees | A faire cote client | l'API propose les sources, l'interface doit les afficher proprement |

### Ce Qui Manque

| Sujet | Priorite | Details |
|---|---:|---|
| Execution officielle des tests Phase 2.8 | Haute | lancer pytest manuellement avant release |
| Observabilite production | Haute | metriques, traces, logs structures, correlation request/user |
| Gestion fine des roles | Moyenne | roles applicatifs plus stricts au-dessus de Supabase Auth |
| Rotation des secrets | Moyenne | procedure et outillage pour cles exposees/expirees |
| Fetch externe automatique | Moyenne | le source router propose, mais ne verifie pas encore les sources en live |
| UI admin securite | Moyenne | monitoring, audit, permissions, statut services |
| Integrations personnelles | Basse | email, calendrier, comptes externes, workflows projet |

## Architecture

Regles de separation:

- `models/`: DTO Pydantic uniquement, zero logique metier.
- `core/`: logique metier, services, orchestration, fournisseurs.
- `routers/`: API FastAPI uniquement, dependances et serialization HTTP.

```text
makenbrain/
|-- main.py
|-- core/
|   |-- version.py
|   |-- config.py
|   |-- auth.py
|   |-- supabase_client.py
|   |-- user_service.py
|   |-- user_memory_service.py
|   |-- consciousness.py
|   |-- response_confidence.py
|   |-- self_critic.py
|   |-- source_router.py
|   |-- memory.py
|   |-- memory_router.py
|   |-- llm.py
|   |-- providers.py
|   |-- graph.py
|   |-- sandbox.py
|   |-- permissions.py
|   |-- audit.py
|   |-- backups.py
|   `-- scheduler.py
|-- models/
|   |-- user.py
|   |-- user_memory.py
|   `-- schemas.py
|-- routers/
|   |-- chat.py
|   |-- brain.py
|   |-- users.py
|   |-- memory.py
|   |-- files.py
|   |-- agent.py
|   |-- audit.py
|   |-- search.py
|   `-- graph.py
|-- brain_data/
|-- workspace/
|-- projects/
|-- uploads/
`-- tests/
```

## Phase 2.8 - Trustworthy Brain

La Phase 2.8 ajoute une couche de fiabilite au cerveau.

### Confidence Score

Module:

```text
core/response_confidence.py
```

Produit:

```json
{
  "answer": "...",
  "confidence": 0.82,
  "risk_level": "low"
}
```

Facteurs:

- memoire utilisateur: poids positif fort;
- donnees Supabase: poids positif;
- memoire vectorielle et graphe: poids positif modere;
- logique IA seule: neutre;
- absence de donnees: penalite;
- incertitude detectee: penalite.

### Self Critic

Module:

```text
core/self_critic.py
```

Role:

- detecter les formulations incertaines;
- signaler les contradictions simples;
- ajouter une formulation prudente si `confidence < 0.60`;
- eviter les reponses trop categoriques sans contexte.

### Source Router

Module:

```text
core/source_router.py
```

Role:

- proposer des sources externes quand la confiance est faible;
- recommander documentation officielle, web, YouTube ou X selon le besoin;
- ne pas pretendre qu'une verification externe a deja ete faite.

Exemple:

```json
{
  "answer": "Avec les donnees actuellement disponibles...",
  "confidence": 0.45,
  "risk_level": "high",
  "suggested_sources": [
    {
      "type": "web",
      "query": "architecture assistant IA memoire auth",
      "reason": "confiance insuffisante pour reponse categorique"
    }
  ]
}
```

## API Principales

### Chat

```http
POST /chat/
Authorization: Bearer <supabase_access_token>
```

Reponse enrichie Phase 2.8:

```json
{
  "response": "...",
  "confidence": 0.73,
  "risk_level": "medium",
  "suggested_sources": [],
  "memories_used": 2,
  "graph_concepts": [],
  "model": "...",
  "provider": "groq"
}
```

Compatibilite:

- le champ historique `response` reste present;
- les anciens consommateurs peuvent ignorer les nouveaux champs;
- `suggested_sources` est une liste vide quand aucune source n'est necessaire.

### Expert Chat

```http
POST /brain/expert-chat
```

Retourne maintenant aussi:

- `confidence`
- `risk_level`
- `suggested_sources`

### Memoire Utilisateur

La memoire utilisateur Supabase est geree par:

```text
models/user_memory.py
core/user_memory_service.py
```

Regle importante:

- `models/` reste pur DTO;
- `core/` contient le service et les acces Supabase;
- chaque operation filtre explicitement par `user_id`.

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

Variables importantes:

```env
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
GROQ_API_KEY=
SUPABASE_URL=
SUPABASE_ANON_KEY=
SUPABASE_SERVICE_ROLE_KEY=
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

## Tests

Commande generale:

```bash
python -m pytest tests -q
```

Tests Phase 2.8 a executer manuellement:

```bash
pytest tests/test_response_confidence.py
pytest tests/test_self_critic.py
pytest tests/test_source_router.py
pytest tests/test_phase28_consciousness.py
pytest tests/test_phase28_routes.py
```

Couverture Phase 2.8:

- confidence faible, moyenne, elevee;
- reponse incertaine;
- reponse contradictoire;
- generation de sources suggerees;
- absence de memoire utilisateur;
- presence de memoire utilisateur;
- retrocompatibilite des nouveaux champs API.

## Roadmap

### Phase 1 - Secure Core

- [x] Configuration centralisee
- [x] Auth API key historique
- [x] CORS configure
- [x] Rate limiting
- [x] Sandbox fichiers
- [x] Permissions
- [x] Audit log
- [x] Backups / rollback
- [x] Dry-run / diff
- [x] Tests securite

### Phase 2 - User Brain

- [x] Supabase client centralise
- [x] Auth Supabase Bearer
- [x] UserService
- [x] User profiles
- [x] User memories
- [x] Memory routing depuis le chat
- [x] Consciousness contextuelle
- [x] Confidence score
- [x] Self critic
- [x] Suggested sources
- [x] Tests unitaires Phase 2.8 ajoutes
- [ ] Execution officielle des tests Phase 2.8
- [ ] Nettoyage du cycle `routers.files -> core.watcher -> routers.files`

### Phase 3 - Production Hardening

- [ ] Observabilite production
- [ ] Gestion fine des roles
- [ ] Rotation des secrets
- [ ] Monitoring Supabase et providers IA
- [ ] UI admin securite
- [ ] Tests d'integration auth/memoire bout en bout

### Phase 4 - Assistant Personnel Etendu

- [ ] Emails
- [ ] Calendrier
- [ ] Comptes externes
- [ ] Workflows projet
- [ ] Automatisation avancee
- [ ] Verification externe automatique des sources

## Notes De Securite

Ne publie jamais `.env`.

Les cles `SUPABASE_SERVICE_ROLE_KEY`, `GROQ_API_KEY`, `HF_API_KEY` et `ADMIN_API_KEY` doivent etre considerees comme des secrets. Si elles ont ete exposees dans un historique Git ou un partage, regenere-les.

Le client Supabase admin contourne les politiques RLS. Les services applicatifs doivent donc toujours filtrer explicitement par `user_id` quand ils manipulent des donnees utilisateur.

---

**MAKEN x MakenBrain** - un cerveau utile, prudent et sous controle.
