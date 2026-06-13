<div align="center">

# 🧠 MakenBrain

**Cerveau numérique personnel de MAKEN**  
*Local AI · Vector Memory · Knowledge Graph · File Agent · Autonomous Growth*

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-llama3.2:3b-000000?style=flat-square)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vector_memory-FF6B35?style=flat-square)
![License](https://img.shields.io/badge/license-Private-red?style=flat-square)

</div>

---

## Vision

MakenBrain est un cerveau numérique 100 % local qui grandit avec toi.  
Il ingère tes recherches, connecte les concepts entre eux, raisonne, et te rapporte des solutions.  
Il peut modifier tes fichiers, se connecter à d'autres IA, et évoluer de façon autonome.

```
                      ┌─────────────────────────────────┐
                      │         🧠 MakenBrain            │
                      │                                 │
  [Toi]──question──▶  │  Cortex (Ollama llama3.2:3b)   │
                      │        ↕ RAG ↕                  │
  [Réponse]  ◀──────  │  Mémoire (ChromaDB vectors)     │
                      │        ↕                        │
  [Fichiers] ◀──────  │  Agent Fichiers (Python)        │
                      │        ↕                        │
  [Ext. AIs]  ──────▶ │  Connecteurs (Groq / HF / Web)  │
                      └─────────────────────────────────┘
```

---

## Architecture — 6 phases

| Phase | Nom | Statut | Description |
|:---:|---|:---:|---|
| 1 | **Foundation** | 🔨 En cours | LLM local + Mémoire vectorielle + API REST |
| 2 | Ingestion | ⬜ | PDF, URLs, fichiers texte → mémoire |
| 3 | Graphe de neurones | ⬜ | Concepts liés, raisonnement multi-sauts |
| 4 | Agent fichiers | ⬜ | Lire / modifier / créer / supprimer |
| 5 | Connecteurs externes | ⬜ | Groq, Hugging Face, Anthropic, Web Search |
| 6 | Croissance autonome | ⬜ | Scheduler, auto-ingestion, auto-update |

---

## Stack Phase 1

| Composant | Technologie | Rôle |
|---|---|---|
| LLM Local | Ollama + `llama3.2:3b` | Raisonner, synthétiser, décider |
| Mémoire | ChromaDB | Stocker les souvenirs vectoriels |
| Embeddings | `all-MiniLM-L6-v2` | Encoder les textes en vecteurs |
| API | FastAPI + Uvicorn | Interface REST + Swagger |

---

## Prérequis

- [Python 3.11+](https://python.org/downloads)
- [Ollama](https://ollama.ai) installé et le modèle téléchargé :

```bash
ollama pull llama3.2:3b
```

---

## Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/leonelmaken/makenbrain.git
cd makenbrain

# 2. Créer l'environnement virtuel
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / Mac
source .venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer l'environnement
copy .env.example .env   # Windows
# cp .env.example .env   # Linux/Mac
```

---

## Lancement

```bash
# Terminal 1 — démarrer Ollama
ollama serve

# Terminal 2 — démarrer MakenBrain
uvicorn main:app --reload --port 8000
```

Ouvre ensuite **http://localhost:8000/docs** pour l'interface Swagger interactive.

---

## API — Phase 1

### Chat avec mémoire (RAG)
```bash
POST /chat/
{
  "message": "Quel est mon projet fintech principal?",
  "use_memory": true,
  "n_context": 5
}
```

### Ajouter un souvenir
```bash
POST /memory/add
{
  "content": "SmartBudget Africa est une application fintech panafricaine.",
  "source": "projet",
  "tags": "smartbudget,fintech,afrique"
}
```

### Recherche sémantique
```bash
POST /memory/search
{
  "query": "fintech Afrique",
  "n_results": 5
}
```

### Ingérer un texte brut
```bash
POST /ingest/text
{
  "text": "La tontine est un système d'épargne collectif...",
  "title": "Tontines Africa",
  "tags": "tontine,finance"
}
```

### Ingérer depuis une URL
```bash
POST /ingest/url
{
  "url": "https://example.com/article-fintech",
  "tags": "web,recherche"
}
```

### Statistiques mémoire
```bash
GET /memory/stats
```

---

## Exemple complet

```bash
# 1. Nourrir le cerveau
curl -X POST http://localhost:8000/memory/add \
  -H "Content-Type: application/json" \
  -d '{"content":"Spring Boot backend de SmartBudget est à 85% et tourne sur Java 17 avec PostgreSQL Neon.", "source":"dev", "tags":"smartbudget,backend,java"}'

# 2. Poser une question avec contexte mémoire
curl -X POST http://localhost:8000/chat/ \
  -H "Content-Type: application/json" \
  -d '{"message":"Quel est l état actuel du backend SmartBudget?"}'
```

---

## Structure du projet

```
makenbrain/
├── main.py              # Point d'entrée FastAPI
├── requirements.txt
├── .env.example
├── core/
│   ├── config.py        # Configuration (pydantic-settings)
│   ├── llm.py           # Client Ollama — le cortex
│   └── memory.py        # Client ChromaDB — la mémoire
├── models/
│   └── schemas.py       # Schémas Pydantic
├── routers/
│   ├── chat.py          # POST /chat/
│   ├── memory.py        # CRUD mémoire
│   └── ingest.py        # Ingestion texte + URL
├── brain_data/          # Données locales (gitignore)
└── tests/
```

---

## Roadmap détaillée

### Phase 2 — Ingestion avancée
- [ ] Ingestion PDF (PyMuPDF)
- [ ] Ingestion de dossiers entiers
- [ ] Déduplication intelligente
- [ ] Résumé automatique à l'ingestion

### Phase 3 — Graphe de neurones
- [ ] Extraction de concepts (NER)
- [ ] Graphe JSON / Neo4j
- [ ] Raisonnement multi-sauts
- [ ] Détection de contradictions

### Phase 4 — Agent fichiers
- [ ] Lecture / écriture de fichiers
- [ ] Modification de code
- [ ] Surveillance de dossiers (Watchdog)
- [ ] Résumé automatique des changements

### Phase 5 — Connecteurs externes
- [ ] Groq API (Llama 3.3 70B gratuit)
- [ ] Hugging Face Inference
- [ ] Web search (DuckDuckGo / SerpAPI)
- [ ] Anthropic Claude (fallback)

### Phase 6 — Croissance autonome
- [ ] Scheduler (APScheduler)
- [ ] Auto-ingestion de sources
- [ ] Mise à jour des connexions logiques
- [ ] Rapport quotidien de croissance

---

<div align="center">

**MAKEN × MakenBrain** — *Le cerveau grandit avec toi.*

</div>
