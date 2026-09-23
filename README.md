# MakenBrain

**A self-hosted personal AI assistant with memory, a reasoning engine, and multi-agent orchestration.**

`v0.13.0` â€” Phase 8: Multi-Agent Collaborative Orchestration

Built in Python with FastAPI. MakenBrain answers questions using its own stored knowledge, reasons about what it does and does not know, and says so when it isn't sure instead of inventing an answer.

*Code documentation is in French; this README is in English.*

---

## Why this exists

Most assistants answer confidently whether or not they should. MakenBrain was built around the opposite constraint: **an assistant that scores its own confidence before replying, and flags uncertainty rather than filling the gap with a plausible guess.**

Everything else â€” the memory, the knowledge graph, the multi-agent orchestration â€” exists to make that possible.

---

## What it does

**Reasons in stages rather than answering in one shot.** A question goes through analysis, hypothesis generation, evidence collection, then decision. Each stage can report that it lacks what it needs.

**Remembers across conversations.** Vector memory (ChromaDB) for semantic recall, a concept graph (NetworkX) for relationships between ideas, and episodic memory for what happened when.

**Delegates to specialised agents.** An orchestrator routes a task to the agent that fits: planning, research, reasoning, memory, engineering, or presentation.

**Acts on files inside a sandbox.** File operations are confined to a controlled workspace, and sensitive actions are traced.

**Runs on free and local models.** A provider layer routes between Ollama running locally and Groq's free tier, so the assistant costs nothing to operate.

---

## Architecture

```
routers/     23 API endpoints - chat, memory, graph, reasoning, agents,
             search, ingest, audit, providers, profile, presentations...
core/
  reasoning/       13 modules - the staged reasoning engine
  agents/          9 agents + orchestrator, registry, task router
  provider_layer/  model selection and routing (local / cloud)
  episodic/        time-indexed memory
  observability/   tracing of sensitive actions
models/      Pydantic data models
tests/       19 test modules
```

**Import direction is enforced:** `routers -> core -> models`. No circular imports.

### The reasoning engine

| Stage | What it does |
|---|---|
| `QuestionAnalyzerAgent` | Works out what is actually being asked |
| `HypothesisEngine` | Generates candidate answers, deduplicated by Jaccard similarity at a 0.55 threshold |
| `Evidence Engine` | Collects supporting evidence through five strategies, scoring each by relevance x credibility |
| `DecisionEngineAgent` | Combines the evidence into a multi-component confidence score and decides whether to answer |

If confidence lands below threshold, MakenBrain says what it is missing instead of answering anyway.

---

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI |
| Auth & database | Supabase |
| Vector memory | ChromaDB + sentence-transformers |
| Knowledge graph | NetworkX |
| Local models | Ollama |
| Cloud models | Groq (Llama 3.3 70B, free tier) |

---

## Running it

```bash
git clone https://github.com/leonelmaken/makenbrain.git
cd makenbrain

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

pip install -r requirements.txt

cp .env.example .env            # then fill in your own keys
python main.py
```

`.env` is git-ignored and has never been committed. `.env.example` lists every variable the application expects.

---

## Engineering notes

### The cost wall, and what I did about it

The original goal was an assistant that improves itself - one that learns from its own mistakes and reasons better over time. That goal ran into a hard economic constraint, and the constraint is worth stating plainly because it shaped the whole design.

**Staged reasoning multiplies API calls.** A single answer is one call. An answer that critiques and rewrites itself is three. The full pipeline here - analyse, generate several hypotheses, collect evidence for each, score, decide - is seven to ten calls for one question. Same question, ten times the bill.

**Verifying an improvement costs as much as making it.** To know whether a change made the assistant better or worse, you replay a reference set of questions against the new version. A hundred reference questions means a hundred calls per iteration, every iteration.

**Genuine self-improvement means retraining,** which means either GPU time or a paid fine-tuning API. Neither was available.

So the architecture went a different way. The **provider layer** routes each request to the cheapest model that can handle it: Ollama locally for everything that fits on the machine, Groq's free tier for the heavier reasoning. Running MakenBrain costs nothing today. What that buys is the ability to keep building; what it does not buy is the self-training loop, which is deferred rather than solved.

### Current limits

- **Self-improvement is not implemented.** The assistant can critique a single answer, but it does not learn across sessions from its own errors. This is the deferred goal above, not an oversight.
- **Evaluation is manual.** There is no automated regression suite measuring answer quality across versions - only unit and integration tests on the code itself.
- **Local model quality bounds the reasoning.** Routing to a small local model keeps costs at zero but caps how well the harder reasoning stages perform.

### A bug worth documenting: conversation histories bleeding across users

While adding an unrelated feature, I noticed something that didn't add up: a chat session was showing history that didn't belong to the account I was signed in as.

The cause was not the authentication itself - sign-in worked. It was that session lookup trusted the session identifier alone. If you held a session id, the API returned that session's history, regardless of who you were authenticated as. Authentication proved *who you were*; nothing checked *what you were allowed to read*. On a single-user machine this is invisible. The moment a second account exists, it is a cross-account data leak.

The fix has two independent layers, deliberately:

1. **Scoped queries.** Every history route now takes `current_user` through a `require_chat_user` dependency, and every lookup is filtered by that user's id - `list_sessions(user_id=...)`, `get_session_for_user(session_id, user_id)`. A session belonging to someone else is never fetched in the first place.
2. **An explicit ownership assertion.** Before any session is returned or mutated, the handler compares the stored owner against the authenticated user and refuses on mismatch.

The second layer is redundant while the first is correct - which is the point. A future refactor that loosens a query still hits a hard stop before data leaves the process.

**How I knew it was fixed:** signing in as a second account and requesting a session id belonging to the first now fails at the ownership check rather than returning data.

**What I took from it:** authentication and authorisation are different questions, and passing the first tells you nothing about the second. I now treat "who is asking" and "what may they see" as two checks, not one.

---



## Status

Personal project, actively developed in phases. Each phase ships on its own branch and merges into `main`.

Author: **Leonel Maken Dongmo Djouake** - Yaounde, Cameroon
[GitHub](https://github.com/leonelmaken) - [LinkedIn](https://www.linkedin.com/in/leonelmaken) - [Portfolio](https://portfolio-leonel-xi.vercel.app)

