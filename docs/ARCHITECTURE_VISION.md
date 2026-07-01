# MakenBrain — Document d'Architecture & Vision Long Terme

**Statut du projet au moment de la rédaction :** Phase 3.4 terminée (Decision Engine). Phase 3.5 (Synthesizer) à venir.
**Auteur :** Architecture proposée par l'assistant IA, validée et amendée par Leonel Maken.
**Objectif du document :** servir de référence stable pour toutes les décisions d'architecture futures. À mettre à jour à chaque évolution majeure — ne pas laisser ce document devenir obsolète.

---

## 0. Cadre de lecture

Ce document répond à 7 questions structurantes. Il est volontairement long et précis car il doit pouvoir trancher des débats d'architecture dans 6 mois sans qu'on ait à reconstruire le raisonnement depuis zéro.

Principe directeur retenu pour toutes les décisions ci-dessous : **MakenBrain progresse par couches stables**, jamais par accumulation désordonnée. Chaque nouvelle capacité doit s'appuyer sur une fondation déjà testée, observable et compréhensible.

---

## 1. Vision long terme

### Dans 1 an (mi-2027)

MakenBrain est un **cerveau numérique personnel opérationnel** pour un seul utilisateur (toi), avec :

- Pipeline de raisonnement complet (3.1 → 3.5) en production, fiable, observé.
- Architecture multi-agents fonctionnelle avec 4-5 agents cœur : Orchestrator, Memory, Reasoning, Planning, Research.
- Mémoire qui se consolide d'elle-même (pas juste un stockage, une vraie mémoire à long terme avec oubli sélectif et renforcement).
- Capacité à exécuter des tâches multi-étapes de bout en bout (ex : "recherche, planifie, code, teste, documente").
- Multi-modèles opérationnel (au moins Claude + un modèle local + un modèle de secours).
- Interface utilisateur web simple mais fonctionnelle (pas juste Swagger/API).
- Espace Super Admin minimal : tableau de bord de coûts, logs structurés, audit des décisions des agents.

**Critère de succès à 1 an :** tu peux donner une tâche de complexité réelle ("construis-moi le MVP d'un outil X") et MakenBrain produit un résultat exploitable avec supervision légère, pas une supervision constante.

### Dans 2 ans (mi-2028)

MakenBrain devient un **collaborateur de développement et de décision** à part entière, dans le contexte où tu es probablement en cours de Master IA/Robotique à Ottawa :

- Apprentissage continu réel : MakenBrain connaît tes habitudes, ton style de travail, tes projets en cours, sans que tu aies à tout rappeler.
- Capacité à mener des projets logiciels complets en autonomie supervisée (CodingAgent + TestingAgent + DocumentationAgent orchestrés).
- Module de veille technologique opérationnel : MakenBrain te signale chaque semaine les évolutions pertinentes (nouveaux modèles, techniques, papers) et te propose des adaptations argumentées pour MakenBrain lui-même.
- Capacité Business Agent : aide à l'analyse de marché, à la structuration d'idées entrepreneuriales, au suivi de métriques business si tu lances d'autres projets.
- Auto-évolution en mode "conseiller" : MakenBrain audite régulièrement son propre code et propose des refactorings argumentés (jamais appliqués sans validation).
- Multi-utilisateurs possible si tu décides d'ouvrir l'accès (famille, associés, employés d'un projet que tu lances).

**Critère de succès à 2 ans :** MakenBrain est devenu un avantage compétitif concret dans tes autres projets — pas un side-project gadget, un outil qui te fait gagner du temps et de la qualité de décision mesurables.

### Dans 5 ans (2031)

C'est l'horizon que tu t'es fixé pour l'objectif financier. À cet horizon, deux scénarios honnêtes sont à distinguer :

**Scénario A — MakenBrain reste un outil personnel/interne.** Il est alors le "cerveau" derrière plusieurs de tes ventures : il connaît l'historique complet de tes décisions, sert d'assistant stratégique, gère une bonne partie de l'automatisation opérationnelle de tes structures (reporting, veille, premières lignes de code, documentation, support). Sa valeur est qu'il **compose** avec le temps — chaque année il en sait plus sur toi et sur ce qui a fonctionné.

**Scénario B — MakenBrain (ou une dérivée) devient un produit.** Si à un moment l'architecture multi-agents + mémoire + auto-évolution atteint un niveau de maturité différenciant, il peut devenir un produit packagé (B2B ou B2C). C'est une option à garder ouverte, pas un objectif fixé maintenant — la décision se prendra avec des données, pas par anticipation.

**Version "ultime" sans contrainte technique :** un système qui combine raisonnement fiable de niveau expert, mémoire véritablement longitudinale (des années d'historique exploitable), autonomie d'exécution sur des tâches complexes avec supervision minimale, capacité d'auto-amélioration argumentée, et une compréhension fine et continuellement mise à jour de l'utilisateur qu'il sert. Pas un assistant générique — un système qui devient meilleur *spécifiquement pour toi* à mesure du temps, ce qu'aucun assistant généraliste grand public ne peut faire par construction (ils servent des millions d'utilisateurs sans mémoire individuelle profonde).

### Remarque honnête sur l'ambition financière

MakenBrain seul ne te rendra pas milliardaire avant 2031 — aucun outil interne ne le fait directement. Ce qui peut réellement contribuer à cet objectif :
1. **MakenBrain comme accélérateur** de tes autres ventures (vitesse d'exécution, qualité de décision, automatisation).
2. **D'autres projets** à fort potentiel de scalabilité que tu lanceras — je peux t'aider à les évaluer/structurer quand tu voudras, séparément de ce document.
3. Le fait que tu ailles chercher un Master IA/Robotique à Ottawa est cohérent avec la stratégie : ça construit ta crédibilité technique et ton réseau, deux actifs qui composent aussi avec le temps.

Je le note ici pour que ce soit dit une fois clairement : on garde MakenBrain comme fondation technique sérieuse, et on traite les choix de "quel projet peut rapporter" comme des conversations séparées et explicites quand tu seras prêt à les avoir.

---

## 2. Architecture générale du système

### 2.1 Principe : MakenBrain en 6 couches

```
┌─────────────────────────────────────────────────────────────┐
│ 6. INTERFACE                                                  │
│    Web UI · API publique · CLI · (futur : voix)               │
├─────────────────────────────────────────────────────────────┤
│ 5. ORCHESTRATION                                               │
│    Brain Orchestrator · Task Router · Agent Registry          │
├─────────────────────────────────────────────────────────────┤
│ 4. AGENTS SPÉCIALISÉS                                          │
│    Reasoning · Memory · Planning · Coding · Research · etc.   │
├─────────────────────────────────────────────────────────────┤
│ 3. INTELLIGENCE PARTAGÉE                                       │
│    Reasoning Pipeline (3.1-3.5) · Confidence · Self-Critic     │
├─────────────────────────────────────────────────────────────┤
│ 2. MÉMOIRE & CONNAISSANCE                                       │
│    Vector Memory · Neuron Graph · User Profile · Episodic Log │
├─────────────────────────────────────────────────────────────┤
│ 1. FONDATIONS                                                   │
│    Auth · Config · Providers LLM · Audit · Sécurité            │
└─────────────────────────────────────────────────────────────┘
```

Chaque couche ne dépend que des couches inférieures. Aucune dépendance remontante. C'est déjà globalement respecté dans le code actuel (`models/` → `core/` → `routers/`) — on l'étend simplement aux nouvelles couches.

### 2.2 Modules détaillés

#### Couche 1 — Fondations (existe déjà, à durcir)
| Module | Responsabilité |
|---|---|
| `core/config.py` | Source unique de configuration |
| `core/auth.py` | Authentification API key + Supabase |
| `core/providers.py` → **à étendre en Provider Abstraction Layer** | Sélection et appel des modèles LLM |
| `core/audit.py` → **à migrer en structured logging** | Traçabilité de toute action sensible |
| `core/rate_limit.py`, `core/permissions.py`, `core/sandbox.py` | Sécurité opérationnelle |

#### Couche 2 — Mémoire & Connaissance (existe partiellement, à enrichir fortement)
| Module | Responsabilité | État |
|---|---|---|
| `core/memory.py` (Vector Memory) | Mémoire sémantique, recherche par similarité | ✅ existe |
| `core/graph.py` (Neuron Graph) | Relations entre concepts, raisonnement structurel | ✅ existe, sous-utilisé |
| `core/user_profile.py` | Profil déclaratif utilisateur | ✅ existe |
| **`core/episodic_memory.py` (nouveau)** | Historique des interactions, "ce qui s'est passé" — distinct de la mémoire sémantique | ❌ à créer |
| **`core/memory_consolidator.py` (nouveau)** | Tâche périodique : fusionne, résume, élague la mémoire | ❌ à créer (Phase 6) |

C'est la couche la plus critique pour la différenciation du projet (voir section 6). Une mémoire purement vectorielle (ce qu'on a) ne suffit pas à faire un "cerveau" — il manque la dimension **épisodique** (quand, dans quel contexte, avec quel résultat) et la dimension **consolidation** (le cerveau humain ne garde pas tout brut, il résume et oublie sélectivement).

#### Couche 3 — Intelligence partagée (existe, à compléter)
| Module | Responsabilité | État |
|---|---|---|
| `core/reasoning/*` | Pipeline expert 3.1 → 3.5 | 🟡 4/5 phases |
| `core/response_confidence.py` | Score de confiance | ✅ existe |
| `core/self_critic.py` | Détection d'incertitude | ✅ existe |

Cette couche est **partagée par tous les agents** — un CodingAgent ou un BusinessAgent doivent pouvoir invoquer le même pipeline de raisonnement que le chat principal. C'est pour ça qu'elle est une couche à part, pas un agent parmi d'autres.

#### Couche 4 — Agents spécialisés (à construire, détaillé en section 3)

#### Couche 5 — Orchestration (à construire en priorité après 3.5)
| Module | Responsabilité |
|---|---|
| `core/orchestrator/brain_orchestrator.py` | Reçoit une intention, décompose, route, agrège |
| `core/orchestrator/agent_registry.py` | Déclaration et découverte des agents disponibles |
| `core/orchestrator/task_router.py` | Décide quel(s) agent(s) traiter une tâche |

#### Couche 6 — Interface
| Module | Responsabilité | État |
|---|---|---|
| `routers/*` (API REST) | Existe déjà, bien structuré | ✅ |
| **Web UI (nouveau)** | Interface utilisateur réelle | ❌ — actuellement seulement Swagger |
| **Admin Dashboard (nouveau)** | Espace Super Admin | ❌ |
| Voix (futur, lointain) | Interaction naturelle, enseignement oral | ❌ pas avant maturité Phase 6+ |

### 2.3 Ordre de développement recommandé

1. **Couche 3** : finir Phase 3.5 (Synthesizer) — déjà planifié.
2. **Couche 1** : durcir (observabilité, provider abstraction) — c'est la fondation de tout le reste, le faire avant d'empiler des agents dessus évite de la dette qui se propage.
3. **Couche 5** : Orchestrator minimal (juste assez pour router vers 2-3 agents).
4. **Couche 4** : agents un par un, dans l'ordre de la section 3.
5. **Couche 2 enrichie** : mémoire épisodique + consolidation, en parallèle des agents dès qu'on a 2-3 agents qui produisent de l'historique à retenir.
6. **Couche 6** : Web UI dès que l'orchestrateur a une valeur démontrable — ne pas attendre la fin, une UI simple tôt aide à *sentir* si le système est utile.

---

## 3. Architecture multi-agents

### 3.1 Principe de communication

Tous les agents communiquent **exclusivement via l'Orchestrator**, jamais entre eux directement. Ça évite le couplage en spaghetti et garde une trace centralisée de tout ce qui se passe.

```python
@dataclass
class AgentTask:
    task_id: str
    intent: str                  # ce qu'on demande à l'agent
    context: dict                # session_id, user_id, données utiles
    parent_task_id: str | None   # si sous-tâche d'une tâche plus large
    priority: int

@dataclass
class AgentResult:
    task_id: str
    agent_name: str
    success: bool
    output: Any
    confidence: float
    cost: dict                   # tokens utilisés, modèle, latence
    needs_followup: list[AgentTask]  # l'agent peut demander des sous-tâches
```

### 3.2 Protocole commun à tous les agents

```python
class BaseAgent(ABC):
    name: str
    description: str
    capabilities: list[str]      # ce que l'agent sait faire, pour le routing
    autonomy_level: AutonomyLevel  # voir 3.3

    async def can_handle(self, task: AgentTask) -> float:
        """Retourne un score 0-1 de pertinence pour cette tâche"""

    async def run(self, task: AgentTask) -> AgentResult: ...

    async def get_memory_context(self, task: AgentTask) -> dict:
        """Chaque agent peut interroger la mémoire partagée, jamais la modifier directement"""
```

### 3.3 Niveau d'autonomie — concept clé

Chaque agent a un **niveau d'autonomie** explicite, pas un seul niveau global pour tout le système :

| Niveau | Signification | Exemples d'agents |
|---|---|---|
| `READ_ONLY` | Peut consulter, ne modifie rien | ResearchAgent, ReasoningAgent |
| `SUGGEST` | Propose une action, attend validation humaine | CodingAgent (propose un diff), BusinessAgent |
| `SANDBOXED_EXECUTE` | Exécute dans un environnement isolé/réversible | TestingAgent, CodingAgent (en sandbox) |
| `AUTONOMOUS` | Exécute sans validation préalable | MemoryAgent (écriture mémoire interne), AutomationAgent (tâches répétitives déjà validées une fois) |

C'est ce niveau qui répond directement à ta demande de la section 7 : "aucune modification importante sans validation". Le niveau d'autonomie n'est pas une promesse vague, c'est un champ structuré sur chaque agent, vérifié par l'Orchestrator avant exécution.

### 3.4 Catalogue des agents

| Agent | Responsabilité | Autonomie | Priorité dev |
|---|---|---|---|
| **Brain Orchestrator** | Décompose l'intention, route, agrège, arbitre les conflits entre agents | N/A (méta) | **1 — fondation** |
| **Reasoning Agent** | Enveloppe le pipeline 3.1-3.5 existant | READ_ONLY | **1 — déjà presque prêt** |
| **Memory Agent** | Lecture/écriture mémoire vectorielle + épisodique, scoring de pertinence | AUTONOMOUS (écriture interne) | **2** |
| **Planning Agent** | Décompose une tâche complexe en sous-tâches ordonnées avec dépendances | SUGGEST | **2** |
| **Research Agent** | Recherche web, ingestion de sources, synthèse documentaire | READ_ONLY | **3** |
| **Coding Agent** | Génération de code, application de diffs en sandbox | SANDBOXED_EXECUTE | **3** |
| **Testing Agent** | Écrit et exécute des tests sur le code produit par CodingAgent | SANDBOXED_EXECUTE | **3** |
| **Documentation Agent** | Génère/maintient la documentation à partir du code et des décisions | SUGGEST | **4** |
| **Learning Agent** | Analyse les interactions passées, ajuste profil utilisateur, alimente la consolidation mémoire | AUTONOMOUS (sur données internes uniquement) | **4** |
| **Business Agent** | Analyse de marché, structuration d'idées, suivi de métriques pour tes ventures | SUGGEST | **5** |
| **Creative Agent** | Idéation, brainstorming, contenu créatif | SUGGEST | **5** |
| **Media Agent** | Génération d'images/audio/vidéo via services spécialisés externes | SUGGEST | **6** |
| **Automation Agent** | Exécution de tâches répétitives déjà validées (rapports, rappels, scraping régulier) | AUTONOMOUS (sur process pré-validés) | **6** |
| **Super Admin Advisor** | Analyse MakenBrain lui-même : dette technique, perf, coûts, propositions d'évolution | READ_ONLY + SUGGEST | **3 — voir section 7** |

**Remarque sur la priorisation :** Super Admin Advisor monte en priorité 3 (pas tout à la fin) car il a une valeur immédiate dès que 2-3 agents existent — il n'a pas besoin que tout le catalogue soit fini pour commencer à observer et conseiller.

### 3.5 Mémoire des agents

Trois portées de mémoire, pour éviter la confusion classique "tout dans le même sac" :

1. **Mémoire de tâche** (éphémère) : contexte d'une seule exécution d'agent, détruite après.
2. **Mémoire de session** (court terme) : partagée entre agents pendant une conversation/projet en cours.
3. **Mémoire long terme** (persistante) : la couche 2 du système — vectorielle + épisodique + graphe. Tous les agents y lisent, seul le Memory Agent y écrit (pour garder une politique d'écriture cohérente et éviter la pollution mémoire par des agents mal calibrés).

---

## 4. Fonctionnalités utilisateur (vision produit final, priorisées)

### Priorité 1 — Cœur d'usage quotidien
- Chat conversationnel avec mémoire longitudinale réelle (pas juste contexte de session)
- Raisonnement expert pour questions complexes avec justification visible
- Recherche dans sa propre mémoire ("qu'est-ce que je t'ai dit sur X le mois dernier ?")
- Historique de conversations organisé par sujet/projet
- Niveau de confiance visible sur chaque réponse + sources citées

### Priorité 2 — Productivité & exécution
- Exécution de tâches multi-étapes supervisées (recherche → synthèse → document)
- Génération de code avec contexte projet (CodingAgent)
- Gestion de projets personnels (suivi, rappels, état d'avancement)
- Génération de documents/présentations/tableaux professionnels
- Ingestion de documents (PDF, notes, liens) dans la mémoire personnelle

### Priorité 3 — Apprentissage personnel
- Mode "professeur" : explications adaptées au niveau, exercices, corrections
- Suivi de progression sur des sujets choisis (ex : tes cours de Master à venir)
- Rappels de séances d'apprentissage planifiées
- Génération de fiches de révision à partir de la mémoire accumulée

### Priorité 4 — Création de contenu
- Génération d'images, présentations, rapports professionnels
- Génération audio/vidéo (via intégrations spécialisées externes)
- Assistance à l'écriture (style adapté, cohérence avec contenu précédent)

### Priorité 5 — Vie professionnelle/entrepreneuriale
- Suivi de métriques business pour tes projets
- Aide à la structuration d'idées et de business plans
- Veille sectorielle automatisée et synthétisée

### Priorité 6 — Confort & accessibilité
- Interface vocale (enseignement et interaction naturelle)
- Multi-device avec synchronisation
- Mode hors-ligne partiel (modèle local pour tâches simples)

---

## 5. Fonctionnalités Super Admin

L'espace Super Admin doit répondre à une question : **"Comment va MakenBrain et que devrais-je faire ensuite ?"**

### Suivi & observabilité
- Dashboard de santé système (latence, taux d'erreur, disponibilité par provider LLM)
- Coûts en temps réel par provider/modèle, avec projection mensuelle
- Logs structurés consultables et filtrables (par agent, session, type d'erreur)
- Trace complète d'une décision d'agent (replay : pourquoi tel choix a été fait)

### Performance & qualité
- Métriques de qualité du raisonnement (taux de contradiction détecté, score de confiance moyen, taux de fallback LLM utilisé)
- Comparaison de performance entre providers LLM sur des tâches similaires
- Détection de régressions (une réponse moins bonne qu'avant sur un type de question donné)

### Dette technique & architecture
- Rapport périodique généré par le **Super Admin Advisor** : modules à risque, complexité cyclomatique, couverture de tests, dépendances obsolètes
- Cartographie des dépendances entre modules (détection automatique de couplages dangereux, comme l'import cyclique déjà identifié)

### Pilotage des modèles IA
- Activation/désactivation de providers
- Règles de routing modèle ↔ type de tâche, modifiables sans redéploiement
- A/B testing entre modèles sur un échantillon de requêtes

### Gouvernance des agents
- Vue d'ensemble de tous les agents actifs, leur niveau d'autonomie, leur taux de succès
- File d'attente des propositions d'agents en attente de validation (CodingAgent, BusinessAgent en mode SUGGEST)
- Historique des validations/rejets (pour que le système apprenne ce que tu valides généralement)

### Évolution & feuille de route
- Propositions d'amélioration argumentées, classées par impact/effort
- Veille technologique synthétisée (voir section 7)
- Feuille de route vivante, mise à jour automatiquement avec statut des phases

### Administration brute
- Gestion des utilisateurs (si MakenBrain s'ouvre à d'autres personnes un jour)
- Gestion des secrets/clés API, rotation
- Sauvegardes et restauration de la mémoire

---

## 6. Différenciation — ce qui rend MakenBrain unique

Les assistants généralistes (Claude, ChatGPT, Gemini) sont structurellement limités sur trois points que MakenBrain peut exploiter :

1. **Mémoire individuelle profonde et permanente.** Les assistants grand public n'ont pas (ou peu) de mémoire longitudinale riche et structurée par utilisateur. MakenBrain peut construire une vraie continuité — des années d'historique exploitable, avec mémoire épisodique + consolidation (section 2.2).

2. **Raisonnement traçable et justifié.** Le pipeline 3.1-3.5 produit une trace explicite : hypothèses considérées, preuves évaluées, contradictions détectées, raison de la décision. Aucun assistant grand public n'expose ce niveau de transparence sur son raisonnement. C'est vendable comme fiabilité, pas comme gadget.

3. **Autonomie d'exécution calibrée et supervisée.** Le concept de `autonomy_level` par agent (section 3.3) permet une autonomie réelle (pas juste du chat) tout en gardant un contrôle humain fin — différent à la fois des agents "tout autonome" risqués et des chatbots purement passifs.

### Idées de fonctionnalités différenciantes concrètes
- **"Pourquoi cette réponse ?"** — bouton qui affiche la trace de raisonnement complète, pas une boîte noire.
- **Mémoire consultable et éditable** — l'utilisateur peut voir, corriger, supprimer ce que MakenBrain "sait" de lui (transparence = confiance).
- **Niveau de confiance affiché systématiquement** — déjà en place, à pousser comme signature de marque ("MakenBrain ne prétend jamais être sûr quand il ne l'est pas").
- **Super Admin Advisor auto-critique** — un système qui audite et critique sa propre architecture est rare, et cohérent avec ta philosophie d'amélioration continue.
- **Spécialisation progressive par utilisateur** — au lieu d'un modèle figé, le comportement de MakenBrain dérive vers ce qui marche pour toi spécifiquement (ton style de validation, tes domaines, ton vocabulaire).

---

## 7. Auto-évolution — le Super Admin Advisor en détail

### Ce que ça fait
Le Super Admin Advisor (agent READ_ONLY + SUGGEST, voir 3.4) tourne en tâche de fond périodique et produit des rapports structurés, jamais des actions automatiques.

**Capacités prévues, par ordre de maturité :**

1. **Audit de code statique** (réaliste dès Phase 5) : complexité, duplication, conventions non respectées, TODO oubliés, couverture de tests par module.
2. **Audit d'architecture** (Phase 5-6) : détection de couplages dangereux (comme l'import cyclique actuel), modules qui grossissent sans découpage, dépendances circulaires entre agents.
3. **Détection de limites fonctionnelles** (Phase 6) : analyse des échecs/fallbacks fréquents du pipeline de raisonnement, des types de questions où la confiance reste systématiquement basse.
4. **Comparaison avec l'état de l'art** (Phase 6-7, **nécessite une source d'information fiable** — ne pas inventer de comparatif sans données réelles) : si on branche une veille (RSS, API documentées, recherche web), l'Advisor peut signaler "Anthropic a publié telle capacité, voici comment on pourrait l'adapter à l'identité de MakenBrain" — toujours en proposition argumentée, jamais en copie aveugle.
5. **Feuille de route vivante** (Phase 7+) : maintien automatique d'un document de roadmap mis à jour avec statut réel, pas un document qui se périme.

### Garde-fous non négociables
- **Aucune action automatique** — l'Advisor produit des rapports et des propositions, jamais des commits, jamais des déploiements.
- **Traçabilité complète** — chaque proposition doit citer ce qui l'a déclenchée (quelle métrique, quel pattern observé).
- **Validation humaine systématique** pour toute modification "importante" — à définir précisément quand on implémentera l'agent (probablement : tout ce qui touche à plus d'un module, ou à la sécurité/auth, ou aux coûts).

### Module de veille technologique — conception réaliste
Tu en as parlé dans le prompt système. Voici une conception modulaire qui évite le piège classique (un scraper fragile qui invente des comparatifs) :

- **Sources structurées uniquement** au départ : changelogs officiels (Anthropic, OpenAI, Google), pas de scraping généraliste non fiable.
- **Pipeline** : ingestion source → extraction des capacités nouvelles → passage dans le `ReasoningAgent` existant pour évaluer la pertinence pour MakenBrain → proposition argumentée stockée pour le Super Admin.
- **Fréquence raisonnable** : hebdomadaire, pas temps réel — la veille n'a pas besoin d'être instantanée pour être utile, et ça limite le bruit et le coût.
- Ce module n'est pas prioritaire avant la Phase 6-7 — il a besoin que l'Advisor et le Reasoning Pipeline soient déjà matures pour produire des propositions de qualité plutôt que du bruit.

---

## Synthèse — Ordre d'implémentation validé

1. **Phase 3.5** — Synthesizer *(prochaine étape immédiate, déjà convenu)*
2. **Phase 4** — Observabilité, provider abstraction, dette technique (couche 1 durcie)
3. **Phase 5** — Orchestrator + agents cœur (Reasoning, Memory, Planning) + Super Admin Advisor minimal
4. **Phase 6** — Research, Coding, Testing agents + mémoire épisodique/consolidation + apprentissage continu
5. **Phase 7** — Documentation, Business, Creative agents + multi-modèles complet + veille technologique
6. **Phase 8** — Media, Automation agents + Web UI complète + interface vocale

Ce document sert de boussole. À chaque début de phase, on le relit pour vérifier qu'on construit toujours dans la bonne direction — et on l'amende si la réalité du terrain (ce qu'on apprend en codant) contredit une hypothèse posée ici.
