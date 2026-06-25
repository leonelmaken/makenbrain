"""
Domain Explorer — Phase 5
Le cerveau sélectionne un domaine, fait des recherches autonomes,
développe une expertise pointue et répond comme un expert humain.

Domaines supportés : ingénierie, robotique, IA, programmation,
sciences, mathématiques, design, business, médecine, droit, etc.
"""
import asyncio
from datetime import datetime
from typing import Optional

# ── Catalogue de domaines ─────────────────────────────────────────────────────
# Structure : domaine → sous-domaines + requêtes de recherche auto-générées

DOMAIN_CATALOG = {
    # ── Ingénierie & Sciences ──────────────────────────────────────────────
    "génie logiciel": {
        "icon": "💻",
        "description": "Architecture logicielle, design patterns, clean code, DevOps",
        "subtopics": ["design patterns", "clean architecture", "TDD", "CI/CD",
                      "microservices", "API REST", "sécurité applicative"],
        "queries_en": ["software engineering best practices 2025",
                       "clean architecture principles",
                       "design patterns software development",
                       "DevOps CI/CD pipeline best practices"],
    },
    "intelligence artificielle": {
        "icon": "🤖",
        "description": "Machine learning, deep learning, LLM, RAG, computer vision",
        "subtopics": ["machine learning", "deep learning", "LLM", "RAG",
                      "computer vision", "NLP", "reinforcement learning"],
        "queries_en": ["AI machine learning fundamentals 2025",
                       "large language models how they work",
                       "RAG retrieval augmented generation",
                       "computer vision deep learning techniques"],
    },
    "robotique": {
        "icon": "🦾",
        "description": "Systèmes robotiques, ROS, capteurs, actionneurs, contrôle",
        "subtopics": ["ROS", "cinématique", "dynamique", "capteurs",
                      "actionneurs", "planification de trajectoire", "SLAM"],
        "queries_en": ["robotics fundamentals sensors actuators",
                       "ROS robot operating system tutorial",
                       "robot kinematics dynamics control",
                       "SLAM simultaneous localization mapping"],
    },
    "mécatronique": {
        "icon": "⚙️",
        "description": "Intégration mécanique, électronique, informatique et contrôle",
        "subtopics": ["systèmes embarqués", "arduino", "raspberry pi",
                      "PLC", "automatisation", "asservissement", "PID"],
        "queries_en": ["mechatronics systems design",
                       "embedded systems programming",
                       "PID controller tutorial",
                       "industrial automation PLC programming"],
    },
    "électronique": {
        "icon": "⚡",
        "description": "Circuits, microcontrôleurs, signal, électronique de puissance",
        "subtopics": ["circuits analogiques", "circuits numériques",
                      "microcontrôleurs", "FPGA", "PCB design", "signal"],
        "queries_en": ["electronics fundamentals circuits",
                       "microcontroller programming tutorial",
                       "FPGA digital design",
                       "PCB design best practices"],
    },
    "mathématiques": {
        "icon": "📐",
        "description": "Algèbre, analyse, probabilités, statistiques, optimisation",
        "subtopics": ["algèbre linéaire", "calcul différentiel",
                      "probabilités", "statistiques", "optimisation",
                      "théorie des graphes", "cryptographie"],
        "queries_en": ["linear algebra machine learning applications",
                       "probability statistics fundamentals",
                       "mathematical optimization techniques",
                       "graph theory algorithms"],
    },
    "physique": {
        "icon": "⚛️",
        "description": "Mécanique, thermodynamique, électromagnétisme, quantique",
        "subtopics": ["mécanique classique", "thermodynamique",
                      "électromagnétisme", "mécanique quantique", "optique"],
        "queries_en": ["physics fundamentals mechanics",
                       "thermodynamics engineering applications",
                       "electromagnetism fundamentals",
                       "quantum physics basics"],
    },

    # ── Programmation ──────────────────────────────────────────────────────
    "python": {
        "icon": "🐍",
        "description": "Python avancé, async, data science, automatisation",
        "subtopics": ["async/await", "FastAPI", "pandas", "numpy",
                      "pytest", "packaging", "optimisation"],
        "queries_en": ["python advanced techniques 2025",
                       "python async programming tutorial",
                       "python data science pandas numpy",
                       "python best practices clean code"],
    },
    "javascript typescript": {
        "icon": "🟨",
        "description": "JS/TS moderne, React, Node.js, bundlers, testing",
        "subtopics": ["TypeScript avancé", "React patterns",
                      "Node.js", "testing", "bundlers", "performance"],
        "queries_en": ["typescript advanced patterns 2025",
                       "react best practices performance",
                       "nodejs express fastify tutorial",
                       "javascript testing jest vitest"],
    },
    "java spring": {
        "icon": "☕",
        "description": "Java moderne, Spring Boot, microservices, JVM",
        "subtopics": ["Spring Boot 3", "Spring Security", "JPA/Hibernate",
                      "microservices", "reactive", "GraalVM"],
        "queries_en": ["spring boot 3 best practices 2025",
                       "spring security jwt authentication",
                       "java microservices architecture",
                       "java performance optimization JVM"],
    },
    "mobile": {
        "icon": "📱",
        "description": "React Native, Expo, Flutter, iOS, Android natif",
        "subtopics": ["React Native", "Expo", "Flutter",
                      "performance mobile", "animation", "navigation"],
        "queries_en": ["react native expo best practices 2025",
                       "flutter mobile development tutorial",
                       "mobile app performance optimization",
                       "cross platform mobile development"],
    },

    # ── Architecture & Cloud ───────────────────────────────────────────────
    "cloud devops": {
        "icon": "☁️",
        "description": "AWS/GCP/Azure, Docker, Kubernetes, IaC, monitoring",
        "subtopics": ["Docker", "Kubernetes", "Terraform", "CI/CD",
                      "monitoring", "logging", "sécurité cloud"],
        "queries_en": ["cloud architecture best practices 2025",
                       "kubernetes deployment tutorial",
                       "terraform infrastructure as code",
                       "docker containerization best practices"],
    },
    "bases de données": {
        "icon": "🗄️",
        "description": "SQL, NoSQL, optimisation, réplication, sharding",
        "subtopics": ["PostgreSQL", "MongoDB", "Redis", "Elasticsearch",
                      "optimisation requêtes", "réplication", "indexation"],
        "queries_en": ["database design best practices",
                       "postgresql performance optimization",
                       "nosql vs sql when to use",
                       "database indexing strategies"],
    },
    "cybersécurité": {
        "icon": "🔐",
        "description": "Sécurité offensive et défensive, pentest, cryptographie",
        "subtopics": ["OWASP", "pentest", "cryptographie", "JWT", "OAuth",
                      "sécurité API", "SIEM", "SOC"],
        "queries_en": ["cybersecurity fundamentals 2025",
                       "OWASP top 10 vulnerabilities",
                       "penetration testing methodology",
                       "API security best practices"],
    },

    # ── Design & Création ──────────────────────────────────────────────────
    "ui ux design": {
        "icon": "🎨",
        "description": "Design d'interface, UX research, prototypage, design systems",
        "subtopics": ["design systems", "prototypage", "accessibilité",
                      "UX research", "Figma", "animation UI"],
        "queries_en": ["UI UX design principles 2025",
                       "design system component library",
                       "UX research methods techniques",
                       "web accessibility WCAG guidelines"],
    },

    # ── Business & Management ──────────────────────────────────────────────
    "product management": {
        "icon": "📊",
        "description": "Gestion produit, roadmap, OKR, analytics, growth",
        "subtopics": ["roadmap produit", "OKR", "analytics",
                      "growth hacking", "A/B testing", "user stories"],
        "queries_en": ["product management best practices",
                       "product roadmap OKR framework",
                       "growth hacking techniques 2025",
                       "product analytics metrics KPI"],
    },
    "entrepreneuriat": {
        "icon": "🚀",
        "description": "Startup, business model, financement, pitch, scaling",
        "subtopics": ["business model", "pitch deck", "financement",
                      "lean startup", "scaling", "go-to-market"],
        "queries_en": ["startup business model canvas",
                       "startup funding venture capital Africa",
                       "lean startup methodology",
                       "go to market strategy"],
    },

    # ── Fintech & Afrique ──────────────────────────────────────────────────
    "fintech afrique": {
        "icon": "💰",
        "description": "Mobile money, tontines, inclusion financière, réglementation",
        "subtopics": ["mobile money", "MTN MoMo", "Orange Money",
                      "tontines ROSCA", "microfinance", "KYC/AML"],
        "queries_en": ["fintech Africa mobile money 2025",
                       "MTN MoMo API integration",
                       "tontine ROSCA digital platform",
                       "financial inclusion Africa solutions"],
    },
}


async def explore_domain(
    domain_name: str,
    depth: str = "standard",   # "rapide" | "standard" | "expert"
    language: str = "fr",
) -> dict:
    """
    Fait des recherches autonomes sur un domaine et nourrit la mémoire.
    depth='rapide'   → 3 requêtes, résultats snippets
    depth='standard' → 5 requêtes, pages complètes
    depth='expert'   → 8 requêtes, Wikipedia + pages + résumés LLM
    """
    from core.web_search import search_and_ingest

    # Trouver le domaine correspondant
    domain_key = None
    domain_data = None
    for key, data in DOMAIN_CATALOG.items():
        if domain_name.lower() in key or key in domain_name.lower():
            domain_key = key
            domain_data = data
            break

    if not domain_data:
        # Domaine inconnu → recherche générique
        domain_key  = domain_name.lower()
        domain_data = {
            "icon": "🔍",
            "description": domain_name,
            "subtopics": [],
            "queries_en": [
                f"{domain_name} fundamentals tutorial",
                f"{domain_name} best practices 2025",
                f"{domain_name} advanced techniques",
            ],
        }

    # Nombre de requêtes selon la profondeur
    max_q = {"rapide": 2, "standard": 4, "expert": 6}.get(depth, 4)
    fetch = {"rapide": False, "standard": True, "expert": True}.get(depth, True)

    queries   = domain_data["queries_en"][:max_q]
    results   = []
    ingested  = 0

    for q in queries:
        try:
            r = await search_and_ingest(
                query=q,
                max_results=3,
                fetch_pages=fetch,
                auto_summarize=False,
                tags=f"domaine,{domain_key}",
            )
            ingested += r.get("ingested", 0)
            results.append({"query": q, "ingested": r.get("ingested", 0)})
        except Exception as e:
            results.append({"query": q, "error": str(e)})
        await asyncio.sleep(0.5)

    # Wikipedia sur le domaine principal
    if depth in ("standard", "expert"):
        try:
            from core.providers import wikipedia_ingest
            wiki_q = domain_key if language == "fr" else domain_data["description"]
            wiki_r = await wikipedia_ingest(wiki_q, lang=language)
            if wiki_r.get("ingested"):
                ingested += 1
                results.append({"query": f"Wikipedia: {wiki_q}", "ingested": 1})
        except Exception:
            pass

    # En mode expert → aussi sous-topics Wikipedia
    if depth == "expert":
        for subtopic in domain_data.get("subtopics", [])[:3]:
            try:
                from core.providers import wikipedia_ingest
                wiki_r = await wikipedia_ingest(subtopic, lang="fr")
                if wiki_r.get("ingested"):
                    ingested += 1
            except Exception:
                pass
        await asyncio.sleep(0.3)

    return {
        "domain":      domain_key,
        "icon":        domain_data["icon"],
        "description": domain_data["description"],
        "depth":       depth,
        "queries_run": len(results),
        "ingested":    ingested,
        "subtopics":   domain_data.get("subtopics", []),
        "details":     results,
        "message":     f"Expertise '{domain_key}' enrichie — {ingested} nouvelles connaissances.",
        "timestamp":   datetime.now().isoformat(),
    }


def list_domains() -> list[dict]:
    """Liste tous les domaines disponibles."""
    return [
        {
            "key":         key,
            "icon":        data["icon"],
            "description": data["description"],
            "subtopics":   data["subtopics"][:4],
        }
        for key, data in DOMAIN_CATALOG.items()
    ]
