"""
Extracteur de concepts — universel, tous domaines.
Ingénierie, sciences, tech, médecine, droit, finance, art...
fast=True  → patterns regex + dictionnaire (< 1ms)
fast=False → LLM pour extraction précise
"""
import re
import json
from pathlib import Path
from core.config import settings

DOMAIN = {
    # ── Ingénierie & Sciences ────────────────────────────────────────────────
    "ingénierie": [
        # Génie logiciel
        "algorithme", "structure de données", "complexité", "récursion",
        "programmation orientée objet", "design pattern", "solid", "clean code",
        "refactoring", "architecture logicielle", "microservice", "monolith",
        "api rest", "graphql", "websocket", "grpc", "message queue",
        # Génie mécanique
        "mécanique", "résistance des matériaux", "thermodynamique",
        "dynamique des fluides", "cad", "cao", "solidworks", "autocad",
        "impression 3d", "fabrication additive", "usinage", "cnc",
        "contrainte", "déformation", "couple", "couple moteur", "rendement",
        # Génie électrique / Électronique
        "circuit", "résistance", "condensateur", "inductance", "transistor",
        "amplificateur", "oscillateur", "microcontrôleur", "arduino", "raspberry pi",
        "stm32", "esp32", "fpga", "vhdl", "verilog", "pcb", "soudure",
        "signal analogique", "signal numérique", "adc", "dac", "pwm", "uart", "spi", "i2c",
        # Génie civil
        "béton", "structure", "fondation", "charge", "moment fléchissant",
        "bim", "revit", "autocad civil", "topographie", "hydraulique",
    ],
    "robotique": [
        "robot", "robots", "bras robotique", "effecteur", "capteur",
        "lidar", "caméra stéréo", "vision par ordinateur", "opencv",
        "ros", "ros2", "gazebo", "slam", "navigation autonome",
        "cinématique", "cinématique inverse", "jacobien", "pid", "contrôle",
        "servo moteur", "moteur pas à pas", "encodeur", "odométrie",
        "drone", "uav", "quadcopter", "autopilote", "ardupilot", "mavlink",
        "manipulation", "préhension", "gripper", "cobots", "cobotique",
    ],
    "mécatronique": [
        "mécatronique", "système embarqué", "temps réel", "rtos", "freertos",
        "automate", "plc", "scada", "industrie 4.0", "iot", "iiot",
        "asservissement", "régulation", "pid", "correcteur", "boucle fermée",
        "acquisition de données", "capteur de température", "pression",
        "actionneur", "vanne", "pompe", "convoyeur",
    ],
    "intelligence_artificielle": [
        "machine learning", "deep learning", "réseau de neurones", "neural network",
        "cnn", "rnn", "lstm", "transformer", "attention", "bert", "gpt",
        "reinforcement learning", "q-learning", "policy gradient",
        "classification", "régression", "clustering", "kmeans",
        "random forest", "gradient boosting", "xgboost", "lightgbm",
        "feature engineering", "overfitting", "underfitting", "régularisation",
        "backpropagation", "gradient descent", "adam", "sgd",
        "computer vision", "nlp", "traitement du langage naturel",
        "génération de texte", "rag", "embedding", "vector store",
        "llm", "llama", "mistral", "gemini", "claude", "gpt-4",
        "fine-tuning", "prompt engineering", "few-shot", "zero-shot",
    ],
    "mathématiques": [
        "algèbre linéaire", "matrice", "vecteur", "déterminant", "valeur propre",
        "calcul différentiel", "intégrale", "dérivée", "gradient", "hessien",
        "probabilité", "statistiques", "loi normale", "distribution",
        "espérance", "variance", "covariance", "corrélation",
        "équation différentielle", "série de fourier", "transformée de laplace",
        "optimisation", "convexe", "simplex", "lagrangien",
        "théorie des graphes", "combinatoire", "logique", "preuve",
    ],
    "physique": [
        "mécanique classique", "newton", "gravitation", "énergie cinétique",
        "thermodynamique", "entropie", "chaleur", "travail",
        "électromagnétisme", "maxwell", "champ électrique", "champ magnétique",
        "optique", "laser", "interférence", "diffraction",
        "mécanique quantique", "qubit", "superposition", "intrication",
        "relativité", "espace-temps", "physique nucléaire",
    ],
    # ── Technologies ────────────────────────────────────────────────────────
    "technologie": [
        # Langages
        "python", "java", "javascript", "typescript", "c", "c++", "c#",
        "rust", "go", "kotlin", "swift", "php", "ruby", "scala", "haskell",
        "matlab", "r", "julia", "assembly", "bash", "powershell",
        # Frameworks Backend
        "spring boot", "django", "flask", "fastapi", "express", "nestjs",
        "laravel", "rails", "asp.net", "gin", "fiber",
        # Frameworks Frontend / Mobile
        "react", "vue", "angular", "svelte", "nextjs", "nuxt",
        "react native", "expo", "flutter", "ionic", "xamarin",
        "tailwind", "bootstrap", "material ui", "shadcn",
        # Bases de données
        "postgresql", "mysql", "mongodb", "redis", "sqlite", "cassandra",
        "elasticsearch", "neo4j", "influxdb", "supabase", "neon", "firebase",
        # DevOps / Cloud
        "docker", "kubernetes", "terraform", "ansible", "jenkins",
        "github actions", "gitlab ci", "aws", "gcp", "azure",
        "vercel", "render", "railway", "heroku", "digitalocean",
        # Auth
        "jwt", "oauth2", "keycloak", "auth0", "firebase auth",
        "bcrypt", "ssl", "tls", "cors",
    ],
    # ── Domaines métier ──────────────────────────────────────────────────────
    "finance": [
        "fintech", "banque", "assurance", "comptabilité", "audit",
        "bourse", "action", "obligation", "portefeuille", "rendement",
        "risque", "hedge", "dérivé", "option", "future",
        "blockchain", "cryptomonnaie", "bitcoin", "ethereum", "defi",
        "paiement", "mobile money", "mtn momo", "orange money",
        "microfinance", "tontine", "wallet", "kyc", "aml",
    ],
    "médecine": [
        "anatomie", "physiologie", "pathologie", "diagnostic", "traitement",
        "chirurgie", "pharmacologie", "médicament", "essai clinique",
        "imagerie médicale", "irm", "scanner", "échographie",
        "biologie", "génétique", "adn", "arn", "protéine",
        "intelligence artificielle médicale", "diagnostic assisté",
    ],
    "éducation": [
        "apprentissage", "pédagogie", "formation", "cours", "tutoriel",
        "e-learning", "mooc", "compétence", "certification",
        "auto-formation", "pratique délibérée", "mémorisation",
        "feynman", "mind map", "flashcard", "spaced repetition",
    ],
    # ── Architecture logicielle ──────────────────────────────────────────────
    "architecture": [
        "microservice", "monolith", "serverless", "event-driven", "cqrs",
        "rest", "graphql", "websocket", "grpc", "message broker",
        "mvc", "mvvm", "clean architecture", "hexagonal", "ddd",
        "entity", "repository", "service", "controller", "dto",
        "migration", "seed", "middleware", "interceptor",
        "circuit breaker", "rate limiting", "caching", "pagination",
        "unit test", "integration test", "tdd", "bdd", "e2e",
    ],
    "projet": [
        "backend", "frontend", "mobile", "admin", "dashboard", "api",
        "database", "monorepo", "cli", "library", "sdk", "plugin",
        "mvp", "prototype", "production", "staging", "roadmap",
        "sprint", "kanban", "scrum", "agile", "jira", "trello",
        "smartbudget africa", "makenbrain", "intia assurance",
    ],
}

# Enrichissement personnalisé — brain_data/custom_domain.json
_custom = Path("brain_data/custom_domain.json")
if _custom.exists():
    try:
        _extra = json.loads(_custom.read_text(encoding="utf-8"))
        for _cat, _terms in _extra.items():
            DOMAIN.setdefault(_cat, []).extend(_terms)
    except Exception:
        pass

ALL_KNOWN = {c: t for t, concepts in DOMAIN.items() for c in concepts}


def _normalize(text: str) -> str:
    return text.lower().strip()


def extract_fast(text: str) -> list[dict]:
    text_lower = text.lower()
    found: dict[str, dict] = {}
    for concept, ctype in ALL_KNOWN.items():
        if concept in text_lower:
            found[concept] = {"name": concept, "type": ctype, "confidence": 0.9}
    for term in re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', text):
        key = term.lower()
        if key not in found and len(term) > 5:
            found[key] = {"name": key, "type": "technique", "confidence": 0.75}
    skip = {"I","A","OK","GET","PUT","SET","AS","IN","IS","OR","ON","TO","DO","IF","AND","THE","NOT"}
    for acr in re.findall(r'\b([A-Z]{2,6})\b', text):
        key = acr.lower()
        if acr not in skip and key not in found:
            found[key] = {"name": key, "type": "acronyme", "confidence": 0.65}
    return list(found.values())[:20]


async def extract_llm(text: str) -> list[dict]:
    prompt = (
        "Extrais les concepts techniques et métier de ce texte.\n"
        "Réponds UNIQUEMENT en JSON valide.\n"
        'Format: {"concepts": [{"name": "...", "type": "...", "relations": ["..."]}]}\n\n'
        f"Texte:\n{text[:600]}\n\nJSON:"
    )
    try:
        if settings.GROQ_API_KEY:
            from groq import Groq as GroqClient
            client = GroqClient(api_key=settings.GROQ_API_KEY)
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=512,
            )
            raw = resp.choices[0].message.content
        else:
            import ollama as _ollama
            raw = _ollama.generate(model=settings.OLLAMA_MODEL, prompt=prompt,
                                   options={"temperature": 0.1}).get("response", "")
        raw = re.sub(r"```(?:json)?", "", raw.strip()).strip().rstrip("`").strip()
        return json.loads(raw).get("concepts", [])
    except Exception:
        return extract_fast(text)


async def extract_and_graph(text: str, memory_id: str = "", fast: bool = True) -> dict:
    from core.graph import neuron_graph
    concepts = extract_fast(text) if fast else await extract_llm(text)
    added = []
    for c in concepts:
        name = _normalize(c.get("name", ""))
        if not name or len(name) < 3:
            continue
        neuron_graph.add_concept(name, c.get("type", "concept"), memory_id)
        added.append(name)
    for i, a in enumerate(added):
        for b in added[i+1:]:
            neuron_graph.add_connection(a, b, "co-occurs", 0.6)
    if not fast:
        for c in concepts:
            name = _normalize(c.get("name", ""))
            for rel in c.get("relations", []):
                rel = _normalize(rel)
                if rel and rel != name and len(rel) > 2:
                    neuron_graph.add_connection(name, rel, "relates_to", 0.8)
    neuron_graph.save()
    return {"concepts_added": len(added), "concepts": added[:10],
            "graph_nodes": neuron_graph.G.number_of_nodes(),
            "graph_edges": neuron_graph.G.number_of_edges()}
