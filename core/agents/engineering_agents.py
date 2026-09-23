"""Équipe d'agents d'ingénierie logicielle — Phase 10.

Trois spécialistes couvrant le cycle de développement complet du SuperAdmin
(full-stack software engineer) :

    ArchitectAgent : conception et architecture (choix techniques, trade-offs).
    CodingAgent    : implémentation, debug, refactoring (code prêt à l'emploi).
    TestAgent      : stratégie de test, cas limites, tests exécutables.

Tous respectent le contrat BaseAgent : jamais d'exception propagée,
duration_ms toujours renseigné, aucune communication directe inter-agents.
Le LLM est appelé via le LLMRouter (Groq 70B → repli Ollama local, où les
questions de code sont routées vers les modèles codeurs installés).

Autonomie : SUGGEST — ces agents produisent des livrables (architecture,
code, tests) mais n'exécutent rien ; l'humain décide de l'application.
"""
from __future__ import annotations

import time

from core.agents.base import AgentAutonomy, BaseAgent
from core.agents.models import AgentResult, AgentTask, ExecutionContext
from core.provider_layer.router import ALL_PROVIDERS_FAILED, get_router


class _EngineeringAgent(BaseAgent):
    """Base commune des agents d'ingénierie : appel LLM + gestion d'échec.

    Les sous-classes ne définissent que leur identité (name, description,
    capabilities) et leur prompt système — toute la mécanique vit ici.
    """

    autonomy            : AgentAutonomy = AgentAutonomy.SUGGEST
    version             : str           = "1.0.0"
    cost_per_call       : float         = 1.0
    confidence_threshold: float         = 0.75

    # À surcharger dans chaque sous-classe.
    system_prompt: str = ""

    async def run(self, task: AgentTask, ctx: ExecutionContext) -> AgentResult:
        """Génère le livrable de l'agent via le LLMRouter.

        Le contexte partagé des agents précédents (ex. plan du
        PlanningAgent, souvenirs du MemoryAgent) est injecté si présent,
        pour que l'équipe travaille comme une chaîne et non en silos.
        """
        t0 = time.monotonic()
        try:
            context = self._build_context(task, ctx)
            output = await get_router().generate(
                prompt       = task.input,
                context      = context,
                system_prompt= self.system_prompt,
            )
            if not output or output == ALL_PROVIDERS_FAILED:
                return self._timed_result(
                    task, t0,
                    success= False,
                    error  = "Aucun provider LLM disponible (Groq et Ollama injoignables).",
                )
            ctx.shared[f"{self.name}_output"] = output
            return self._timed_result(
                task, t0,
                success    = True,
                output     = output,
                confidence = self.confidence_threshold,
                metadata   = {"context_used": bool(context)},
            )
        except Exception as exc:  # noqa: BLE001 — contrat BaseAgent : ne jamais lever.
            return self._timed_result(task, t0, success=False, error=str(exc))

    def _build_context(self, task: AgentTask, ctx: ExecutionContext) -> str:
        """Assemble le contexte disponible : task.context + sorties partagées."""
        parts: list[str] = []
        extra = task.context.get("context") or task.context.get("code")
        if extra:
            parts.append(str(extra)[:4000])
        for key, value in ctx.shared.items():
            if key.endswith("_output") and isinstance(value, str) and value:
                parts.append(f"[Travail de {key.removesuffix('_output')}]\n{value[:2000]}")
        return "\n\n".join(parts)


class ArchitectAgent(_EngineeringAgent):
    """Conception et architecture logicielle — décisions techniques argumentées."""

    name        = "architect_agent"
    description = (
        "Architecte logiciel senior : conçoit des architectures, compare les "
        "options techniques avec leurs trade-offs, structure les systèmes "
        "(API, données, découpage en modules) avant toute ligne de code."
    )
    capabilities = ["architecture", "conception", "design", "system_design"]

    system_prompt = """Tu es un architecte logiciel senior (15+ ans, systèmes distribués, web full-stack, cloud).

Ta mission : produire des décisions d'architecture claires et défendables.

Règles :
- Commence par reformuler le besoin en une phrase, puis structure ta réponse en markdown (## sections).
- Pour chaque choix technique important, donne 2-3 options avec un tableau avantages/inconvénients, puis TRANCHE avec une recommandation justifiée.
- Décris le découpage : composants, responsabilités, flux de données, contrats d'API.
- Inclus un diagramme Mermaid (```mermaid) quand la structure le mérite.
- Pense production : sécurité, coûts, montée en charge, simplicité d'exploitation.
- Sois direct et critique : si la demande contient une mauvaise idée, dis-le et propose mieux.
- Réponds en français (sauf question en anglais)."""


class CodingAgent(_EngineeringAgent):
    """Implémentation, debug et refactoring — code de qualité production."""

    name        = "coding_agent"
    description = (
        "Ingénieur full-stack senior : écrit du code prêt pour la production, "
        "débogue, refactore et explique. Python, JavaScript/TypeScript, SQL, "
        "et l'écosystème web moderne."
    )
    capabilities = ["coding", "code", "programming", "debug", "refactor", "implementation"]

    system_prompt = """Tu es un ingénieur logiciel full-stack senior. Ton code part en production.

Règles :
- Produis du code COMPLET et fonctionnel (jamais de `# ...` ou de TODO à la place de la logique).
- Blocs de code avec le langage annoté (```python, ```javascript…).
- Gestion d'erreurs réelle, cas limites couverts, nommage clair — pas de code jouet.
- Respecte les conventions du langage (PEP 8, ESLint) et commente uniquement le non-évident.
- Après le code : explication courte des choix + points de vigilance (sécurité, perf).
- Pour un bug : identifie la CAUSE RACINE avant de proposer le correctif, et explique-la.
- Pour un refactoring : préserve le comportement, liste ce qui change et pourquoi.
- Si la demande est ambiguë, choisis l'interprétation la plus probable et annonce-la en une ligne.
- Réponds en français, code et identifiants en anglais."""


class TestAgent(_EngineeringAgent):
    """Stratégie de test et tests exécutables — qualité indiscutable."""

    name        = "test_agent"
    description = (
        "Ingénieur qualité : conçoit des stratégies de test, écrit des tests "
        "exécutables (pytest, Jest) et traque les cas limites que tout le "
        "monde oublie."
    )
    capabilities = ["testing", "tests", "quality", "test_plan"]

    system_prompt = """Tu es un ingénieur test senior, obsédé par les cas limites.

Règles :
- Si on te donne du code : écris des tests EXÉCUTABLES (pytest pour Python, Jest/Vitest pour JS), avec arrange/act/assert lisibles.
- Couvre systématiquement : cas nominal, cas limites (vide, None/null, très grand, unicode, concurrence si pertinent), cas d'erreur attendus.
- Nomme chaque test d'après le comportement vérifié (test_refuse_montant_negatif, pas test_1).
- Si on te demande une stratégie : structure par niveaux (unitaire, intégration, e2e) avec les priorités et ce qu'il ne faut PAS tester.
- Signale les parties du code difficiles à tester et propose le refactoring minimal qui les rendrait testables.
- Termine par la liste des risques résiduels non couverts — l'honnêteté avant le vernis.
- Réponds en français, code en anglais."""
