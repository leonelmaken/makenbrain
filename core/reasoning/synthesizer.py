"""Synthesizer — Phase 3.5 du moteur de raisonnement expert.

Cinquième et dernière étape du pipeline. Reçoit le contexte enrichi
par toutes les étapes précédentes (QuestionAnalysis, Hypotheses, Evidence,
DecisionResult) et produit une réponse finale en langage naturel,
directement exploitable par l'utilisateur.

Rôle dans le pipeline :
    La DecisionEngine (Phase 3.4) choisit *quelle* hypothèse retenir et
    *pourquoi*, sous forme de structures de données internes.
    Le Synthesizer traduit ce raisonnement machine en prose humaine :
    une réponse fluide, professionnelle, calibrée sur le niveau de
    confiance et les risques détectés.

Stratégie d'exécution (dégradation gracieuse identique aux autres agents) :
    1. Tentative LLM avec prompt structuré (qualité optimale).
    2. Fallback template déterministe si LLM indisponible ou échoue.
    Le pipeline ne s'arrête jamais : un échec LLM déclenche le fallback.

Registre de prudence (SynthesisTone) :
    AFFIRMATIVE (confiance >= 0.75) : réponse directe, sans réserves inutiles.
    BALANCED    (confiance in [0.60, 0.75)) : nuances exprimées sobrement.
    CAUTIOUS    (confiance < 0.60) : limites explicites, vérification recommandée.

Note d'architecture pour les phases futures :
    - Le Synthesizer ne génère pas d'hypothèses et n'interroge pas la mémoire.
      Il consomme uniquement ce que les étapes précédentes ont produit dans ctx.
    - La fonction ``llm_generate`` est injectable (tests, future Skills layer).
    - La construction du prompt est isolée dans ``_build_llm_prompt`` pour
      permettre son remplacement par un Skill de prompt-engineering sans toucher
      à l'orchestration.

Dépendances autorisées :
    ✓ core.reasoning.context
    ✓ models.reasoning
    ✗ routers/  (jamais)
    ✗ core.reasoning.reasoning_engine  (évite le cycle)
    ✗ core.memory, core.user_service, etc.  (le Synthesizer consomme le contexte,
      il n'interroge pas les sources directement)
"""
from __future__ import annotations

import logging
import time
from typing import Awaitable, Callable, Optional

from models.reasoning import (
    Evidence,
    EvidenceSourceType,
    ReasoningStepRecord,
    SynthesisResult,
    SynthesisStrategy,
    SynthesisTone,
)
from core.reasoning.context import ReasoningContext

logger = logging.getLogger("makenbrain.reasoning.synthesizer")

# ── Types ─────────────────────────────────────────────────────────────────────

LLMCallable = Callable[..., Awaitable[str]]

# ── Seuils de ton (alignés sur les seuils risk_level du pipeline) ─────────────

_TONE_AFFIRMATIVE_THRESHOLD : float = 0.75
_TONE_BALANCED_THRESHOLD    : float = 0.60

# Nombre maximum de preuves citées dans le prompt LLM (équilibre contexte/tokens)
_MAX_EVIDENCE_IN_PROMPT : int = 4
# Longueur max d'un extrait de preuve dans le prompt (évite les prompts trop longs)
_EVIDENCE_EXCERPT_LENGTH : int = 220


# ── Agent (protocole ReasoningAgent) ─────────────────────────────────────────

class SynthesizerAgent:
    """Agent de synthèse finale — étape 5 du pipeline de raisonnement.

    Implémente le protocole ReasoningAgent (run(ctx) → ctx).

    Délègue à ``synthesize()`` pour rester testable indépendamment
    du protocole agent.
    """

    name: str = "synthesizer"

    async def run(self, ctx: ReasoningContext) -> ReasoningContext:
        """Exécute la synthèse et enrichit le contexte.

        Args:
            ctx: Contexte de raisonnement avec étapes 1-4 complétées.

        Returns:
            ctx avec ctx.final_answer et ctx.synthesis_result peuplés.
        """
        return await synthesize(ctx)


# ── Point d'entrée public ─────────────────────────────────────────────────────

async def synthesize(
    ctx          : ReasoningContext,
    llm_generate : Optional[LLMCallable] = None,
) -> ReasoningContext:
    """Produit la réponse finale en langage naturel et peuple ctx.final_answer.

    Fonction publique du module, testable sans instancier l'agent.

    Étapes :
        1. Calculer le ton adapté au score de confiance.
        2. Tenter la génération LLM (prompt structuré).
        3. En cas d'échec LLM, appliquer le fallback déterministe.
        4. Construire SynthesisResult (métadonnées d'observabilité).
        5. Alimenter ctx.trace avec un ReasoningStepRecord.

    Args:
        ctx          : Contexte de raisonnement courant.
        llm_generate : Fonction LLM injectable. Si None, importe core.llm.generate
                       au runtime (lazy import pour éviter les cycles).

    Returns:
        ctx enrichi : ctx.final_answer et ctx.synthesis_result peuplés.
    """
    step_start = time.monotonic()
    tone       = _select_tone(ctx.confidence)

    logger.info(
        "[Synthesizer] Synthèse | confiance=%.2f | ton=%s | dégradé=%s",
        ctx.confidence, tone.value, ctx.pipeline_degraded,
    )

    # ── Résoudre la fonction LLM (import lazy — évite les cycles) ─────────────
    _generate = llm_generate
    if _generate is None:
        try:
            from core.llm import generate as _core_generate  # noqa: PLC0415
            _generate = _core_generate
        except ImportError:
            logger.warning("[Synthesizer] core.llm introuvable — fallback template")

    # ── Tentative LLM ─────────────────────────────────────────────────────────
    answer   : Optional[str] = None
    strategy : SynthesisStrategy = SynthesisStrategy.TEMPLATE
    prompt_tokens : int = 0

    if _generate is not None:
        try:
            prompt        = _build_llm_prompt(ctx, tone)
            prompt_tokens = _estimate_tokens(prompt)
            raw           = await _generate(prompt)
            answer        = raw.strip() if raw and raw.strip() else None
            if answer:
                strategy = SynthesisStrategy.LLM
                logger.info("[Synthesizer] LLM OK | ~%d tokens prompt", prompt_tokens)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Synthesizer] LLM échoué (%s) — fallback template", exc)
            answer = None

    # ── Fallback déterministe ─────────────────────────────────────────────────
    if not answer:
        answer        = _template_synthesis(ctx, tone)
        strategy      = SynthesisStrategy.TEMPLATE
        prompt_tokens = 0
        logger.info("[Synthesizer] Template appliqué")

    # ── Construire les métadonnées ─────────────────────────────────────────────
    gaps  = ctx.evaluation.knowledge_gaps  if ctx.evaluation else []
    risks = ctx.decision.reason.residual_risks if ctx.decision else []

    synthesis_result = SynthesisResult(
        strategy                  = strategy,
        tone                      = tone,
        sources_cited             = _count_sources_cited(ctx),
        has_uncertainty_statement = tone != SynthesisTone.AFFIRMATIVE,
        has_gap_statement         = bool(gaps),
        has_risk_warning          = bool(risks) or tone == SynthesisTone.CAUTIOUS,
        llm_prompt_tokens         = prompt_tokens,
    )

    ctx.final_answer     = answer
    ctx.synthesis_result = synthesis_result

    # ── Trace ─────────────────────────────────────────────────────────────────
    duration_ms = (time.monotonic() - step_start) * 1000
    ctx.trace.steps.append(ReasoningStepRecord(
        step_name      = "synthesizer",
        duration_ms    = round(duration_ms, 2),
        success        = True,
        input_summary  = (
            f"confiance={ctx.confidence:.2f} | ton={tone.value} | "
            f"hypothèses={len(ctx.hypotheses)} | preuves={len(ctx.evidence)}"
        ),
        output_summary = (
            f"stratégie={strategy.value} | "
            f"longueur={len(answer)} chars | "
            f"gaps={len(gaps)} | risques={len(risks)}"
        ),
        degraded       = strategy == SynthesisStrategy.TEMPLATE,
    ))

    logger.info(
        "[Synthesizer] Terminé %.0fms | stratégie=%s | %d chars",
        duration_ms, strategy.value, len(answer),
    )
    return ctx


# ── Construction du prompt LLM ────────────────────────────────────────────────

def _build_llm_prompt(ctx: ReasoningContext, tone: SynthesisTone) -> str:
    """Construit le prompt envoyé au LLM pour la génération de la réponse.

    Le prompt présente toutes les données pertinentes de manière structurée
    et naturelle, sans exposer les noms de champs internes au modèle.

    Args:
        ctx  : Contexte complet du pipeline.
        tone : Registre de prudence à appliquer.

    Returns:
        Prompt en texte brut, optimisé pour les modèles d'instruction.
    """
    analysis       = ctx.analysis_or_fallback
    selected_text  = _get_selected_hypothesis_content(ctx)
    top_evidence   = _select_top_evidence(ctx)
    gaps           = ctx.evaluation.knowledge_gaps  if ctx.evaluation else []
    contradictions = ctx.evaluation.contradictions  if ctx.evaluation else []
    risks          = ctx.decision.reason.residual_risks if ctx.decision else []
    selection_reasons = (
        ctx.decision.reason.selection_reasons if ctx.decision else []
    )

    # ── Sections optionnelles ──────────────────────────────────────────────────
    evidence_section = ""
    if top_evidence:
        lines = [f"  • {ev.content[:_EVIDENCE_EXCERPT_LENGTH]}" for ev in top_evidence]
        evidence_section = "Éléments d'analyse disponibles :\n" + "\n".join(lines)

    reasons_section = ""
    if selection_reasons:
        lines = [f"  • {r}" for r in selection_reasons[:3]]
        reasons_section = "Raisons soutenant cette conclusion :\n" + "\n".join(lines)

    gaps_section = ""
    if gaps:
        gaps_section = (
            "Informations manquantes pour une réponse complète :\n"
            + "\n".join(f"  • {g}" for g in gaps[:3])
        )

    contradictions_note = (
        "Note : des points de vue divergents ont été identifiés sur ce sujet."
        if contradictions else ""
    )

    risks_section = ""
    if risks:
        risks_section = (
            "Points de vigilance :\n"
            + "\n".join(f"  • {r}" for r in risks[:2])
        )

    # ── Instruction de ton ─────────────────────────────────────────────────────
    tone_instruction = _tone_instruction(tone, ctx.confidence)

    # ── Assemblage ────────────────────────────────────────────────────────────
    sections = [
        s for s in [
            evidence_section,
            reasons_section,
            gaps_section,
            contradictions_note,
            risks_section,
        ] if s
    ]
    context_block = ("\n\n".join(sections) + "\n\n") if sections else ""

    prompt = (
        f"Tu es un assistant expert. Formule une réponse claire et naturelle "
        f"à la question suivante.\n\n"
        f"Question : {ctx.request.question}\n\n"
        f"Domaine : {analysis.domain} | "
        f"Type : {analysis.question_type.value}\n\n"
        f"Conclusion du raisonnement :\n{selected_text}\n\n"
        f"{context_block}"
        f"Niveau de confiance de l'analyse : {ctx.confidence:.0%}\n\n"
        f"Instructions :\n"
        f"{tone_instruction}\n"
        f"- Réponds directement à la question, en prose fluide et professionnelle.\n"
        f"- Ne mentionne pas de termes techniques internes "
        f"(\"hypothèse\", \"score\", \"pipeline\", \"confiance numérique\").\n"
        f"- Si des limites existent, mentionne-les sobrement en fin de réponse.\n"
        f"- Réponds en français."
    )
    return prompt


def _tone_instruction(tone: SynthesisTone, confidence: float) -> str:
    """Retourne l'instruction de ton adaptée au registre de prudence.

    Args:
        tone       : Registre calculé par _select_tone().
        confidence : Score de confiance (pour préciser dans le texte).

    Returns:
        Instruction de rédaction en une ou deux lignes.
    """
    if tone == SynthesisTone.AFFIRMATIVE:
        return (
            "- Réponds de manière directe et affirmative. "
            "L'analyse est solide — pas besoin de réserves excessives."
        )
    if tone == SynthesisTone.BALANCED:
        return (
            "- Exprime quelques nuances là où c'est pertinent. "
            "Indique que ta réponse est bien fondée mais que des incertitudes mineures subsistent."
        )
    # CAUTIOUS
    return (
        f"- L'analyse présente des limites importantes (confiance : {confidence:.0%}). "
        "Formule ta réponse avec prudence, mentionne clairement les incertitudes "
        "et recommande une vérification auprès de sources spécialisées."
    )


# ── Fallback déterministe ─────────────────────────────────────────────────────

def _template_synthesis(ctx: ReasoningContext, tone: SynthesisTone) -> str:
    """Construit une réponse structurée sans LLM.

    Produit une réponse en prose utilisable même quand aucun modèle
    de langage n'est disponible. La qualité est moindre que la version
    LLM mais la réponse reste informative et honnête.

    Args:
        ctx  : Contexte complet du pipeline.
        tone : Registre de prudence calculé.

    Returns:
        Réponse en langage naturel construite à partir de templates.
    """
    selected = _get_selected_hypothesis_content(ctx)
    analysis = ctx.analysis_or_fallback
    gaps     = ctx.evaluation.knowledge_gaps  if ctx.evaluation else []
    risks    = ctx.decision.reason.residual_risks if ctx.decision else []
    n_evidence = len(ctx.evidence)

    parts: list[str] = []

    # ── Corps principal — adapté au ton ───────────────────────────────────────
    if tone == SynthesisTone.AFFIRMATIVE:
        parts.append(f"D'après mon analyse, {_lowercase_first(selected)}.")
    elif tone == SynthesisTone.BALANCED:
        parts.append(
            f"Sur la base des informations disponibles, "
            f"il semble que {_lowercase_first(selected)}."
        )
    else:  # CAUTIOUS
        parts.append(
            f"Les données disponibles conduisent à l'hypothèse suivante, "
            f"bien que la confiance reste limitée : {_lowercase_first(selected)}."
        )

    # ── Appui sur les preuves ─────────────────────────────────────────────────
    if n_evidence > 0:
        label = "élément" if n_evidence == 1 else "éléments"
        parts.append(
            f"Cette conclusion s'appuie sur {n_evidence} {label} d'analyse."
        )

    # ── Type de question (contexte) ───────────────────────────────────────────
    qtype_labels = {
        "procedural"   : "Il s'agit d'une question de type procédural.",
        "comparative"  : "Cette analyse compare plusieurs approches.",
        "analytical"   : "Cette question nécessitait une analyse de causalité.",
        "evaluative"   : "Cette question demandait une évaluation comparative.",
        "hypothetical" : "Il s'agit d'une question hypothétique.",
    }
    qtype_note = qtype_labels.get(analysis.question_type.value, "")
    if qtype_note:
        parts.append(qtype_note)

    # ── Lacunes de connaissance ───────────────────────────────────────────────
    if gaps:
        if len(gaps) == 1:
            parts.append(f"Limite identifiée : {gaps[0]}.")
        else:
            joined = " ; ".join(gaps[:3])
            parts.append(f"Limites identifiées : {joined}.")

    # ── Risques résiduels / avertissement ────────────────────────────────────
    if risks and tone in (SynthesisTone.BALANCED, SynthesisTone.CAUTIOUS):
        parts.append(risks[0])

    if tone == SynthesisTone.CAUTIOUS:
        parts.append(
            "Une vérification auprès de sources spécialisées est recommandée "
            "avant toute prise de décision."
        )

    return " ".join(parts)


# ── Helpers privés ────────────────────────────────────────────────────────────

def _select_tone(confidence: float) -> SynthesisTone:
    """Calcule le registre de prudence à partir du score de confiance global.

    Args:
        confidence : Score de confiance [0.0, 1.0].

    Returns:
        SynthesisTone adapté.
    """
    if confidence >= _TONE_AFFIRMATIVE_THRESHOLD:
        return SynthesisTone.AFFIRMATIVE
    if confidence >= _TONE_BALANCED_THRESHOLD:
        return SynthesisTone.BALANCED
    return SynthesisTone.CAUTIOUS


def _get_selected_hypothesis_content(ctx: ReasoningContext) -> str:
    """Extrait le contenu textuel de l'hypothèse sélectionnée.

    Priorise dans l'ordre :
    1. ctx.decision.selected_hypothesis_id (source de vérité Phase 3.4).
    2. Première hypothèse de la liste (fallback si décision absente).
    3. Texte d'absence si aucune hypothèse n'est disponible.

    Args:
        ctx : Contexte du pipeline.

    Returns:
        Contenu textuel de l'hypothèse sélectionnée.
    """
    if not ctx.hypotheses:
        return "Aucune conclusion n'a pu être établie faute d'hypothèses."

    if ctx.decision and ctx.decision.selected_hypothesis_id:
        for h in ctx.hypotheses:
            if h.hypothesis_id == ctx.decision.selected_hypothesis_id:
                return h.content

    return ctx.hypotheses[0].content


def _select_top_evidence(ctx: ReasoningContext) -> list[Evidence]:
    """Sélectionne les preuves les plus pertinentes pour le prompt LLM.

    Tri par score de pertinence décroissant. Privilégie les preuves
    USER_MEMORY (contexte personnel) puis NEURON_GRAPH (cohérence
    structurelle) avant les preuves génériques.

    Args:
        ctx : Contexte du pipeline.

    Returns:
        Liste de preuves triées, limitée à _MAX_EVIDENCE_IN_PROMPT.
    """
    if not ctx.evidence:
        return []

    def _sort_key(ev: Evidence) -> tuple:
        priority = (
            0 if ev.source_type == EvidenceSourceType.USER_MEMORY
            else 1 if ev.source_type == EvidenceSourceType.NEURON_GRAPH
            else 2
        )
        return (priority, -ev.relevance_score)

    sorted_ev = sorted(ctx.evidence, key=_sort_key)
    return sorted_ev[:_MAX_EVIDENCE_IN_PROMPT]


def _count_sources_cited(ctx: ReasoningContext) -> int:
    """Compte les types de sources distincts présents dans les preuves.

    Args:
        ctx : Contexte du pipeline.

    Returns:
        Nombre de types de sources distincts (0 si aucune preuve).
    """
    if not ctx.evidence:
        return 0
    return len({ev.source_type for ev in ctx.evidence})


def _estimate_tokens(text: str) -> int:
    """Estimation grossière du nombre de tokens d'un texte.

    Approximation : 1 token ≈ 4 caractères (valable pour le français).
    Utilisé uniquement pour l'observabilité — pas pour le billing.

    Args:
        text : Texte à estimer.

    Returns:
        Estimation du nombre de tokens.
    """
    return max(1, len(text) // 4)


def _lowercase_first(text: str) -> str:
    """Met en minuscule le premier caractère d'une chaîne.

    Permet d'insérer le contenu d'une hypothèse dans une phrase
    sans majuscule redondante.

    Args:
        text : Texte source.

    Returns:
        Texte avec premier caractère en minuscule.
    """
    if not text:
        return text
    return text[0].lower() + text[1:]
