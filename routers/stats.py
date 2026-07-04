"""Statistiques d'utilisation — tableau de bord SuperAdmin (Phase 9).

Agrège trois sources :
- core.usage       : messages/jour, utilisateurs actifs, top users (local)
- core.chat_history: totaux conversations/messages (local)
- Supabase users   : inscriptions et rôles (best-effort — le tableau de
  bord fonctionne même quand le réseau vers Supabase est dégradé)

Réservé au SuperAdmin (JWT app_metadata ou X-API-Key admin).
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from core.auth import require_chat_user
from core.chat_history import history_stats
from core.usage import get_plans, get_user_plans, set_user_plan, update_plan, usage_stats
from models.user import User, UserRole

router = APIRouter()
logger = logging.getLogger("makenbrain.stats")


def _require_superadmin(user: User) -> None:
    if user.role != UserRole.SUPERADMIN:
        raise HTTPException(status_code=403, detail="Accès réservé au SuperAdmin.")


@router.get("/stats")
async def admin_stats(
    days: int = 7,
    current_user: User = Depends(require_chat_user),
):
    """Statistiques globales d'utilisation (SuperAdmin uniquement)."""
    _require_superadmin(current_user)

    usage = usage_stats(days=max(1, min(days, 30)))
    history = history_stats()

    # Inscriptions Supabase — best-effort : le reste du tableau de bord
    # reste disponible même si Supabase est injoignable.
    users_info = None
    try:
        from core.user_service import UserService

        users = UserService().list_users()
        by_role: dict[str, int] = {}
        for u in users:
            role = u.role.value if hasattr(u.role, "value") else str(u.role)
            by_role[role] = by_role.get(role, 0) + 1
        users_info = {"total": len(users), "by_role": by_role}
    except Exception as exc:
        logger.warning("Stats utilisateurs Supabase indisponibles : %s", exc)

    return {"usage": usage, "history": history, "users": users_info}


# ── Gestion des plans & tarifs (vente de tokens — gérée par le SuperAdmin) ────

class PlanUpdate(BaseModel):
    """Modification d'un plan : limite journalière et/ou prix mensuel ($)."""
    daily_messages: int | None = Field(default=None, ge=0)
    price_usd: float | None = Field(default=None, ge=0)


class UserPlanRequest(BaseModel):
    plan: str = Field(min_length=1)


@router.get("/plans")
async def list_plans(current_user: User = Depends(require_chat_user)):
    """Plans effectifs (limites + prix) et attributions par utilisateur."""
    _require_superadmin(current_user)
    return {"plans": get_plans(), "user_plans": get_user_plans()}


@router.put("/plans/{plan_name}")
async def edit_plan(
    plan_name: str,
    req: PlanUpdate,
    current_user: User = Depends(require_chat_user),
):
    """Modifie la limite journalière et/ou le prix d'un plan (SuperAdmin)."""
    _require_superadmin(current_user)
    if plan_name not in get_plans():
        raise HTTPException(status_code=404, detail=f"Plan inconnu : {plan_name}")
    changes = req.model_dump(exclude_unset=True)
    try:
        updated = update_plan(plan_name, changes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"plan": plan_name, "config": updated}


@router.put("/users/{user_id}/plan")
async def assign_user_plan(
    user_id: str,
    req: UserPlanRequest,
    current_user: User = Depends(require_chat_user),
):
    """Attribue un plan à un utilisateur (ex. après une vente de tokens)."""
    _require_superadmin(current_user)
    try:
        set_user_plan(user_id, req.plan)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"user_id": user_id, "plan": req.plan, "message": "Plan attribué."}
