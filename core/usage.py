"""Comptage d'usage et quotas par utilisateur — Phase 9 (fondations SaaS).

Stockage local (brain_data/usage.json) : aucun aller-retour Supabase sur le
chemin critique du chat — les quotas fonctionnent même quand le réseau vers
Supabase est dégradé. Structure du fichier :

    { "2026-07-04": { "<user_id>": {"messages": 12} }, ... }

Les plans définissent les limites journalières. Le rôle détermine le plan
pour l'instant (SuperAdmin/Admin = illimité, User = free) ; un champ `plan`
en base prendra le relais quand la facturation existera.
"""
from __future__ import annotations

import json
import threading
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from models.user import User, UserRole

USAGE_FILE = Path("brain_data/usage.json")
PLANS_FILE = Path("brain_data/plans.json")            # surcharges éditées par le SuperAdmin
USER_PLANS_FILE = Path("brain_data/user_plans.json")  # attributions user_id → plan
_RETENTION_DAYS = 30

# ── Plans par défaut ──────────────────────────────────────────────────────────
# daily_messages = None → illimité. price_usd = prix mensuel en dollars.
# Le SuperAdmin modifie limites et prix via PUT /admin/plans/{nom} — les
# surcharges sont persistées dans brain_data/plans.json.
PLANS: dict[str, dict[str, Any]] = {
    "free":      {"daily_messages": 50,   "price_usd": 0.0},
    "pro":       {"daily_messages": 500,  "price_usd": 5.0},
    "unlimited": {"daily_messages": None, "price_usd": None},
}

_lock = threading.Lock()


def _read_json(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def get_plans() -> dict[str, dict[str, Any]]:
    """Plans effectifs : défauts fusionnés avec les surcharges du SuperAdmin."""
    plans = {name: dict(cfg) for name, cfg in PLANS.items()}
    for name, cfg in _read_json(PLANS_FILE).items():
        if isinstance(cfg, dict):
            plans.setdefault(name, {}).update(cfg)
    return plans


def update_plan(name: str, changes: dict[str, Any]) -> dict[str, Any]:
    """Persiste une modification de plan (limite journalière et/ou prix).

    Seules les clés connues sont acceptées. Retourne le plan effectif.
    """
    allowed = {k: v for k, v in changes.items() if k in ("daily_messages", "price_usd")}
    if not allowed:
        raise ValueError("Aucun champ modifiable fourni (daily_messages, price_usd).")
    with _lock:
        overrides = _read_json(PLANS_FILE)
        overrides.setdefault(name, {}).update(allowed)
        _write_json(PLANS_FILE, overrides)
    return get_plans().get(name, {})


def get_user_plans() -> dict[str, str]:
    """Attributions explicites user_id → plan (gérées par le SuperAdmin)."""
    return {str(k): str(v) for k, v in _read_json(USER_PLANS_FILE).items()}


def set_user_plan(user_id: str, plan: str) -> None:
    """Attribue un plan à un utilisateur (ex. après une vente de tokens)."""
    if plan not in get_plans():
        raise ValueError(f"Plan inconnu : {plan}")
    with _lock:
        assignments = _read_json(USER_PLANS_FILE)
        assignments[str(user_id)] = plan
        _write_json(USER_PLANS_FILE, assignments)


def plan_for_user(user: User) -> str:
    """Retourne le plan applicable à un utilisateur.

    SuperAdmin et Admin : illimité. Sinon : le plan attribué par le
    SuperAdmin (vente de tokens), à défaut le plan free.
    """
    if user.role in (UserRole.SUPERADMIN, UserRole.ADMIN):
        return "unlimited"
    return get_user_plans().get(str(user.id), "free")


def _load() -> dict[str, Any]:
    if USAGE_FILE.exists():
        try:
            return json.loads(USAGE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save(data: dict[str, Any]) -> None:
    USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    USAGE_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _prune(data: dict[str, Any]) -> None:
    """Supprime les jours plus vieux que la rétention (fichier compact)."""
    cutoff = (date.today() - timedelta(days=_RETENTION_DAYS)).isoformat()
    for day in [d for d in data if d < cutoff]:
        del data[day]


def record_message(user_id: str) -> int:
    """Comptabilise un message pour l'utilisateur et retourne son total du jour."""
    today = date.today().isoformat()
    with _lock:
        data = _load()
        day = data.setdefault(today, {})
        entry = day.setdefault(str(user_id), {"messages": 0})
        entry["messages"] += 1
        _prune(data)
        _save(data)
        return entry["messages"]


def messages_today(user_id: str) -> int:
    """Retourne le nombre de messages envoyés aujourd'hui par l'utilisateur."""
    data = _load()
    return int(((data.get(date.today().isoformat()) or {}).get(str(user_id)) or {}).get("messages", 0))


def enforce_quota(user: User) -> None:
    """Garde de quota : comptabilise le message et refuse au-delà de la limite.

    Lève HTTPException 429 avec un message clair quand la limite journalière
    du plan est atteinte. Le comptage est fait ici (un seul point d'entrée).
    """
    plans = get_plans()
    plan_name = plan_for_user(user)
    limit = plans.get(plan_name, plans["free"]).get("daily_messages")
    used = record_message(str(user.id))
    if limit is not None and used > limit:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Quota journalier atteint ({limit} messages/jour, plan {plan_name}). "
                "Réessaie demain ou passe à un plan supérieur."
            ),
        )


def usage_stats(days: int = 7) -> dict[str, Any]:
    """Statistiques d'usage pour le tableau de bord SuperAdmin.

    Retourne la série des messages par jour, les totaux et le top des
    utilisateurs sur la période. Aucune donnée sensible : uniquement des
    identifiants et des compteurs.
    """
    data = _load()
    series: list[dict[str, Any]] = []
    per_user: dict[str, int] = {}
    active_users_today = 0
    today = date.today()

    for offset in range(days - 1, -1, -1):
        day = (today - timedelta(days=offset)).isoformat()
        day_data = data.get(day) or {}
        day_total = sum(int(u.get("messages", 0)) for u in day_data.values())
        series.append({"date": day, "messages": day_total, "users": len(day_data)})
        for uid, u in day_data.items():
            per_user[uid] = per_user.get(uid, 0) + int(u.get("messages", 0))
        if day == today.isoformat():
            active_users_today = len(day_data)

    top_users = sorted(per_user.items(), key=lambda x: x[1], reverse=True)[:10]
    return {
        "days": days,
        "series": series,
        "messages_total": sum(d["messages"] for d in series),
        "active_users_today": active_users_today,
        "active_users_period": len(per_user),
        "top_users": [{"user_id": uid, "messages": count} for uid, count in top_users],
        "plans": get_plans(),
    }
