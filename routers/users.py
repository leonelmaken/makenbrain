"""REST API for Supabase-backed application users."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from core.auth import require_supabase_user
from core.user_service import UserNotFoundError, UserService, UserServiceError
from models.user import User, UserRole, UserUpdate

router = APIRouter()


def get_user_service() -> UserService:
    """Return the user business service used by the API layer."""
    return UserService()


def _is_admin(user: User) -> bool:
    """Return whether the authenticated profile has admin privileges."""
    return user.role == UserRole.ADMIN


def _ensure_admin(user: User) -> None:
    """Reject callers that are not allowed to perform admin-only actions."""
    if not _is_admin(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Action reservee aux administrateurs.",
        )


def _ensure_self_or_admin(current_user: User, target_user_id: str) -> None:
    """Allow users to access their own profile, or admins to access any profile."""
    if current_user.id != target_user_id and not _is_admin(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Acces interdit a ce profil utilisateur.",
        )


def _service_error_to_http(exc: Exception) -> HTTPException:
    """Translate user service exceptions into stable HTTP responses."""
    if isinstance(exc, UserNotFoundError):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Operation utilisateur impossible : {exc}",
    )


@router.get("/me", response_model=User)
async def get_me(current_user: User = Depends(require_supabase_user)) -> User:
    """Return the authenticated user's synchronized profile."""
    return current_user


@router.get("/users/{user_id}", response_model=User)
async def get_user(
    user_id: str,
    current_user: User = Depends(require_supabase_user),
    user_service: UserService = Depends(get_user_service),
) -> User:
    """Return one profile when the caller owns it or has admin privileges."""
    _ensure_self_or_admin(current_user, user_id)
    try:
        return user_service.get_user(user_id)
    except (UserServiceError, ValueError) as exc:
        raise _service_error_to_http(exc) from exc


@router.get("/users", response_model=list[User])
async def list_users(
    limit: int = 100,
    current_user: User = Depends(require_supabase_user),
    user_service: UserService = Depends(get_user_service),
) -> list[User]:
    """Return users visible to administrators."""
    _ensure_admin(current_user)
    try:
        return user_service.list_users(limit=limit)
    except (UserServiceError, ValueError) as exc:
        raise _service_error_to_http(exc) from exc


@router.patch("/users/{user_id}", response_model=User)
async def update_user(
    user_id: str,
    updates: UserUpdate,
    current_user: User = Depends(require_supabase_user),
    user_service: UserService = Depends(get_user_service),
) -> User:
    """Update one profile, with role/email changes reserved for admins."""
    _ensure_self_or_admin(current_user, user_id)
    if not _is_admin(current_user) and (updates.role is not None or updates.email is not None):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Seul un administrateur peut modifier l'email ou le role.",
        )

    try:
        return user_service.update_user(user_id, updates)
    except (UserServiceError, ValueError) as exc:
        raise _service_error_to_http(exc) from exc


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: str,
    current_user: User = Depends(require_supabase_user),
    user_service: UserService = Depends(get_user_service),
) -> None:
    """Delete a user profile. This is an admin-only operation."""
    _ensure_admin(current_user)
    try:
        user_service.delete_user(user_id)
    except (UserServiceError, ValueError) as exc:
        raise _service_error_to_http(exc) from exc
