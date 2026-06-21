"""Pydantic models for user-scoped memories stored in Supabase."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class UserMemorySource(str, Enum):
    """Allowed sources for user memory entries."""

    CHAT = "chat"
    SYSTEM = "system"
    UPLOAD = "upload"
    EXTERNAL = "external"


class UserMemoryBase(BaseModel):
    """Shared fields for user memory creation and reads."""

    user_id: UUID
    content: str = Field(min_length=1)
    source: UserMemorySource
    metadata: dict[str, Any] = Field(default_factory=dict)


class UserMemoryCreate(UserMemoryBase):
    """Payload used to create a user memory row."""


class UserMemoryUpdate(BaseModel):
    """Payload used to update mutable memory fields."""

    content: str | None = Field(default=None, min_length=1)
    source: UserMemorySource | None = None
    metadata: dict[str, Any] | None = None


class UserMemory(UserMemoryBase):
    """User memory row returned by Supabase and exposed by the service."""

    model_config = ConfigDict(extra="allow")

    id: UUID
    created_at: datetime
    updated_at: datetime
