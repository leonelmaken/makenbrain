"""Pydantic models for the MakenBrain user business layer."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class UserRole(str, Enum):
    """Allowed application roles for a MakenBrain user."""

    ADMIN = "admin"
    USER = "user"


class UserProfile(BaseModel):
    """Optional profile data carried by the application layer."""

    full_name: str | None = Field(default=None, min_length=1)
    avatar_url: str | None = None
    preferences: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class UserBase(BaseModel):
    """Shared user fields accepted by create and read operations."""

    email: str | None = Field(default=None, min_length=3)
    name: str | None = Field(default=None, min_length=1)
    role: UserRole = UserRole.USER
    profile: UserProfile | None = None

    @field_validator("email")
    @classmethod
    def validate_email_shape(cls, value: str | None) -> str | None:
        """Reject obvious invalid emails without adding an extra dependency."""
        if value is None:
            return value
        if "@" not in value or "." not in value.rsplit("@", maxsplit=1)[-1]:
            raise ValueError("email must look like a valid email address")
        return value


class UserCreate(UserBase):
    """Payload used to create a user row in Supabase."""

    id: str | None = Field(default=None, min_length=1)


class UserUpdate(BaseModel):
    """Payload used to update mutable user fields."""

    email: str | None = Field(default=None, min_length=3)
    name: str | None = Field(default=None, min_length=1)
    role: UserRole | None = None
    profile: UserProfile | None = None

    @field_validator("email")
    @classmethod
    def validate_email_shape(cls, value: str | None) -> str | None:
        """Reject obvious invalid emails without adding an extra dependency."""
        if value is None:
            return value
        if "@" not in value or "." not in value.rsplit("@", maxsplit=1)[-1]:
            raise ValueError("email must look like a valid email address")
        return value


class AuthUserIdentity(BaseModel):
    """Minimal Supabase Auth identity used to sync the application profile."""

    id: str = Field(min_length=1)
    email: str | None = Field(default=None, min_length=3)
    name: str | None = Field(default=None, min_length=1)
    profile: UserProfile | None = None

    @field_validator("email")
    @classmethod
    def validate_email_shape(cls, value: str | None) -> str | None:
        """Reject obvious invalid emails without adding an extra dependency."""
        if value is None:
            return value
        if "@" not in value or "." not in value.rsplit("@", maxsplit=1)[-1]:
            raise ValueError("email must look like a valid email address")
        return value


class User(UserBase):
    """User row returned by Supabase and exposed to the application."""

    model_config = ConfigDict(extra="allow")

    id: str
    created_at: datetime | None = None
    updated_at: datetime | None = None

