"""DTO Pydantic pour les memoires utilisateur.

Ce module appartient strictement a la couche ``models``: il ne contient
aucune logique metier, aucun acces Supabase et aucun import vers ``core``.
Les classes ci-dessous servent uniquement a valider, normaliser et
serialiser les donnees qui circulent entre les services et les routes API.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class UserMemorySource(str, Enum):
    """Origines fonctionnelles connues pour une memoire utilisateur.

    L'enum fournit des constantes lisibles aux services, tout en laissant
    les DTO stocker la valeur finale comme une chaine simple dans les
    payloads JSON envoyes a Supabase.
    """

    MANUAL = "manual"
    CHAT = "chat"
    PROFILE = "profile"
    IMPORT = "import"
    SYSTEM = "system"


class UserMemoryBase(BaseModel):
    """Champs communs d'une memoire utilisateur.

    Represente une information durable associee a un utilisateur
    Supabase. Cette classe sert uniquement de DTO pour validation et
    serialisation; elle ne doit jamais porter de logique metier.
    """

    user_id: UUID = Field(description="Identifiant Supabase du proprietaire de la memoire.")
    content: str = Field(min_length=1, description="Contenu textuel de la memoire.")
    source: str = Field(default=UserMemorySource.MANUAL.value, min_length=1, max_length=64)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        """Refuse les contenus vides apres normalisation des espaces."""
        content = value.strip()
        if not content:
            raise ValueError("content must not be blank")
        return content

    @field_validator("source", mode="before")
    @classmethod
    def normalize_source(cls, value: UserMemorySource | str) -> str:
        """Convertit les constantes d'origine en chaines JSON stables."""
        if isinstance(value, UserMemorySource):
            value = value.value
        source = str(value).strip()
        if not source:
            raise ValueError("source must not be blank")
        return source


class UserMemoryCreate(UserMemoryBase):
    """Payload de creation d'une memoire utilisateur.

    Utilise par la couche service avant insertion Supabase afin de
    garantir que chaque ligne creee possede un proprietaire, un contenu,
    une origine et des metadonnees JSON normalisees.
    """


class UserMemoryUpdate(BaseModel):
    """Payload de mise a jour partielle d'une memoire utilisateur.

    Tous les champs sont optionnels pour permettre les PATCH fonctionnels.
    Le service retire ensuite les valeurs non definies avant d'envoyer la
    mise a jour a Supabase.
    """

    content: str | None = Field(default=None, min_length=1)
    source: str | None = Field(default=None, min_length=1, max_length=64)
    metadata: dict[str, Any] | None = None

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str | None) -> str | None:
        """Refuse les contenus explicitement fournis mais vides."""
        if value is None:
            return value
        content = value.strip()
        if not content:
            raise ValueError("content must not be blank")
        return content

    @field_validator("source", mode="before")
    @classmethod
    def normalize_source(cls, value: UserMemorySource | str | None) -> str | None:
        """Normalise l'origine lors d'une mise a jour partielle."""
        if value is None:
            return value
        if isinstance(value, UserMemorySource):
            value = value.value
        source = str(value).strip()
        if not source:
            raise ValueError("source must not be blank")
        return source


class UserMemory(UserMemoryBase):
    """Memoire utilisateur retournee par Supabase.

    Represente une ligne persistee dans ``user_memories`` et exposee a la
    couche API. Les champs techniques de persistance sont presents ici,
    mais la classe reste un DTO pur sans dependance vers les services.
    """

    model_config = ConfigDict(extra="allow")

    id: UUID
    created_at: datetime | None = None
    updated_at: datetime | None = None
