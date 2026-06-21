"""Tests for the Supabase-backed user business service."""
from __future__ import annotations

import pytest

from core.user_service import USERS_TABLE, UserNotFoundError, UserService
from models.user import UserCreate, UserRole, UserUpdate


class FakeResponse:
    """Small response object matching the Supabase SDK `.data` contract."""

    def __init__(self, data):
        self.data = data


class FakeUsersTable:
    """In-memory query builder for the subset used by UserService."""

    def __init__(self, rows: dict[str, dict]):
        self.rows = rows
        self.operation = ""
        self.payload = None
        self.filters: dict[str, str] = {}
        self.limit_value: int | None = None

    def insert(self, payload: dict):
        self.operation = "insert"
        self.payload = payload
        return self

    def select(self, _columns: str):
        self.operation = "select"
        return self

    def update(self, payload: dict):
        self.operation = "update"
        self.payload = payload
        return self

    def delete(self):
        self.operation = "delete"
        return self

    def eq(self, field: str, value: str):
        self.filters[field] = value
        return self

    def limit(self, value: int):
        self.limit_value = value
        return self

    def execute(self) -> FakeResponse:
        if self.operation == "insert":
            row = {"id": self.payload.get("id", "generated-id"), **self.payload}
            self.rows[row["id"]] = row
            return FakeResponse([row])

        if self.operation == "select":
            rows = list(self.rows.values())
            if "id" in self.filters:
                rows = [row for row in rows if row["id"] == self.filters["id"]]
            if self.limit_value is not None:
                rows = rows[: self.limit_value]
            return FakeResponse(rows)

        if self.operation == "update":
            user_id = self.filters.get("id")
            if user_id not in self.rows:
                return FakeResponse([])
            self.rows[user_id] = {**self.rows[user_id], **self.payload}
            return FakeResponse([self.rows[user_id]])

        if self.operation == "delete":
            user_id = self.filters.get("id")
            if user_id not in self.rows:
                return FakeResponse([])
            return FakeResponse([self.rows.pop(user_id)])

        raise AssertionError(f"Unsupported fake operation: {self.operation}")


class FakeSupabaseClient:
    """In-memory Supabase client dedicated to the `users` table."""

    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}

    def table(self, table_name: str) -> FakeUsersTable:
        assert table_name == USERS_TABLE
        return FakeUsersTable(self.rows)


@pytest.fixture
def user_service() -> UserService:
    """Return a UserService wired to an in-memory Supabase fake."""
    return UserService(client=FakeSupabaseClient())


def test_create_user(user_service: UserService) -> None:
    """create_user inserts a row and returns a validated User model."""
    user = user_service.create_user(
        UserCreate(id="user-1", email="maken@example.com", name="MAKEN")
    )

    assert user.id == "user-1"
    assert user.email == "maken@example.com"
    assert user.name == "MAKEN"
    assert user.role is UserRole.USER


def test_get_user(user_service: UserService) -> None:
    """get_user reads an existing user by id."""
    user_service.create_user({"id": "user-1", "email": "maken@example.com"})

    user = user_service.get_user("user-1")

    assert user.id == "user-1"
    assert user.email == "maken@example.com"


def test_update_user(user_service: UserService) -> None:
    """update_user persists mutable fields and validates the returned row."""
    user_service.create_user({"id": "user-1", "email": "maken@example.com"})

    user = user_service.update_user("user-1", UserUpdate(name="Admin", role=UserRole.ADMIN))

    assert user.name == "Admin"
    assert user.role is UserRole.ADMIN


def test_delete_user(user_service: UserService) -> None:
    """delete_user removes a row and future reads fail clearly."""
    user_service.create_user({"id": "user-1", "email": "maken@example.com"})

    user_service.delete_user("user-1")

    with pytest.raises(UserNotFoundError):
        user_service.get_user("user-1")


def test_list_users(user_service: UserService) -> None:
    """list_users returns validated users without hitting a live Supabase project."""
    user_service.create_user({"id": "user-1", "email": "one@example.com"})
    user_service.create_user({"id": "user-2", "email": "two@example.com", "role": "admin"})

    users = user_service.list_users()

    assert [user.id for user in users] == ["user-1", "user-2"]
    assert users[1].role is UserRole.ADMIN

