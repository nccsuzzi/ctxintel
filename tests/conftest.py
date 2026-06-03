"""Shared test fixtures for ctxintel tests."""

import pytest

from ctxintel.models import Message
from ctxintel.pipeline import ContextIntel


@pytest.fixture
def sample_messages():
    """Return 20 realistic Message objects covering diverse scenarios."""
    return [
        Message(role="system", content="You are a helpful coding assistant."),
        Message(role="user", content="Hi, my name is John. I'm a backend developer."),
        Message(role="assistant", content="Hello John! How can I help you today?"),
        Message(role="user", content="I prefer Python for all my backend projects."),
        Message(role="assistant", content="Python is an excellent choice for backend development."),
        Message(role="user", content="I'm using FastAPI with clean architecture."),
        Message(role="assistant", content="FastAPI with clean architecture is a great pattern."),
        Message(role="user", content="We decided to deploy on AWS ECS."),
        Message(role="assistant", content="AWS ECS is solid for container orchestration."),
        Message(role="user", content="Don't use synchronous code anywhere. Everything must be async."),
        Message(role="user", content="ok"),
        Message(role="user", content="cool"),
        Message(role="user", content="hmm"),
        Message(role="user", content="I need to build a REST API for user management."),
        Message(role="assistant", content="I can help you build that REST API."),
        Message(role="user", content="Remember, I prefer Python for backend."),
        Message(role="assistant", content="Yes, we'll use Python with FastAPI."),
        Message(role="user", content="Also we need PostgreSQL as the database."),
        Message(role="assistant", content="PostgreSQL works great with async SQLAlchemy."),
        Message(role="user", content="Now add JWT authentication to the API."),
    ]


@pytest.fixture
def sdk(tmp_path):
    """Return a ContextIntel instance with a temporary memory path."""
    return ContextIntel(memory_path=str(tmp_path / "memory.json"))


@pytest.fixture
def raw_messages():
    """Return raw message dicts (as would come from an API)."""
    return [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Hi, my name is John."},
        {"role": "assistant", "content": "Hello John!"},
        {"role": "user", "content": "I prefer Python."},
        {"role": "user", "content": "Build a REST API."},
        {"role": "user", "content": "Don't use synchronous code."},
        {"role": "user", "content": "We decided to deploy on AWS."},
        {"role": "user", "content": "ok"},
        {"role": "user", "content": "cool"},
        {"role": "user", "content": "Now add JWT authentication."},
    ]
