"""Test defaults: no network, no surprises.

`.env` carries real credentials, and several of them switch behaviour on merely
by being present. Without this the suite quietly became a live integration run
— billed, slow, and dependent on a network round trip for results that are
supposed to be deterministic. Worse, it wrote its fixture leads ("Ravi /
plumber / Pune") straight into the shared production Neon instance.

Everything is blanked here. Each of these degrades exactly as it does on a host
where the credential was never set, so the tests exercise a real code path
rather than a mocked one.
"""

from __future__ import annotations

import pytest

from agent.config import settings as agent_settings
from backend.config import settings as backend_settings

#: (settings object, field) pairs blanked for every test. Both `settings` are
#: frozen dataclasses, hence `object.__setattr__`.
_BLANKED = [
    (backend_settings, "database_url"),    # brain + lead store report unavailable
    (backend_settings, "sarvam_api_key"),  # translate falls back to the offline table
    (backend_settings, "whatsapp_api_key"),
    (agent_settings, "backend_url"),       # the agent's event queue stays local
    (agent_settings, "backend_token"),
]


@pytest.fixture(autouse=True)
def _offline_by_default():
    original = [(obj, name, getattr(obj, name)) for obj, name in _BLANKED]
    for obj, name in _BLANKED:
        object.__setattr__(obj, name, "")
    yield
    for obj, name, value in original:
        object.__setattr__(obj, name, value)
