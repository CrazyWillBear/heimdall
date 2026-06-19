"""Doc-completeness guard: the README must document the whole config surface.

These tests introspect the Pydantic models that make up the ``.github/heimdall.yml``
config (:mod:`heimdall.repo_config`) and the service env settings
(:mod:`heimdall.config`) and assert every field name appears in the README's config
reference.  They exist so a future field added to the config can't silently drift out
of the documented surface — the acceptance criterion for issue #12 is that the config
reference matches the implemented knobs exactly.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from heimdall.config import Settings
from heimdall.repo_config import (
    CustomLensConfig,
    GuardrailCaps,
    LensConfig,
    RepoConfig,
    ScopeFilters,
)

_README = Path(__file__).resolve().parent.parent / "README.md"


def _readme_text() -> str:
    return _README.read_text(encoding="utf-8")


# Every model whose field names must appear verbatim in the README config reference.
_CONFIG_MODELS = (
    RepoConfig,
    LensConfig,
    CustomLensConfig,
    ScopeFilters,
    GuardrailCaps,
)


@pytest.mark.parametrize("model", _CONFIG_MODELS, ids=lambda m: m.__name__)
def test_readme_documents_every_repo_config_field(model: type[BaseModel]) -> None:
    """The README mentions every field of each heimdall.yml config model by name."""
    text = _readme_text()
    missing = [name for name in model.model_fields if name not in text]
    assert not missing, f"README is missing {model.__name__} fields: {missing}"


def test_readme_documents_every_service_env_field() -> None:
    """The README mentions every service env Setting by its env-var (UPPER) name."""
    text = _readme_text()
    missing = [
        name.upper() for name in Settings.model_fields if name.upper() not in text
    ]
    assert not missing, f"README is missing service env vars: {missing}"
