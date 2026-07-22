"""Doc-completeness guard: the docs must document the whole config surface.

These tests introspect the Pydantic models that make up the ``.github/heimdall.yml``
config (:mod:`heimdall.repo_config`) and the service env settings
(:mod:`heimdall.config`) and assert every field name appears somewhere in the
documented surface — the README plus every page under ``docs/``.  They exist so a
future field added to the config can't silently drift out of the documented surface —
the acceptance criterion for issue #12 is that the config reference matches the
implemented knobs exactly.  The scan spans README + ``docs/`` so the reference is free
to live in ``docs/configuration.md`` without the guard caring where exactly.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from heimdall.config import Settings
from heimdall.repo_config import (
    CommentIncorporation,
    CustomLensConfig,
    GuardrailCaps,
    LensConfig,
    RepoConfig,
    ResourceLimits,
    ScopeFilters,
    SynthesisConfig,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DOC_SOURCES = (_REPO_ROOT / "README.md", *sorted((_REPO_ROOT / "docs").glob("*.md")))


def _docs_text() -> str:
    """Concatenated text of the README plus every page under ``docs/``."""
    return "\n".join(p.read_text(encoding="utf-8") for p in _DOC_SOURCES)


# Every model whose field names must appear verbatim in the documented config reference.
_CONFIG_MODELS = (
    RepoConfig,
    LensConfig,
    CustomLensConfig,
    SynthesisConfig,
    ResourceLimits,
    ScopeFilters,
    GuardrailCaps,
    CommentIncorporation,
)


@pytest.mark.parametrize("model", _CONFIG_MODELS, ids=lambda m: m.__name__)
def test_readme_documents_every_repo_config_field(model: type[BaseModel]) -> None:
    """The docs mention every field of each heimdall.yml config model by name."""
    text = _docs_text()
    missing = [name for name in model.model_fields if name not in text]
    assert not missing, f"docs are missing {model.__name__} fields: {missing}"


def test_readme_documents_every_service_env_field() -> None:
    """The docs mention every service env Setting by its env-var (UPPER) name."""
    text = _docs_text()
    missing = [
        name.upper() for name in Settings.model_fields if name.upper() not in text
    ]
    assert not missing, f"docs are missing service env vars: {missing}"
