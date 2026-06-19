"""FastAPI application factory.

Wires together config, database, and the webhook router into a single ASGI app.
The factory pattern lets tests inject a custom Settings instance without touching
environment variables.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI

from heimdall.config import Settings
from heimdall.webhook import make_webhook_router

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, arq_pool: object = None) -> FastAPI:
    """Construct and return the Heimdall FastAPI application.

    Args:
        settings: Optional Settings override (defaults to reading from env/dotenv).
        arq_pool: Optional Arq Redis pool (tests can pass None; real app wires in
                  the live pool via lifespan or a startup hook).

    Returns:
        A configured FastAPI instance.
    """
    if settings is None:
        settings = Settings()  # type: ignore[call-arg]

    app = FastAPI(title="Heimdall", version="0.1.0")

    webhook_router = make_webhook_router(
        webhook_secret=settings.webhook_secret,
        arq_pool=arq_pool,
    )
    app.include_router(webhook_router)

    return app
