"""Small, centralized logging setup for CLI and API entrypoints."""

import logging

from customer_service_agent.config import Settings


def configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )
