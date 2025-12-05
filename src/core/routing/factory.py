import logging

from .base import Router
from .semantic_router import SemanticRouter

logger = logging.getLogger(__name__)


def get_router(qa_generator, log_step=None, router_embedding_manager=None) -> Router:
    try:
        return SemanticRouter(
            qa_generator=qa_generator,
            log_step=log_step,
            router_embedding_manager=router_embedding_manager
        )
    except Exception as e:
        logger.error(f"Failed to initialize semantic router: {e}")
        raise
