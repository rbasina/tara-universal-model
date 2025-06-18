"""
TARA Universal Model integration module.
Provides backward compatibility adapter for tara-ai-companion integration.
"""

from .adapter import (
    TARAAdapter,
    AdapterConfig,
    create_tara_adapter,
    TARAAdapterContext,
    AsyncTARAAdapterContext
)

__all__ = [
    "TARAAdapter",
    "AdapterConfig", 
    "create_tara_adapter",
    "TARAAdapterContext",
    "AsyncTARAAdapterContext"
] 