"""
TARA Universal Model domain experts module.
Provides professional domain routing and context switching.
"""

from .router import DomainRouter, DomainConfig

__all__ = [
    "DomainRouter", 
    "DomainConfig"
]
