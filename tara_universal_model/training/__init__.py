"""
TARA Universal Model training module.
Provides LoRA/QLoRA training for domain-specific adaptation.
"""

from .trainer import TARATrainer, ConversationDataset
from .cli import main as cli_main

__all__ = [
    "TARATrainer",
    "ConversationDataset", 
    "cli_main"
]
