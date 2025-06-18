"""
TARA Universal Model - Privacy-first conversational AI with emotional intelligence.

A cost-effective alternative to enterprise AI solutions, providing professional
domain expertise in healthcare, business, education, creative, and leadership.

Features:
- 100% local processing for privacy
- Emotional intelligence with professional context
- Domain-specific adapters via LoRA/QLoRA
- HIPAA compliant for healthcare applications
- Integration with tara-ai-companion

Cost: <$100 total vs $3,000+ enterprise solutions
"""

from .serving.model import TARAUniversalModel, ChatResponse, ChatMessage
from .integration.adapter import TARAAdapter, create_tara_adapter
from .utils.config import get_config, TARAConfig

__version__ = "1.0.0"
__author__ = "TARA Development Team"
__email__ = "contact@tara-ai.com"
__description__ = "Privacy-first conversational AI with emotional intelligence and professional domain expertise"

# Package metadata
__all__ = [
    "TARAUniversalModel",
    "TARAAdapter", 
    "create_tara_adapter",
    "ChatResponse",
    "ChatMessage",
    "get_config",
    "TARAConfig",
    "__version__"
]

# Supported professional domains
SUPPORTED_DOMAINS = [
    "healthcare",
    "business", 
    "education",
    "creative",
    "leadership",
    "universal"
]

# Quick start example
QUICK_START = """
Quick Start:

# 1. Install dependencies
pip install -r requirements.txt

# 2. Download models
python scripts/download_models.py --setup

# 3. Generate training data
python scripts/train_domain.py --domain healthcare --generate-data --samples 5000

# 4. Train domain adapter  
python scripts/train_domain.py --domain healthcare --generate-data --epochs 3

# 5. Serve the model
python scripts/serve_model.py --preload-domains healthcare

# 6. Use in code
from tara_universal_model import create_tara_adapter

adapter = create_tara_adapter(domain="healthcare")
response = adapter.process_message("I'm feeling anxious about my upcoming surgery")
print(response["response"])
"""

def get_version():
    """Get the current version of TARA Universal Model."""
    return __version__

def get_supported_domains():
    """Get list of supported professional domains."""
    return SUPPORTED_DOMAINS.copy()

def print_quick_start():
    """Print quick start guide."""
    print(QUICK_START)
