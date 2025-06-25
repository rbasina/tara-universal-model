#!/usr/bin/env python3
"""
Model serving script for TARA Universal Model.
Provides FastAPI server with health checks, monitoring, and rate limiting.
"""

import os
import sys
import argparse
import logging
import asyncio
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tara_universal_model.serving.api import run_server
from tara_universal_model.utils.config import get_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for model serving."""
    parser = argparse.ArgumentParser(
        description="Serve TARA Universal Model via FastAPI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Serve with default settings
  python scripts/serve_model.py
  
  # Serve on specific port
  python scripts/serve_model.py --port 8080
  
  # Serve with multiple workers
  python scripts/serve_model.py --workers 4
  
  # Serve with custom config
  python scripts/serve_model.py --config configs/production.yaml
        """
    )
    
    parser.add_argument('--host', default='0.0.0.0',
                       help='Host to bind server to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port to bind server to (default: 8000)')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of worker processes (default: 1)')
    parser.add_argument('--config', default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--reload', action='store_true',
                       help='Enable auto-reload for development')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--preload-domains', nargs='+',
                       default=['healthcare', 'business', 'education'],
                       help='Domains to preload on startup')
    
    args = parser.parse_args()
    
    # Set up logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        # Load configuration
        config = get_config(args.config)
        
        # Update config with CLI arguments
        config.serving_config.host = args.host
        config.serving_config.port = args.port
        config.serving_config.workers = args.workers
        config.serving_config.preload_domains = args.preload_domains
        config.serving_config.log_level = args.log_level
        
        logger.info("🚀 Starting TARA Universal Model Server")
        logger.info(f"Host: {args.host}")
        logger.info(f"Port: {args.port}")
        logger.info(f"Workers: {args.workers}")
        logger.info(f"Preload Domains: {args.preload_domains}")
        logger.info(f"Config: {args.config}")
        
        # Check if models directory exists
        models_dir = Path("models")
        if not models_dir.exists():
            logger.warning("Models directory not found. Run 'python scripts/download_models.py --setup' first.")
        
        # Start server
        run_server(
            host=args.host,
            port=args.port,
            workers=args.workers
        )
        
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 