#!/usr/bin/env python3
"""
🎯 TARA Universal GGUF CLI - Phase-Wise Model Creation
Comprehensive command-line interface for creating phase-wise GGUF models with intelligent routing and emotional intelligence
"""

import argparse
import logging
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import our utilities
from universal_gguf_factory import UniversalGGUFFactory, QuantizationType, CompressionType
from cleanup_utilities import ModelCleanupUtilities
from phase_manager import PhaseManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class TARAGGUFCLI:
    """Command-line interface for TARA Universal GGUF creation"""
    
    def __init__(self):
        self.factory = UniversalGGUFFactory()
        self.cleanup_utils = ModelCleanupUtilities()
        self.phase_manager = PhaseManager()
        
    def create_phase(self, args):
        """Create a new phase with domains"""
        logger.info(f"🎯 Creating Phase {args.phase_number}...")
        
        # Parse domains
        domains = args.domains.split(',') if args.domains else []
        
        # Create compression config
        compression_config = {
            'quantization': args.quantization,
            'compression_type': args.compression_type,
            'target_size_mb': args.target_size,
            'quality_threshold': args.quality_threshold,
            'speed_priority': args.speed_priority
        }
        
        # Create phase
        success = self.phase_manager.create_phase(
            args.phase_number, 
            domains, 
            compression_config
        )
        
        if success:
            logger.info(f"✅ Phase {args.phase_number} created successfully")
            
            # Show phase summary
            summary = self.phase_manager.get_phase_summary(args.phase_number)
            print(json.dumps(summary, indent=2))
        else:
            logger.error(f"❌ Failed to create Phase {args.phase_number}")
            sys.exit(1)
    
    def add_domain(self, args):
        """Add domain to existing phase"""
        logger.info(f"➕ Adding domain {args.domain} to Phase {args.phase_number}...")
        
        success = self.phase_manager.add_domain_to_phase(
            args.phase_number,
            args.domain,
            Path(args.adapter_path) if args.adapter_path else None
        )
        
        if success:
            logger.info(f"✅ Domain {args.domain} added to Phase {args.phase_number}")
        else:
            logger.error(f"❌ Failed to add domain {args.domain}")
            sys.exit(1)
    
    def clean_model(self, args):
        """Clean model directory"""
        logger.info(f"🧹 Cleaning model: {args.model_path}...")
        
        model_path = Path(args.model_path)
        output_path = Path(args.output_path) if args.output_path else None
        
        # Clean model
        result = self.cleanup_utils.clean_model_directory(model_path, output_path)
        
        if result.success:
            logger.info(f"✅ Model cleaned successfully")
            logger.info(f"📊 Original size: {result.original_size_mb:.1f}MB")
            logger.info(f"📊 Cleaned size: {result.cleaned_size_mb:.1f}MB")
            logger.info(f"🗑️ Removed {len(result.removed_files)} files")
            
            # Save cleanup report
            if args.report_path:
                self.cleanup_utils.save_cleanup_report(result, Path(args.report_path))
            
            # Show validation result
            validation = result.validation_result
            print(f"Validation Score: {validation.validation_score:.2f}")
            if validation.issues:
                print(f"Issues: {validation.issues}")
            if validation.warnings:
                print(f"Warnings: {validation.warnings}")
        else:
            logger.error(f"❌ Model cleaning failed: {result.error_message}")
            sys.exit(1)
    
    def build_phase(self, args):
        """Build GGUF for specific phase"""
        logger.info(f"🔨 Building Phase {args.phase_number} GGUF...")
        
        # Get phase info
        phase_info = self.phase_manager.get_phase_info(args.phase_number)
        if not phase_info:
            logger.error(f"❌ Phase {args.phase_number} not found")
            sys.exit(1)
        
        # Get ready domains
        ready_domains = self.phase_manager.get_ready_domains(args.phase_number)
        if not ready_domains:
            logger.error(f"❌ No ready domains for Phase {args.phase_number}")
            sys.exit(1)
        
        logger.info(f"📊 Building with {len(ready_domains)} domains: {ready_domains}")
        
        # Add domains to factory
        for domain in ready_domains:
            domain_status = self.phase_manager.get_domain_status(domain)
            if domain_status:
                success = self.factory.add_domain_phase(
                    domain,
                    domain_status.adapter_path,
                    training_quality=domain_status.training_quality,
                    response_speed=0.8,  # Default
                    emotional_intensity=0.7,  # Default
                    specialties=[domain]
                )
                if not success:
                    logger.error(f"❌ Failed to add domain {domain}")
                    sys.exit(1)
        
        # Update phase status
        self.phase_manager.update_phase_status(args.phase_number, 'merging')
        
        # Create GGUF
        quantization = QuantizationType(args.quantization)
        compression = CompressionType(args.compression_type)
        
        success = self.factory.create_phase_gguf(args.phase_number, quantization, compression)
        
        if success:
            # Update phase status
            phase_summary = self.factory.get_phase_summary(args.phase_number)
            self.phase_manager.update_phase_status(
                args.phase_number, 
                'deployed',
                Path(phase_summary['model_path']) if phase_summary['model_path'] else None,
                phase_summary
            )
            
            logger.info(f"🎉 Phase {args.phase_number} GGUF built successfully!")
            print(json.dumps(phase_summary, indent=2))
        else:
            self.phase_manager.update_phase_status(args.phase_number, 'failed')
            logger.error(f"❌ Failed to build Phase {args.phase_number} GGUF")
            sys.exit(1)
    
    def deploy_phase(self, args):
        """Deploy phase to target"""
        logger.info(f"🚀 Deploying Phase {args.phase_number} to {args.target}...")
        
        success = self.phase_manager.deploy_phase(args.phase_number, args.target)
        
        if success:
            logger.info(f"✅ Phase {args.phase_number} deployed successfully")
        else:
            logger.error(f"❌ Failed to deploy Phase {args.phase_number}")
            sys.exit(1)
    
    def list_phases(self, args):
        """List all phases"""
        summary = self.phase_manager.get_overall_summary()
        print(json.dumps(summary, indent=2))
    
    def show_phase(self, args):
        """Show specific phase details"""
        summary = self.phase_manager.get_phase_summary(args.phase_number)
        if summary:
            print(json.dumps(summary, indent=2))
        else:
            logger.error(f"❌ Phase {args.phase_number} not found")
            sys.exit(1)
    
    def update_domain(self, args):
        """Update domain status"""
        logger.info(f"📝 Updating domain {args.domain}...")
        
        # Parse performance metrics
        performance_metrics = {}
        if args.metrics:
            try:
                performance_metrics = json.loads(args.metrics)
            except json.JSONDecodeError:
                logger.error("❌ Invalid metrics JSON")
                sys.exit(1)
        
        success = self.phase_manager.update_domain_status(
            args.domain,
            args.status,
            args.quality,
            performance_metrics
        )
        
        if success:
            logger.info(f"✅ Domain {args.domain} updated successfully")
        else:
            logger.error(f"❌ Failed to update domain {args.domain}")
            sys.exit(1)
    
    def advance_phase(self, args):
        """Advance to next phase"""
        new_phase = self.phase_manager.advance_phase()
        logger.info(f"🚀 Advanced to Phase {new_phase}")
    
    def cleanup_phase(self, args):
        """Clean up phase resources"""
        logger.info(f"🧹 Cleaning up Phase {args.phase_number}...")
        
        success = self.phase_manager.cleanup_phase(args.phase_number)
        
        if success:
            logger.info(f"✅ Phase {args.phase_number} cleaned up successfully")
        else:
            logger.error(f"❌ Failed to cleanup Phase {args.phase_number}")
            sys.exit(1)

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="TARA Universal GGUF CLI - Phase-Wise Model Creation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create Phase 1 with healthcare and business domains
  python tara_gguf_cli.py create-phase 1 --domains healthcare,business --quantization Q4_K_M
  
  # Add education domain to Phase 1
  python tara_gguf_cli.py add-domain 1 education --adapter-path models/adapters/education
  
  # Clean a model directory
  python tara_gguf_cli.py clean-model models/adapters/healthcare --output-path models/cleaned/healthcare
  
  # Build Phase 1 GGUF
  python tara_gguf_cli.py build-phase 1 --quantization Q4_K_M --compression-type standard
  
  # Deploy Phase 1 to MeeTARA
  python tara_gguf_cli.py deploy-phase 1 --target /path/to/meetara/models
  
  # List all phases
  python tara_gguf_cli.py list-phases
  
  # Show Phase 1 details
  python tara_gguf_cli.py show-phase 1
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create phase command
    create_parser = subparsers.add_parser('create-phase', help='Create a new phase')
    create_parser.add_argument('phase_number', type=int, help='Phase number')
    create_parser.add_argument('--domains', help='Comma-separated list of domains')
    create_parser.add_argument('--quantization', default='Q4_K_M', 
                              choices=['Q2_K', 'Q4_K_M', 'Q5_K_M', 'Q8_0'],
                              help='Quantization type')
    create_parser.add_argument('--compression-type', default='standard',
                              choices=['standard', 'sparse', 'hybrid', 'distilled'],
                              help='Compression type')
    create_parser.add_argument('--target-size', type=float, help='Target size in MB')
    create_parser.add_argument('--quality-threshold', type=float, default=0.95,
                              help='Quality threshold (0-1)')
    create_parser.add_argument('--speed-priority', action='store_true',
                              help='Prioritize speed over quality')
    
    # Add domain command
    add_parser = subparsers.add_parser('add-domain', help='Add domain to phase')
    add_parser.add_argument('phase_number', type=int, help='Phase number')
    add_parser.add_argument('domain', help='Domain name')
    add_parser.add_argument('--adapter-path', help='Path to domain adapter')
    
    # Clean model command
    clean_parser = subparsers.add_parser('clean-model', help='Clean model directory')
    clean_parser.add_argument('model_path', help='Path to model directory')
    clean_parser.add_argument('--output-path', help='Output path for cleaned model')
    clean_parser.add_argument('--report-path', help='Path to save cleanup report')
    
    # Build phase command
    build_parser = subparsers.add_parser('build-phase', help='Build GGUF for phase')
    build_parser.add_argument('phase_number', type=int, help='Phase number')
    build_parser.add_argument('--quantization', default='Q4_K_M',
                              choices=['Q2_K', 'Q4_K_M', 'Q5_K_M', 'Q8_0'],
                              help='Quantization type')
    build_parser.add_argument('--compression-type', default='standard',
                              choices=['standard', 'sparse', 'hybrid', 'distilled'],
                              help='Compression type')
    
    # Deploy phase command
    deploy_parser = subparsers.add_parser('deploy-phase', help='Deploy phase to target')
    deploy_parser.add_argument('phase_number', type=int, help='Phase number')
    deploy_parser.add_argument('target', help='Deployment target path')
    
    # List phases command
    subparsers.add_parser('list-phases', help='List all phases')
    
    # Show phase command
    show_parser = subparsers.add_parser('show-phase', help='Show phase details')
    show_parser.add_argument('phase_number', type=int, help='Phase number')
    
    # Update domain command
    update_parser = subparsers.add_parser('update-domain', help='Update domain status')
    update_parser.add_argument('domain', help='Domain name')
    update_parser.add_argument('--status', choices=['pending', 'training', 'complete', 'failed'],
                              help='Training status')
    update_parser.add_argument('--quality', type=float, help='Training quality (0-1)')
    update_parser.add_argument('--metrics', help='Performance metrics as JSON')
    
    # Advance phase command
    subparsers.add_parser('advance-phase', help='Advance to next phase')
    
    # Cleanup phase command
    cleanup_parser = subparsers.add_parser('cleanup-phase', help='Clean up phase resources')
    cleanup_parser.add_argument('phase_number', type=int, help='Phase number')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Create CLI instance
    cli = TARAGGUFCLI()
    
    # Execute command
    try:
        if args.command == 'create-phase':
            cli.create_phase(args)
        elif args.command == 'add-domain':
            cli.add_domain(args)
        elif args.command == 'clean-model':
            cli.clean_model(args)
        elif args.command == 'build-phase':
            cli.build_phase(args)
        elif args.command == 'deploy-phase':
            cli.deploy_phase(args)
        elif args.command == 'list-phases':
            cli.list_phases(args)
        elif args.command == 'show-phase':
            cli.show_phase(args)
        elif args.command == 'update-domain':
            cli.update_domain(args)
        elif args.command == 'advance-phase':
            cli.advance_phase(args)
        elif args.command == 'cleanup-phase':
            cli.cleanup_phase(args)
        else:
            logger.error(f"❌ Unknown command: {args.command}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("⏹️ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 