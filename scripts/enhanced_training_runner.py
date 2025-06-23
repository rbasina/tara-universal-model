#!/usr/bin/env python3
"""
Enhanced Training Runner - Production-Ready TARA Universal Model Training
Ensures backend compatibility, provides real-time monitoring, and validates during training.
"""

import os
import sys
import asyncio
import logging
import json
import subprocess
import time
import threading
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tara_universal_model.training.enhanced_trainer import TrainingOrchestrator
from tara_universal_model.utils.config import get_config

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/enhanced_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TrainingRunner:
    """
    Comprehensive training runner with backend monitoring and real-time progress tracking.
    """
    
    def __init__(self):
        self.config = get_config()
        self.orchestrator = TrainingOrchestrator()
        self.backend_process = None
        self.monitoring_active = False
        self.progress_file = "training_progress.json"
        
    async def run_complete_training(self):
        """Run complete training pipeline with backend monitoring."""
        logger.info("🎬 Starting Complete TARA Universal Model Training")
        logger.info("✨ Enhanced validation + Backend integration + Real-time monitoring")
        
        try:
            # Step 1: Start backend server
            await self._start_backend_server()
            
            # Step 2: Start progress monitoring
            self._start_progress_monitoring()
            
            # Step 3: Generate training data if needed
            await self._ensure_training_data()
            
            # Step 4: Run enhanced training
            training_results = await self.orchestrator.train_all_domains_enhanced()
            
            # Step 5: Final validation and reporting
            await self._final_validation_report(training_results)
            
            logger.info("🎉 Complete training pipeline finished successfully!")
            
        except Exception as e:
            logger.error(f"❌ Training pipeline failed: {e}")
            raise
        finally:
            # Cleanup
            self._cleanup()
    
    async def _start_backend_server(self):
        """Start the backend voice server for validation."""
        logger.info("🚀 Starting backend voice server...")
        
        try:
            # Check if already running
            import requests
            response = requests.get("http://localhost:5000/health", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Backend server already running")
                return
        except:
            pass
        
        # Start voice server
        try:
            self.backend_process = subprocess.Popen([
                sys.executable, "voice_server.py"
            ], cwd=str(project_root))
            
            # Wait for server to be ready
            for i in range(30):
                try:
                    import requests
                    response = requests.get("http://localhost:5000/health", timeout=2)
                    if response.status_code == 200:
                        logger.info("✅ Backend server started successfully")
                        return
                except:
                    pass
                await asyncio.sleep(1)
            
            logger.warning("⚠️ Backend server may not be fully ready")
            
        except Exception as e:
            logger.error(f"❌ Failed to start backend server: {e}")
            logger.info("📝 Continuing without backend validation")
    
    def _start_progress_monitoring(self):
        """Start real-time progress monitoring in background thread."""
        logger.info("📊 Starting real-time progress monitoring")
        
        self.monitoring_active = True
        monitoring_thread = threading.Thread(target=self._monitor_progress)
        monitoring_thread.daemon = True
        monitoring_thread.start()
    
    def _monitor_progress(self):
        """Monitor training progress and update progress file."""
        while self.monitoring_active:
            try:
                progress_data = {
                    "timestamp": datetime.now().isoformat(),
                    "status": "training",
                    "domains": {},
                    "system_info": self._get_system_info()
                }
                
                # Check training progress for each domain
                for domain in ["healthcare", "business", "education", "creative", "leadership"]:
                    domain_info = self._check_domain_progress(domain)
                    progress_data["domains"][domain] = domain_info
                
                # Save progress
                with open(self.progress_file, 'w') as f:
                    json.dump(progress_data, f, indent=2)
                
                # Log summary every 30 seconds
                self._log_progress_summary(progress_data)
                
            except Exception as e:
                logger.error(f"Progress monitoring error: {e}")
            
            time.sleep(30)  # Update every 30 seconds
    
    def _check_domain_progress(self, domain: str) -> dict:
        """Check progress for a specific domain."""
        domain_info = {
            "status": "pending",
            "model_files": [],
            "validation_results": None,
            "training_active": False
        }
        
        # Check for model files
        model_dir = Path(f"models/{domain}")
        if model_dir.exists():
            model_files = list(model_dir.rglob("*.bin")) + list(model_dir.rglob("*.safetensors"))
            domain_info["model_files"] = [str(f) for f in model_files]
            if model_files:
                domain_info["status"] = "training" if self._is_training_active(domain) else "completed"
        
        # Check for validation results
        validation_dir = Path("validation_results")
        if validation_dir.exists():
            validation_files = list(validation_dir.glob(f"{domain}_validation_*.json"))
            if validation_files:
                latest_validation = max(validation_files, key=lambda x: x.stat().st_mtime)
                try:
                    with open(latest_validation) as f:
                        domain_info["validation_results"] = json.load(f)
                except:
                    pass
        
        return domain_info
    
    def _is_training_active(self, domain: str) -> bool:
        """Check if training is currently active for domain."""
        # Check for recent log activity
        log_file = Path("logs/enhanced_training.log")
        if log_file.exists():
            # Check if domain mentioned in recent logs (last 2 minutes)
            try:
                import time as time_module
                if time_module.time() - log_file.stat().st_mtime < 120:  # 2 minutes
                    with open(log_file, 'r') as f:
                        recent_lines = f.readlines()[-20:]  # Last 20 lines
                        return any(domain in line for line in recent_lines)
            except:
                pass
        return False
    
    def _get_system_info(self) -> dict:
        """Get current system information."""
        import psutil
        import torch
        
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "gpu_available": torch.cuda.is_available(),
            "gpu_memory": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
        }
    
    def _log_progress_summary(self, progress_data: dict):
        """Log a summary of current progress."""
        domains = progress_data["domains"]
        
        completed = sum(1 for d in domains.values() if d["status"] == "completed")
        training = sum(1 for d in domains.values() if d["status"] == "training")
        pending = sum(1 for d in domains.values() if d["status"] == "pending")
        
        system = progress_data["system_info"]
        
        logger.info(f"📊 Progress: {completed} completed, {training} training, {pending} pending")
        logger.info(f"💻 System: CPU {system['cpu_percent']:.1f}%, RAM {system['memory_percent']:.1f}%")
    
    async def _ensure_training_data(self):
        """Ensure training data exists for all domains."""
        logger.info("📚 Checking training data availability...")
        
        from tara_universal_model.utils.data_generator import DataGenerator
        data_generator = DataGenerator(self.config.data_config)
        
        domains = ["healthcare", "business", "education", "creative", "leadership"]
        
        for domain in domains:
            data_file = Path(f"data/synthetic/{domain}_training_data.json")
            
            if not data_file.exists():
                logger.info(f"📝 Generating training data for {domain}...")
                try:
                    data_generator.generate_domain_data(domain, num_samples=2000)
                    logger.info(f"✅ {domain} training data generated")
                except Exception as e:
                    logger.error(f"❌ Failed to generate {domain} data: {e}")
            else:
                logger.info(f"✅ {domain} training data exists")
    
    async def _final_validation_report(self, training_results: dict):
        """Generate final validation report."""
        logger.info("📋 Generating final validation report...")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "training_summary": training_results,
            "production_ready_models": [],
            "failed_models": [],
            "recommendations": []
        }
        
        # Analyze results
        for domain, result in training_results.get("domain_progress", {}).items():
            if result["status"] == "completed":
                # Check if production ready
                model_path = result.get("model_path", "")
                if model_path and Path(model_path).exists():
                    report["production_ready_models"].append({
                        "domain": domain,
                        "model_path": model_path,
                        "training_duration": result.get("duration", 0)
                    })
            else:
                report["failed_models"].append({
                    "domain": domain,
                    "error": result.get("error", "Unknown error")
                })
        
        # Generate recommendations
        if report["failed_models"]:
            report["recommendations"].append("Review failed model training logs and retry with adjusted parameters")
        
        if len(report["production_ready_models"]) < 5:
            report["recommendations"].append("Complete training for all 5 domains before production deployment")
        
        if len(report["production_ready_models"]) == 5:
            report["recommendations"].append("All models ready - proceed with Phase 2 Perplexity Intelligence integration")
        
        # Save report
        report_path = "training_results/final_validation_report.json"
        os.makedirs("training_results", exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Log summary
        logger.info("📊 Final Validation Report:")
        logger.info(f"✅ Production Ready: {len(report['production_ready_models'])}/5 domains")
        logger.info(f"❌ Failed: {len(report['failed_models'])}/5 domains")
        logger.info(f"📄 Full report: {report_path}")
        
        return report
    
    def _cleanup(self):
        """Cleanup resources."""
        self.monitoring_active = False
        
        if self.backend_process:
            try:
                self.backend_process.terminate()
                self.backend_process.wait(timeout=10)
                logger.info("✅ Backend server stopped")
            except:
                try:
                    self.backend_process.kill()
                except:
                    pass

async def main():
    """Main entry point."""
    logger.info("🎬 TARA Universal Model - Enhanced Training Pipeline")
    logger.info("🛡️ Production-ready training with validation and monitoring")
    
    runner = TrainingRunner()
    
    try:
        await runner.run_complete_training()
        logger.info("🎉 Training pipeline completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("⏹️ Training interrupted by user")
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {e}")
        raise
    finally:
        runner._cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 