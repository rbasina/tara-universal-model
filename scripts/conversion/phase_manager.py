#!/usr/bin/env python3
"""
🎯 Phase Manager for TARA Universal Model
Handles phase-wise domain expansion, model lifecycle, and deployment management
"""

import os
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import time

logger = logging.getLogger(__name__)

@dataclass
class PhaseInfo:
    phase_number: int
    domains: List[str]
    model_path: Optional[Path]
    created_date: str
    status: str  # 'planning', 'training', 'merging', 'compressing', 'deployed', 'failed'
    compression_config: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    deployment_target: Optional[str] = None

@dataclass
class DomainStatus:
    name: str
    phase: int
    training_status: str  # 'pending', 'training', 'complete', 'failed'
    training_quality: float
    adapter_path: Path
    last_updated: str
    performance_metrics: Dict[str, Any]

class PhaseManager:
    """Manages phase-wise domain expansion and model lifecycle"""
    
    def __init__(self, base_dir: Path = None):
        self.base_dir = base_dir or Path(".")
        self.phases_dir = self.base_dir / "phases"
        self.phase_data_file = self.phases_dir / "phase_data.json"
        self.current_phase = 1
        
        # Initialize directories
        self.phases_dir.mkdir(exist_ok=True)
        
        # Load existing phase data
        self.phases: Dict[int, PhaseInfo] = {}
        self.domain_status: Dict[str, DomainStatus] = {}
        self._load_phase_data()
    
    def _load_phase_data(self):
        """Load existing phase data from file"""
        if self.phase_data_file.exists():
            try:
                with open(self.phase_data_file, 'r') as f:
                    data = json.load(f)
                
                # Load phases
                for phase_num, phase_data in data.get('phases', {}).items():
                    phase_info = PhaseInfo(**phase_data)
                    self.phases[int(phase_num)] = phase_info
                
                # Load domain status
                for domain_name, domain_data in data.get('domains', {}).items():
                    domain_status = DomainStatus(**domain_data)
                    self.domain_status[domain_name] = domain_status
                
                # Set current phase
                self.current_phase = data.get('current_phase', 1)
                
                logger.info(f"📊 Loaded {len(self.phases)} phases and {len(self.domain_status)} domains")
                
            except Exception as e:
                logger.error(f"❌ Failed to load phase data: {e}")
    
    def _save_phase_data(self):
        """Save phase data to file"""
        try:
            data = {
                'current_phase': self.current_phase,
                'phases': {str(phase_num): asdict(phase_info) for phase_num, phase_info in self.phases.items()},
                'domains': {domain_name: asdict(domain_status) for domain_name, domain_status in self.domain_status.items()}
            }
            
            with open(self.phase_data_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"❌ Failed to save phase data: {e}")
    
    def create_phase(self, phase_number: int, domains: List[str], 
                    compression_config: Dict[str, Any] = None) -> bool:
        """Create a new phase"""
        
        try:
            # Validate phase number
            if phase_number in self.phases:
                logger.warning(f"⚠️ Phase {phase_number} already exists")
                return False
            
            # Create phase info
            phase_info = PhaseInfo(
                phase_number=phase_number,
                domains=domains,
                model_path=None,
                created_date=datetime.now().isoformat(),
                status='planning',
                compression_config=compression_config or {},
                performance_metrics={}
            )
            
            # Add to phases
            self.phases[phase_number] = phase_info
            
            # Update domain status
            for domain in domains:
                if domain not in self.domain_status:
                    self.domain_status[domain] = DomainStatus(
                        name=domain,
                        phase=phase_number,
                        training_status='pending',
                        training_quality=0.0,
                        adapter_path=Path(f"models/adapters/{domain}"),
                        last_updated=datetime.now().isoformat(),
                        performance_metrics={}
                    )
                else:
                    self.domain_status[domain].phase = phase_number
                    self.domain_status[domain].last_updated = datetime.now().isoformat()
            
            # Save data
            self._save_phase_data()
            
            logger.info(f"✅ Created Phase {phase_number} with {len(domains)} domains")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create Phase {phase_number}: {e}")
            return False
    
    def add_domain_to_phase(self, phase_number: int, domain_name: str, 
                           adapter_path: Path = None) -> bool:
        """Add a domain to an existing phase"""
        
        try:
            if phase_number not in self.phases:
                logger.error(f"❌ Phase {phase_number} does not exist")
                return False
            
            phase_info = self.phases[phase_number]
            
            if domain_name in phase_info.domains:
                logger.warning(f"⚠️ Domain {domain_name} already in Phase {phase_number}")
                return True
            
            # Add domain to phase
            phase_info.domains.append(domain_name)
            
            # Update domain status
            if domain_name not in self.domain_status:
                self.domain_status[domain_name] = DomainStatus(
                    name=domain_name,
                    phase=phase_number,
                    training_status='pending',
                    training_quality=0.0,
                    adapter_path=adapter_path or Path(f"models/adapters/{domain_name}"),
                    last_updated=datetime.now().isoformat(),
                    performance_metrics={}
                )
            else:
                self.domain_status[domain_name].phase = phase_number
                self.domain_status[domain_name].last_updated = datetime.now().isoformat()
            
            # Save data
            self._save_phase_data()
            
            logger.info(f"✅ Added {domain_name} to Phase {phase_number}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add domain {domain_name} to Phase {phase_number}: {e}")
            return False
    
    def update_domain_status(self, domain_name: str, training_status: str = None,
                           training_quality: float = None, 
                           performance_metrics: Dict[str, Any] = None) -> bool:
        """Update domain training status"""
        
        try:
            if domain_name not in self.domain_status:
                logger.error(f"❌ Domain {domain_name} not found")
                return False
            
            domain_status = self.domain_status[domain_name]
            
            if training_status:
                domain_status.training_status = training_status
            if training_quality is not None:
                domain_status.training_quality = training_quality
            if performance_metrics:
                domain_status.performance_metrics.update(performance_metrics)
            
            domain_status.last_updated = datetime.now().isoformat()
            
            # Save data
            self._save_phase_data()
            
            logger.info(f"✅ Updated {domain_name} status: {training_status}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update domain {domain_name}: {e}")
            return False
    
    def update_phase_status(self, phase_number: int, status: str, 
                          model_path: Path = None, performance_metrics: Dict[str, Any] = None) -> bool:
        """Update phase status"""
        
        try:
            if phase_number not in self.phases:
                logger.error(f"❌ Phase {phase_number} not found")
                return False
            
            phase_info = self.phases[phase_number]
            phase_info.status = status
            
            if model_path:
                phase_info.model_path = model_path
            if performance_metrics:
                phase_info.performance_metrics.update(performance_metrics)
            
            # Save data
            self._save_phase_data()
            
            logger.info(f"✅ Updated Phase {phase_number} status: {status}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update Phase {phase_number}: {e}")
            return False
    
    def get_phase_info(self, phase_number: int) -> Optional[PhaseInfo]:
        """Get phase information"""
        return self.phases.get(phase_number)
    
    def get_domain_status(self, domain_name: str) -> Optional[DomainStatus]:
        """Get domain status"""
        return self.domain_status.get(domain_name)
    
    def get_ready_domains(self, phase_number: int) -> List[str]:
        """Get domains ready for phase processing"""
        if phase_number not in self.phases:
            return []
        
        phase_info = self.phases[phase_number]
        ready_domains = []
        
        for domain in phase_info.domains:
            if domain in self.domain_status:
                domain_status = self.domain_status[domain]
                if domain_status.training_status == 'complete':
                    ready_domains.append(domain)
        
        return ready_domains
    
    def get_phase_summary(self, phase_number: int) -> Dict[str, Any]:
        """Get comprehensive phase summary"""
        if phase_number not in self.phases:
            return {}
        
        phase_info = self.phases[phase_number]
        ready_domains = self.get_ready_domains(phase_number)
        
        return {
            "phase_number": phase_number,
            "status": phase_info.status,
            "total_domains": len(phase_info.domains),
            "ready_domains": len(ready_domains),
            "pending_domains": len(phase_info.domains) - len(ready_domains),
            "model_path": str(phase_info.model_path) if phase_info.model_path else None,
            "created_date": phase_info.created_date,
            "compression_config": phase_info.compression_config,
            "performance_metrics": phase_info.performance_metrics,
            "domains": [
                {
                    "name": domain,
                    "status": self.domain_status.get(domain, {}).get('training_status', 'unknown'),
                    "quality": self.domain_status.get(domain, {}).get('training_quality', 0.0)
                }
                for domain in phase_info.domains
            ]
        }
    
    def get_overall_summary(self) -> Dict[str, Any]:
        """Get overall project summary"""
        total_phases = len(self.phases)
        total_domains = len(self.domain_status)
        completed_phases = sum(1 for phase in self.phases.values() if phase.status == 'deployed')
        completed_domains = sum(1 for domain in self.domain_status.values() if domain.training_status == 'complete')
        
        return {
            "total_phases": total_phases,
            "total_domains": total_domains,
            "completed_phases": completed_phases,
            "completed_domains": completed_domains,
            "current_phase": self.current_phase,
            "phase_completion_rate": completed_phases / total_phases if total_phases > 0 else 0,
            "domain_completion_rate": completed_domains / total_domains if total_domains > 0 else 0,
            "phases": [self.get_phase_summary(phase_num) for phase_num in sorted(self.phases.keys())]
        }
    
    def advance_phase(self) -> int:
        """Advance to next phase"""
        self.current_phase += 1
        self._save_phase_data()
        logger.info(f"🚀 Advanced to Phase {self.current_phase}")
        return self.current_phase
    
    def deploy_phase(self, phase_number: int, deployment_target: str) -> bool:
        """Deploy phase to target"""
        
        try:
            if phase_number not in self.phases:
                logger.error(f"❌ Phase {phase_number} not found")
                return False
            
            phase_info = self.phases[phase_number]
            
            if not phase_info.model_path or not phase_info.model_path.exists():
                logger.error(f"❌ Phase {phase_number} model not found")
                return False
            
            # Copy model to deployment target
            target_path = Path(deployment_target) / phase_info.model_path.name
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(phase_info.model_path, target_path)
            
            # Update phase status
            phase_info.status = 'deployed'
            phase_info.deployment_target = deployment_target
            
            # Save data
            self._save_phase_data()
            
            logger.info(f"✅ Deployed Phase {phase_number} to {deployment_target}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to deploy Phase {phase_number}: {e}")
            return False
    
    def cleanup_phase(self, phase_number: int) -> bool:
        """Clean up phase resources"""
        
        try:
            if phase_number not in self.phases:
                logger.error(f"❌ Phase {phase_number} not found")
                return False
            
            phase_info = self.phases[phase_number]
            
            # Remove phase model if exists
            if phase_info.model_path and phase_info.model_path.exists():
                phase_info.model_path.unlink()
                logger.info(f"🗑️ Removed Phase {phase_number} model")
            
            # Remove phase from tracking
            del self.phases[phase_number]
            
            # Save data
            self._save_phase_data()
            
            logger.info(f"✅ Cleaned up Phase {phase_number}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup Phase {phase_number}: {e}")
            return False

def main():
    """Main function for testing phase manager"""
    
    # Create phase manager
    manager = PhaseManager()
    
    # Create Phase 1
    logger.info("🎯 Creating Phase 1...")
    manager.create_phase(1, ['healthcare', 'business'], {
        'quantization': 'Q4_K_M',
        'compression_type': 'standard'
    })
    
    # Update domain status
    manager.update_domain_status('healthcare', 'complete', 0.97, {
        'training_loss': 0.03,
        'validation_accuracy': 0.97
    })
    
    manager.update_domain_status('business', 'complete', 0.95, {
        'training_loss': 0.05,
        'validation_accuracy': 0.95
    })
    
    # Update phase status
    manager.update_phase_status(1, 'merging', Path('models/phase-1.gguf'), {
        'merge_time': 120.5,
        'model_size_mb': 850.2
    })
    
    # Show summaries
    phase_summary = manager.get_phase_summary(1)
    overall_summary = manager.get_overall_summary()
    
    logger.info(f"📊 Phase 1 Summary: {phase_summary}")
    logger.info(f"📈 Overall Summary: {overall_summary}")

if __name__ == "__main__":
    main() 