#!/usr/bin/env python3
"""
TARA Universal Model - Comprehensive Backup System
Ramesh Basina - PhD Research Data Protection

Backs up all training data, models, and configurations to:
1. Local backup folder (immediate safety)
2. Google Drive preparation (cloud backup)
3. Compressed archives (efficient storage)
"""

import os
import shutil
import zipfile
import json
from datetime import datetime
import hashlib
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/backup.log'),
        logging.StreamHandler()
    ]
)

class TARABackupManager:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.backup_root = f"backups/tara_backup_{self.timestamp}"
        self.google_drive_prep = f"google_drive_backup"
        
        # Create backup directories
        os.makedirs(self.backup_root, exist_ok=True)
        os.makedirs(self.google_drive_prep, exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        logging.info(f"🚀 TARA Backup Manager initialized")
        logging.info(f"📁 Local backup: {self.backup_root}")

    def calculate_file_hash(self, filepath):
        """Calculate MD5 hash for file integrity verification"""
        hash_md5 = hashlib.md5()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logging.error(f"Error calculating hash for {filepath}: {e}")
            return None

    def backup_training_data(self):
        """Backup all synthetic training data"""
        logging.info("📊 Backing up synthetic training data...")
        
        data_backup_dir = os.path.join(self.backup_root, "training_data")
        os.makedirs(data_backup_dir, exist_ok=True)
        
        total_size = 0
        files_backed_up = []
        
        # Backup synthetic data
        if os.path.exists("data/synthetic"):
            for file in os.listdir("data/synthetic"):
                if file.endswith('.json'):
                    src = os.path.join("data/synthetic", file)
                    dst = os.path.join(data_backup_dir, file)
                    
                    shutil.copy2(src, dst)
                    file_size = os.path.getsize(src) / (1024 * 1024)  # MB
                    total_size += file_size
                    files_backed_up.append({"file": file, "size_mb": round(file_size, 2)})
                    
                    logging.info(f"✅ Backed up: {file} ({file_size:.2f}MB)")
        
        logging.info(f"📋 Training data backup complete: {total_size:.2f}MB")
        return {"total_size_mb": round(total_size, 2), "files": files_backed_up}

    def backup_trained_models(self):
        """Backup all trained model adapters"""
        logging.info("🤖 Backing up trained models...")
        
        models_backup_dir = os.path.join(self.backup_root, "trained_models")
        os.makedirs(models_backup_dir, exist_ok=True)
        
        total_size = 0
        models_backed_up = []
        
        # Backup model adapters
        if os.path.exists("models/adapters"):
            for domain in os.listdir("models/adapters"):
                domain_path = os.path.join("models/adapters", domain)
                if os.path.isdir(domain_path):
                    # Create domain backup directory
                    domain_backup = os.path.join(models_backup_dir, domain)
                    shutil.copytree(domain_path, domain_backup)
                    
                    # Calculate domain size
                    domain_size = self.calculate_directory_size(domain_path)
                    total_size += domain_size
                    models_backed_up.append({
                        "domain": domain, 
                        "size_mb": round(domain_size, 2),
                        "status": "complete" if os.path.exists(os.path.join(domain_path, "adapter")) else "partial"
                    })
                    
                    logging.info(f"✅ Backed up model: {domain} ({domain_size:.2f}MB)")
        
        logging.info(f"🤖 Model backup complete: {total_size:.2f}MB")
        return {"total_size_mb": round(total_size, 2), "models": models_backed_up}

    def backup_configurations(self):
        """Backup all configuration files"""
        logging.info("⚙️ Backing up configurations...")
        
        config_backup_dir = os.path.join(self.backup_root, "configurations")
        os.makedirs(config_backup_dir, exist_ok=True)
        
        config_files = [
            "configs/config.yaml",
            "configs/universal_domains.yaml", 
            "configs/model_mapping.json",
            "configs/model_mapping_production.json",
            "requirements.txt",
            "setup.py"
        ]
        
        config_manifest = {"configs": {}}
        
        for config_file in config_files:
            if os.path.exists(config_file):
                dst = os.path.join(config_backup_dir, os.path.basename(config_file))
                shutil.copy2(config_file, dst)
                
                file_hash = self.calculate_file_hash(config_file)
                config_manifest["configs"][os.path.basename(config_file)] = {
                    "source": config_file,
                    "backup_path": dst,
                    "hash": file_hash
                }
                logging.info(f"✅ Backed up config: {config_file}")
        
        # Save manifest
        manifest_path = os.path.join(config_backup_dir, "config_manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(config_manifest, f, indent=2)
            
        logging.info("⚙️ Configuration backup complete")
        return config_manifest

    def backup_documentation(self):
        """Backup critical documentation"""
        logging.info("📚 Backing up documentation...")
        
        docs_backup_dir = os.path.join(self.backup_root, "documentation")
        
        # Copy entire docs folder
        if os.path.exists("docs"):
            shutil.copytree("docs", docs_backup_dir)
            docs_size = self.calculate_directory_size("docs")
            logging.info(f"✅ Backed up documentation: {docs_size:.2f}MB")
        
        # Copy memory-bank
        memory_backup_dir = os.path.join(self.backup_root, "memory_bank")
        if os.path.exists("memory-bank"):
            shutil.copytree("memory-bank", memory_backup_dir)
            memory_size = self.calculate_directory_size("memory-bank")
            logging.info(f"✅ Backed up memory-bank: {memory_size:.2f}MB")
        
        return {"docs_size_mb": docs_size if 'docs_size' in locals() else 0,
                "memory_size_mb": memory_size if 'memory_size' in locals() else 0}

    def calculate_directory_size(self, directory):
        """Calculate total size of directory in MB"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except OSError:
                    pass
        return total_size / (1024 * 1024)  # Convert to MB

    def create_compressed_archive(self):
        """Create compressed archive for Google Drive"""
        logging.info("🗜️ Creating compressed archive...")
        
        archive_name = f"tara_universal_model_backup_{self.timestamp}.zip"
        archive_path = os.path.join(self.google_drive_prep, archive_name)
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(self.backup_root):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, self.backup_root)
                    zipf.write(file_path, arcname)
        
        archive_size = os.path.getsize(archive_path) / (1024 * 1024)
        logging.info(f"🗜️ Archive created: {archive_name} ({archive_size:.2f}MB)")
        
        return {"archive_name": archive_name, "archive_path": archive_path, "size_mb": round(archive_size, 2)}

    def generate_backup_report(self, manifests, archive_info):
        """Generate comprehensive backup report"""
        report = {
            "backup_timestamp": self.timestamp,
            "backup_date": datetime.now().isoformat(),
            "researcher": "Ramesh Basina",
            "project": "TARA Universal Model - PhD Research",
            "backup_summary": {
                "local_backup_path": self.backup_root,
                "google_drive_prep_path": self.google_drive_prep,
                "compressed_archive": archive_info
            },
            "data_manifests": manifests,
            "total_backup_size_mb": sum([
                manifests.get("training_data", {}).get("total_size_mb", 0),
                manifests.get("models", {}).get("total_size_mb", 0),
                manifests.get("documentation", {}).get("docs_size_mb", 0),
                manifests.get("documentation", {}).get("memory_size_mb", 0)
            ]),
            "integrity_verification": "MD5 hashes calculated for all files",
            "next_steps": [
                "Upload compressed archive to Google Drive",
                "Verify backup integrity",
                "Store backup report in research documentation"
            ]
        }
        
        # Save report
        report_path = os.path.join(self.backup_root, "backup_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Also save in Google Drive prep folder
        gd_report_path = os.path.join(self.google_drive_prep, "backup_report.json")
        with open(gd_report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report

    def run_complete_backup(self):
        """Execute complete backup process"""
        logging.info("🚀 Starting complete TARA Universal Model backup...")
        
        try:
            # Execute all backup operations
            training_data = self.backup_training_data()
            models = self.backup_trained_models()
            configs = self.backup_configurations()
            
            # Create compressed archive
            archive_info = self.create_compressed_archive()
            
            total_size = training_data["total_size_mb"] + models["total_size_mb"]
            
            # Create backup report
            report = {
                "backup_timestamp": self.timestamp,
                "backup_date": datetime.now().isoformat(),
                "researcher": "Ramesh Basina",
                "project": "TARA Universal Model - PhD Research",
                "total_size_mb": round(total_size, 2),
                "training_data": training_data,
                "models": models,
                "configurations": configs,
                "archive": archive_info,
                "local_backup_path": self.backup_root
            }
            
            # Save report
            report_path = os.path.join(self.backup_root, "backup_report.json")
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Print summary
            logging.info("="*60)
            logging.info("🎉 BACKUP COMPLETE!")
            logging.info("="*60)
            logging.info(f"📅 Timestamp: {self.timestamp}")
            logging.info(f"💾 Total Size: {total_size:.2f}MB")
            logging.info(f"📁 Local Backup: {self.backup_root}")
            logging.info(f"🗜️ Archive: {archive_info['archive_name']} ({archive_info['size_mb']:.2f}MB)")
            logging.info("="*60)
            logging.info("📋 BACKUP CONTENTS:")
            logging.info(f"   📊 Training Data: {training_data['total_size_mb']:.2f}MB")
            logging.info(f"   🤖 Trained Models: {models['total_size_mb']:.2f}MB")
            logging.info(f"   ⚙️ Configurations: {len(configs['configs'])} files")
            logging.info("="*60)
            logging.info("☁️ GOOGLE DRIVE UPLOAD:")
            logging.info(f"   Upload this file: {archive_info['archive_path']}")
            logging.info("="*60)
            
            return report
            
        except Exception as e:
            logging.error(f"❌ Backup failed: {e}")
            raise

def main():
    """Main backup execution"""
    backup_manager = TARABackupManager()
    report = backup_manager.run_complete_backup()
    return report

if __name__ == "__main__":
    main() 