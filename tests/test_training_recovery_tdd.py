#!/usr/bin/env python3
"""
TDD Test Suite for Training Recovery Scenarios
Focus: Education Domain Training with All Edge Cases
"""

import pytest
import json
import os
import shutil
import tempfile
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the actual implementation
from scripts.training.training_recovery_utils import (
    detect_checkpoint,
    detect_state_inconsistency,
    correct_state_file,
    validate_checkpoint_integrity,
    create_backup_checkpoint,
    update_training_progress,
    find_best_checkpoint,
    complete_training_workflow
)

class TestTrainingRecoveryTDD:
    """TDD Test Suite for Training Recovery Scenarios"""
    
    @pytest.fixture
    def temp_project_dir(self):
        """Create temporary project directory for testing"""
        temp_dir = tempfile.mkdtemp()
        project_structure = {
            'models': {
                'education': {
                    'meetara_trinity_phase_efficient_core_processing': {}
                },
                'adapters': {
                    'education_qwen2.5-3b-instruct': {
                        'checkpoint-134': {
                            'trainer_state.json': json.dumps({
                                'global_step': 134,
                                'epoch': 0.335,
                                'best_model_checkpoint': None
                            }),
                            'pytorch_model.bin': b'fake_model_data',
                            'config.json': json.dumps({'model_type': 'test'})
                        }
                    }
                },
                'checkpoints': {},
                'checkpoints_backup': {}
            },
            'training_state': {
                'education_training_state.json': json.dumps({
                    'domain': 'education',
                    'base_model': 'Qwen/Qwen2.5-3B-Instruct',
                    'output_dir': 'models/education/meetara_trinity_phase_efficient_core_processing',
                    'resume_checkpoint': None,
                    'start_time': '2025-06-26T22:14:21.314673',
                    'status': 'loading_model',
                    'retries': 0
                })
            },
            'logs': {},
            'configs': {
                'config.yaml': 'batch_size: 2\nnum_epochs: 3\nlearning_rate: 2e-4'
            }
        }
        
        self._create_project_structure(temp_dir, project_structure)
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def _create_project_structure(self, base_dir, structure):
        """Create project directory structure for testing"""
        for path, content in structure.items():
            full_path = os.path.join(base_dir, path)
            if isinstance(content, dict):
                os.makedirs(full_path, exist_ok=True)
                self._create_project_structure(full_path, content)
            elif isinstance(content, str):
                with open(full_path, 'w') as f:
                    f.write(content)
            elif isinstance(content, bytes):
                # For model files, create large enough files
                if 'pytorch_model.bin' in full_path:
                    self._create_large_model_file(full_path)
                else:
                    with open(full_path, 'wb') as f:
                        f.write(content)
    
    def _create_large_model_file(self, file_path):
        """Create a model file that's large enough to pass validation (>1KB)"""
        # Create 2KB of fake model data
        fake_model_data = b'fake_model_data_' * 128  # 16 bytes * 128 = 2048 bytes
        with open(file_path, 'wb') as f:
            f.write(fake_model_data)
    
    def test_001_checkpoint_detection_with_valid_checkpoint(self, temp_project_dir):
        """Test: Should detect valid checkpoint and update state accordingly"""
        # Act
        result = detect_checkpoint('education', temp_project_dir)
        
        # Assert
        assert result['found'] == True
        assert 'checkpoint-134' in result['checkpoint_path']
        assert result['step'] == 134
        assert result['epoch'] == 0.335
    
    def test_002_state_file_inconsistency_detection(self, temp_project_dir):
        """Test: Should detect when state file doesn't match actual checkpoint status"""
        # Act
        inconsistency = detect_state_inconsistency('education', temp_project_dir)
        
        # Assert
        assert inconsistency['detected'] == True
        assert inconsistency['state_status'] == 'loading_model'
        assert inconsistency['has_checkpoint'] == True
        assert inconsistency['checkpoint_step'] == 134
    
    def test_003_state_file_correction(self, temp_project_dir):
        """Test: Should correct state file to match actual checkpoint status"""
        # Arrange
        checkpoint_path = os.path.join(temp_project_dir, 'models/adapters/education_qwen2.5-3b-instruct/checkpoint-134')
        
        # Act
        success = correct_state_file('education', checkpoint_path, temp_project_dir)
        
        # Assert
        assert success == True
        
        # Verify state file was corrected
        state_file = os.path.join(temp_project_dir, 'training_state/education_training_state.json')
        with open(state_file, 'r') as f:
            corrected_state = json.load(f)
        
        assert corrected_state['status'] == 'ready_to_resume'
        assert corrected_state['resume_checkpoint'] == checkpoint_path
        assert corrected_state['last_progress'] == 134
    
    def test_004_checkpoint_validation_integrity(self, temp_project_dir):
        """Test: Should validate checkpoint integrity before resuming"""
        # Arrange
        checkpoint_path = os.path.join(temp_project_dir, 'models/adapters/education_qwen2.5-3b-instruct/checkpoint-134')
        
        # Act
        validation = validate_checkpoint_integrity(checkpoint_path)
        
        # Assert
        assert validation['valid'] == True
        assert validation['has_trainer_state'] == True
        assert validation['has_model_file'] == True
        assert validation['has_config'] == True
        assert validation['step'] == 134
    
    def test_005_corrupted_checkpoint_detection(self, temp_project_dir):
        """Test: Should detect corrupted checkpoint and handle gracefully"""
        # Arrange
        corrupted_checkpoint = os.path.join(temp_project_dir, 'models/adapters/education_qwen2.5-3b-instruct/corrupted-checkpoint')
        os.makedirs(corrupted_checkpoint, exist_ok=True)
        
        # Create corrupted trainer_state.json
        with open(os.path.join(corrupted_checkpoint, 'trainer_state.json'), 'w') as f:
            f.write('{"invalid": "json"')  # Invalid JSON
        
        # Act
        validation = validate_checkpoint_integrity(corrupted_checkpoint)
        
        # Assert
        assert validation['valid'] == False
        assert 'corrupted' in validation['error'].lower()
    
    def test_006_missing_checkpoint_files(self, temp_project_dir):
        """Test: Should detect missing essential checkpoint files"""
        # Arrange
        incomplete_checkpoint = os.path.join(temp_project_dir, 'models/adapters/education_qwen2.5-3b-instruct/incomplete-checkpoint')
        os.makedirs(incomplete_checkpoint, exist_ok=True)
        
        # Only create trainer_state.json, missing model file
        with open(os.path.join(incomplete_checkpoint, 'trainer_state.json'), 'w') as f:
            f.write(json.dumps({'global_step': 100}))
        
        # Act
        validation = validate_checkpoint_integrity(incomplete_checkpoint)
        
        # Assert
        assert validation['valid'] == False
        assert 'missing' in validation['error'].lower()
    
    def test_007_backup_checkpoint_creation(self, temp_project_dir):
        """Test: Should create backup checkpoints before training"""
        # Arrange
        checkpoint_path = os.path.join(temp_project_dir, 'models/adapters/education_qwen2.5-3b-instruct/checkpoint-134')
        backup_dir = os.path.join(temp_project_dir, 'models/checkpoints_backup')
        
        # Act
        backup_path = create_backup_checkpoint(checkpoint_path, backup_dir)
        
        # Assert
        assert backup_path is not None
        assert os.path.exists(backup_path)
        assert os.path.exists(os.path.join(backup_path, 'trainer_state.json'))
        assert os.path.exists(os.path.join(backup_path, 'pytorch_model.bin'))
    
    def test_008_training_progress_monitoring(self, temp_project_dir):
        """Test: Should monitor training progress and update state files"""
        # Act
        success = update_training_progress('education', 200, 0.5, temp_project_dir)
        
        # Assert
        assert success == True
        
        # Verify state file was updated
        state_file = os.path.join(temp_project_dir, 'training_state/education_training_state.json')
        with open(state_file, 'r') as f:
            updated_state = json.load(f)
        
        assert updated_state['status'] == 'training'
        assert updated_state['last_progress'] == 200
        assert updated_state['current_epoch'] == 0.5
    
    def test_009_multiple_checkpoint_locations(self, temp_project_dir):
        """Test: Should check multiple checkpoint locations for recovery"""
        # Arrange
        # Create checkpoints in different locations
        locations = [
            'models/education/meetara_trinity_phase_efficient_core_processing/checkpoint-100',
            'models/adapters/education_qwen2.5-3b-instruct/checkpoint-134',
            'models/checkpoints/education/checkpoint-150'
        ]
        
        for location in locations:
            full_path = os.path.join(temp_project_dir, location)
            os.makedirs(full_path, exist_ok=True)
            with open(os.path.join(full_path, 'trainer_state.json'), 'w') as f:
                f.write(json.dumps({'global_step': int(location.split('-')[-1])}))
            with open(os.path.join(full_path, 'pytorch_model.bin'), 'wb') as f:
                f.write(b'fake_model_data')
            with open(os.path.join(full_path, 'config.json'), 'w') as f:
                f.write(json.dumps({'model_type': 'test'}))
        
        # Act
        best_checkpoint = find_best_checkpoint('education', temp_project_dir)
        
        # Assert
        assert best_checkpoint['path'] == os.path.join(temp_project_dir, 'models/checkpoints/education/checkpoint-150')
        assert best_checkpoint['step'] == 150
    
    def test_010_complete_training_workflow(self, temp_project_dir):
        """Test: Complete end-to-end training workflow with all safety measures"""
        # Act
        result = complete_training_workflow('education', temp_project_dir)
        
        # Assert
        assert result['success'] == True
        assert result['checkpoint_found'] == True
        assert result['checkpoint_validated'] == True
        assert result['backup_created'] == True
        assert result['state_corrected'] == True

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 