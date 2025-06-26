#!/usr/bin/env python
"""
TARA Universal Model - Test Recovery Feature
This script tests the training recovery feature by creating a mock recovery state
and calling the recovery endpoint.
"""

import os
import sys
import json
import requests
from datetime import datetime
from pathlib import Path

# Create a mock recovery state
def create_mock_recovery_state():
    """Create a mock recovery state file for testing"""
    state = {
        "domains": "education,creative,leadership",
        "model": "Qwen2.5-3B-Instruct",
        "start_time": datetime.now().isoformat(),
        "last_check": datetime.now().isoformat(),
        "checkpoints": {
            "education": "models/adapters/education_qwen25/checkpoint-100",
            "creative": "models/adapters/creative_qwen25/checkpoint-100", 
            "leadership": "models/adapters/leadership_qwen25/checkpoint-100"
        }
    }
    
    with open("training_recovery_state.json", "w") as f:
        json.dump(state, f, indent=2)
    
    print(f"✅ Created mock recovery state: {state}")
    return state

def test_recovery_endpoint():
    """Test the recovery endpoint"""
    try:
        response = requests.post("http://localhost:8001/recover_training")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Recovery endpoint response: {data}")
            return data
        else:
            print(f"❌ Recovery endpoint failed with status code: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"❌ Error calling recovery endpoint: {e}")
        return None

def main():
    """Main function"""
    print("🔄 TARA Universal Model - Testing Recovery Feature")
    
    # Create mock recovery state
    state = create_mock_recovery_state()
    
    # Test recovery endpoint
    response = test_recovery_endpoint()
    
    if response and response.get("success"):
        print("✅ Recovery test successful!")
        print(f"Domains: {response.get('domains')}")
        print(f"Model: {response.get('model')}")
        print(f"Checkpoints found: {response.get('checkpoints_found')}")
    else:
        print("❌ Recovery test failed!")
    
if __name__ == "__main__":
    main() 