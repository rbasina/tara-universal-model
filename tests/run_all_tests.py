#!/usr/bin/env python3
"""
🧪 Comprehensive Test Runner for TARA Universal Model
Runs all tests with coverage reporting and generates detailed reports
"""

import unittest
import sys
import os
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts" / "conversion"))
sys.path.insert(0, str(project_root / "scripts" / "training"))
sys.path.insert(0, str(project_root / "scripts" / "monitoring"))

def run_pytest_with_coverage():
    """Run pytest with coverage reporting"""
    print("🧪 Running pytest with coverage...")
    
    # Install coverage if not available
    try:
        import coverage
    except ImportError:
        print("📦 Installing coverage package...")
        subprocess.run([sys.executable, "-m", "pip", "install", "coverage"], check=True)
    
    # Run pytest with coverage
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "--cov=scripts",
        "--cov=tara_universal_model",
        "--cov-report=html:tests/coverage_html",
        "--cov-report=term-missing",
        "--cov-report=json:tests/coverage.json",
        "--verbose",
        "--tb=short"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0

def run_unittest_tests():
    """Run unittest-based tests"""
    print("🧪 Running unittest-based tests...")
    
    # Import test modules
    test_modules = [
        "tests.test_gguf_conversion_system",
        "tests.test_training_recovery", 
        "tests.test_connection_recovery"
    ]
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for module_name in test_modules:
        try:
            module = __import__(module_name, fromlist=[''])
            tests = loader.loadTestsFromModule(module)
            suite.addTests(tests)
            print(f"✅ Loaded tests from {module_name}")
        except ImportError as e:
            print(f"❌ Failed to load {module_name}: {e}")
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def generate_test_report():
    """Generate comprehensive test report"""
    print("📊 Generating test report...")
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "project": "TARA Universal Model",
        "test_summary": {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "skipped": 0,
            "success_rate": 0.0
        },
        "coverage": {
            "overall_coverage": 0.0,
            "modules": {}
        },
        "test_categories": {
            "gguf_conversion": {
                "tests": 0,
                "passed": 0,
                "failed": 0
            },
            "training_recovery": {
                "tests": 0,
                "passed": 0,
                "failed": 0
            },
            "connection_recovery": {
                "tests": 0,
                "passed": 0,
                "failed": 0
            },
            "integration": {
                "tests": 0,
                "passed": 0,
                "failed": 0
            }
        },
        "performance_metrics": {
            "test_execution_time": 0.0,
            "memory_usage": 0.0,
            "cpu_usage": 0.0
        }
    }
    
    # Save report
    report_file = Path("tests/test_report.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Test report saved to {report_file}")
    return report

def check_code_quality():
    """Check code quality with linting tools"""
    print("🔍 Checking code quality...")
    
    # Install linting tools if needed
    try:
        import flake8
    except ImportError:
        print("📦 Installing flake8...")
        subprocess.run([sys.executable, "-m", "pip", "install", "flake8"], check=True)
    
    try:
        import black
    except ImportError:
        print("📦 Installing black...")
        subprocess.run([sys.executable, "-m", "pip", "install", "black"], check=True)
    
    # Run flake8
    print("Running flake8...")
    flake8_result = subprocess.run([
        sys.executable, "-m", "flake8",
        "scripts/",
        "tara_universal_model/",
        "--max-line-length=120",
        "--ignore=E501,W503"
    ], capture_output=True, text=True)
    
    if flake8_result.returncode == 0:
        print("✅ Code style check passed")
    else:
        print("❌ Code style issues found:")
        print(flake8_result.stdout)
    
    # Run black check
    print("Running black check...")
    black_result = subprocess.run([
        sys.executable, "-m", "black",
        "--check",
        "scripts/",
        "tara_universal_model/"
    ], capture_output=True, text=True)
    
    if black_result.returncode == 0:
        print("✅ Code formatting check passed")
    else:
        print("❌ Code formatting issues found:")
        print(black_result.stdout)
    
    return flake8_result.returncode == 0 and black_result.returncode == 0

def run_security_checks():
    """Run security checks"""
    print("🔒 Running security checks...")
    
    # Install security tools if needed
    try:
        import bandit
    except ImportError:
        print("📦 Installing bandit...")
        subprocess.run([sys.executable, "-m", "pip", "install", "bandit"], check=True)
    
    # Run bandit
    bandit_result = subprocess.run([
        sys.executable, "-m", "bandit",
        "-r", "scripts/",
        "-r", "tara_universal_model/",
        "-f", "json",
        "-o", "tests/security_report.json"
    ], capture_output=True, text=True)
    
    if bandit_result.returncode == 0:
        print("✅ Security check passed")
    else:
        print("⚠️ Security issues found (check tests/security_report.json)")
    
    return bandit_result.returncode == 0

def create_test_dashboard():
    """Create test dashboard HTML"""
    print("📊 Creating test dashboard...")
    
    dashboard_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TARA Universal Model - Test Dashboard</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        .header p {
            margin: 10px 0 0 0;
            opacity: 0.9;
        }
        .content {
            padding: 30px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #667eea;
        }
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        .stat-label {
            color: #666;
            margin-top: 5px;
        }
        .test-categories {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .category-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #e9ecef;
        }
        .category-title {
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
        }
        .progress-bar {
            background: #e9ecef;
            border-radius: 10px;
            height: 20px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #28a745, #20c997);
            transition: width 0.3s ease;
        }
        .coverage-section {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .coverage-title {
            font-size: 1.3em;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
        }
        .footer {
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            border-top: 1px solid #e9ecef;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-passed { background: #28a745; }
        .status-failed { background: #dc3545; }
        .status-warning { background: #ffc107; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧪 TARA Universal Model</h1>
            <p>Test Dashboard & Quality Assurance</p>
        </div>
        
        <div class="content">
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number" id="total-tests">0</div>
                    <div class="stat-label">Total Tests</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="passed-tests">0</div>
                    <div class="stat-label">Passed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="failed-tests">0</div>
                    <div class="stat-label">Failed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="coverage-percent">0%</div>
                    <div class="stat-label">Code Coverage</div>
                </div>
            </div>
            
            <div class="test-categories">
                <div class="category-card">
                    <div class="category-title">
                        <span class="status-indicator status-passed"></span>
                        GGUF Conversion System
                    </div>
                    <div>Tests: <span id="gguf-tests">0</span></div>
                    <div>Passed: <span id="gguf-passed">0</span></div>
                    <div>Failed: <span id="gguf-failed">0</span></div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="gguf-progress" style="width: 0%"></div>
                    </div>
                </div>
                
                <div class="category-card">
                    <div class="category-title">
                        <span class="status-indicator status-passed"></span>
                        Training Recovery
                    </div>
                    <div>Tests: <span id="recovery-tests">0</span></div>
                    <div>Passed: <span id="recovery-passed">0</span></div>
                    <div>Failed: <span id="recovery-failed">0</span></div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="recovery-progress" style="width: 0%"></div>
                    </div>
                </div>
                
                <div class="category-card">
                    <div class="category-title">
                        <span class="status-indicator status-passed"></span>
                        Connection Recovery
                    </div>
                    <div>Tests: <span id="connection-tests">0</span></div>
                    <div>Passed: <span id="connection-passed">0</span></div>
                    <div>Failed: <span id="connection-failed">0</span></div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="connection-progress" style="width: 0%"></div>
                    </div>
                </div>
                
                <div class="category-card">
                    <div class="category-title">
                        <span class="status-indicator status-passed"></span>
                        Integration Tests
                    </div>
                    <div>Tests: <span id="integration-tests">0</span></div>
                    <div>Passed: <span id="integration-passed">0</span></div>
                    <div>Failed: <span id="integration-failed">0</span></div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="integration-progress" style="width: 0%"></div>
                    </div>
                </div>
            </div>
            
            <div class="coverage-section">
                <div class="coverage-title">Code Coverage Details</div>
                <div id="coverage-details">
                    <p>Loading coverage data...</p>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Generated on <span id="timestamp"></span></p>
            <p>TARA Universal Model - Trinity Architecture Implementation</p>
        </div>
    </div>
    
    <script>
        // Load test data
        fetch('test_report.json')
            .then(response => response.json())
            .then(data => {
                updateDashboard(data);
            })
            .catch(error => {
                console.error('Error loading test data:', error);
            });
        
        function updateDashboard(data) {
            // Update timestamp
            document.getElementById('timestamp').textContent = new Date().toLocaleString();
            
            // Update overall stats
            const summary = data.test_summary;
            document.getElementById('total-tests').textContent = summary.total_tests;
            document.getElementById('passed-tests').textContent = summary.passed;
            document.getElementById('failed-tests').textContent = summary.failed;
            document.getElementById('coverage-percent').textContent = data.coverage.overall_coverage.toFixed(1) + '%';
            
            // Update categories
            updateCategory('gguf', data.test_categories.gguf_conversion);
            updateCategory('recovery', data.test_categories.training_recovery);
            updateCategory('connection', data.test_categories.connection_recovery);
            updateCategory('integration', data.test_categories.integration);
            
            // Update coverage details
            updateCoverageDetails(data.coverage);
        }
        
        function updateCategory(prefix, category) {
            document.getElementById(prefix + '-tests').textContent = category.tests;
            document.getElementById(prefix + '-passed').textContent = category.passed;
            document.getElementById(prefix + '-failed').textContent = category.failed;
            
            const progress = category.tests > 0 ? (category.passed / category.tests) * 100 : 0;
            document.getElementById(prefix + '-progress').style.width = progress + '%';
            
            // Update status indicator
            const indicator = document.querySelector(`#${prefix}-progress`).parentElement.parentElement.querySelector('.status-indicator');
            if (category.failed === 0) {
                indicator.className = 'status-indicator status-passed';
            } else if (category.passed > 0) {
                indicator.className = 'status-indicator status-warning';
            } else {
                indicator.className = 'status-indicator status-failed';
            }
        }
        
        function updateCoverageDetails(coverage) {
            const details = document.getElementById('coverage-details');
            let html = '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">';
            
            for (const [module, moduleCoverage] of Object.entries(coverage.modules)) {
                html += `
                    <div style="background: white; padding: 15px; border-radius: 5px; border: 1px solid #e9ecef;">
                        <div style="font-weight: bold; margin-bottom: 5px;">${module}</div>
                        <div style="color: #667eea; font-size: 1.2em;">${moduleCoverage.toFixed(1)}%</div>
                    </div>
                `;
            }
            
            html += '</div>';
            details.innerHTML = html;
        }
    </script>
</body>
</html>
    """
    
    dashboard_file = Path("tests/test_dashboard.html")
    with open(dashboard_file, "w") as f:
        f.write(dashboard_html)
    
    print(f"✅ Test dashboard created: {dashboard_file}")

def main():
    """Main test runner function"""
    print("🧪 TARA Universal Model - Comprehensive Test Suite")
    print("=" * 60)
    
    start_time = time.time()
    all_passed = True
    
    # Create tests directory if it doesn't exist
    tests_dir = Path("tests")
    tests_dir.mkdir(exist_ok=True)
    
    # Run different types of tests
    print("\n1️⃣ Running pytest with coverage...")
    pytest_success = run_pytest_with_coverage()
    all_passed = all_passed and pytest_success
    
    print("\n2️⃣ Running unittest-based tests...")
    unittest_success = run_unittest_tests()
    all_passed = all_passed and unittest_success
    
    print("\n3️⃣ Checking code quality...")
    quality_success = check_code_quality()
    all_passed = all_passed and quality_success
    
    print("\n4️⃣ Running security checks...")
    security_success = run_security_checks()
    all_passed = all_passed and security_success
    
    print("\n5️⃣ Generating test report...")
    generate_test_report()
    
    print("\n6️⃣ Creating test dashboard...")
    create_test_dashboard()
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎯 TEST EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Total Execution Time: {execution_time:.2f} seconds")
    print(f"Pytest Coverage: {'✅ PASSED' if pytest_success else '❌ FAILED'}")
    print(f"Unittest Tests: {'✅ PASSED' if unittest_success else '❌ FAILED'}")
    print(f"Code Quality: {'✅ PASSED' if quality_success else '❌ FAILED'}")
    print(f"Security Checks: {'✅ PASSED' if security_success else '❌ FAILED'}")
    print(f"Overall Result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    print("=" * 60)
    
    print("\n📊 Test Reports Generated:")
    print("  - tests/coverage_html/ (HTML coverage report)")
    print("  - tests/coverage.json (JSON coverage data)")
    print("  - tests/test_report.json (Comprehensive test report)")
    print("  - tests/security_report.json (Security analysis)")
    print("  - tests/test_dashboard.html (Interactive dashboard)")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 