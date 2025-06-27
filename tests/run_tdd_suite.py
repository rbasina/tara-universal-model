#!/usr/bin/env python3
"""
🧪 TDD Test Suite Runner for TARA Universal Model
Comprehensive test execution with coverage reporting and quality checks
"""

import unittest
import sys
import os
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime
import coverage

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts" / "conversion"))
sys.path.insert(0, str(project_root / "scripts" / "training"))
sys.path.insert(0, str(project_root / "scripts" / "monitoring"))

class TDDRunner:
    """TDD Test Suite Runner with comprehensive coverage"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.tests_dir = self.project_root / "tests"
        self.reports_dir = self.tests_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        # Test results
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "skipped": 0,
            "coverage": 0.0,
            "test_categories": {},
            "performance_metrics": {},
            "quality_metrics": {}
        }
    
    def run_all_tests(self):
        """Run all tests with coverage"""
        print("🧪 TARA Universal Model - TDD Test Suite")
        print("=" * 60)
        
        start_time = time.time()
        
        # 1. Run unit tests
        print("\n1️⃣ Running Unit Tests...")
        unit_results = self._run_unit_tests()
        
        # 2. Run integration tests
        print("\n2️⃣ Running Integration Tests...")
        integration_results = self._run_integration_tests()
        
        # 3. Run system tests
        print("\n3️⃣ Running System Tests...")
        system_results = self._run_system_tests()
        
        # 4. Generate coverage report
        print("\n4️⃣ Generating Coverage Report...")
        coverage_results = self._generate_coverage_report()
        
        # 5. Run quality checks
        print("\n5️⃣ Running Quality Checks...")
        quality_results = self._run_quality_checks()
        
        # 6. Generate comprehensive report
        print("\n6️⃣ Generating Comprehensive Report...")
        self._generate_comprehensive_report(start_time)
        
        # 7. Create test dashboard
        print("\n7️⃣ Creating Test Dashboard...")
        self._create_test_dashboard()
        
        return self.test_results
    
    def _run_unit_tests(self):
        """Run unit tests"""
        test_modules = [
            "test_gguf_conversion_system",
            "test_training_recovery",
            "test_connection_recovery"
        ]
        
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        for module_name in test_modules:
            try:
                module = __import__(f"tests.{module_name}", fromlist=[''])
                tests = loader.loadTestsFromModule(module)
                suite.addTests(tests)
                print(f"  ✅ Loaded {module_name}")
            except ImportError as e:
                print(f"  ❌ Failed to load {module_name}: {e}")
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Store results
        self.test_results["test_categories"]["unit_tests"] = {
            "total": result.testsRun,
            "passed": result.testsRun - len(result.failures) - len(result.errors),
            "failed": len(result.failures),
            "errors": len(result.errors)
        }
        
        return result.wasSuccessful()
    
    def _run_integration_tests(self):
        """Run integration tests"""
        # Integration tests focus on component interactions
        integration_tests = [
            "test_gguf_conversion_system.TestIntegration",
            "test_training_recovery.TestIntegration",
            "test_connection_recovery.TestIntegration"
        ]
        
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        for test_class in integration_tests:
            try:
                module_name, class_name = test_class.rsplit('.', 1)
                module = __import__(f"tests.{module_name}", fromlist=[class_name])
                test_class_obj = getattr(module, class_name)
                tests = loader.loadTestsFromTestCase(test_class_obj)
                suite.addTests(tests)
                print(f"  ✅ Loaded {test_class}")
            except (ImportError, AttributeError) as e:
                print(f"  ❌ Failed to load {test_class}: {e}")
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Store results
        self.test_results["test_categories"]["integration_tests"] = {
            "total": result.testsRun,
            "passed": result.testsRun - len(result.failures) - len(result.errors),
            "failed": len(result.failures),
            "errors": len(result.errors)
        }
        
        return result.wasSuccessful()
    
    def _run_system_tests(self):
        """Run system tests"""
        # System tests focus on end-to-end functionality
        system_tests = [
            "test_gguf_conversion_system.TestUniversalGGUFFactory",
            "test_training_recovery.TestTrainingRecovery",
            "test_connection_recovery.TestConnectionRecovery"
        ]
        
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        for test_class in system_tests:
            try:
                module_name, class_name = test_class.rsplit('.', 1)
                module = __import__(f"tests.{module_name}", fromlist=[class_name])
                test_class_obj = getattr(module, class_name)
                tests = loader.loadTestsFromTestCase(test_class_obj)
                suite.addTests(tests)
                print(f"  ✅ Loaded {test_class}")
            except (ImportError, AttributeError) as e:
                print(f"  ❌ Failed to load {test_class}: {e}")
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Store results
        self.test_results["test_categories"]["system_tests"] = {
            "total": result.testsRun,
            "passed": result.testsRun - len(result.failures) - len(result.errors),
            "failed": len(result.failures),
            "errors": len(result.errors)
        }
        
        return result.wasSuccessful()
    
    def _generate_coverage_report(self):
        """Generate coverage report"""
        try:
            # Start coverage
            cov = coverage.Coverage()
            cov.start()
            
            # Import and run test modules to measure coverage
            test_modules = [
                "tests.test_gguf_conversion_system",
                "tests.test_training_recovery",
                "tests.test_connection_recovery"
            ]
            
            for module_name in test_modules:
                try:
                    __import__(module_name)
                except ImportError:
                    pass
            
            # Stop coverage and generate report
            cov.stop()
            cov.save()
            
            # Generate reports
            cov.html_report(directory=str(self.reports_dir / "coverage_html"))
            cov.xml_report(outfile=str(self.reports_dir / "coverage.xml"))
            
            # Get coverage percentage
            total_coverage = cov.report()
            self.test_results["coverage"] = total_coverage
            
            print(f"  ✅ Coverage: {total_coverage:.1f}%")
            return True
            
        except Exception as e:
            print(f"  ❌ Coverage generation failed: {e}")
            return False
    
    def _run_quality_checks(self):
        """Run code quality checks"""
        quality_results = {}
        
        # 1. Flake8 (style checking)
        try:
            result = subprocess.run([
                sys.executable, "-m", "flake8",
                "scripts/",
                "tara_universal_model/",
                "--max-line-length=120",
                "--ignore=E501,W503",
                "--count"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                quality_results["flake8"] = {"status": "passed", "issues": 0}
                print("  ✅ Flake8: Passed")
            else:
                issues = int(result.stdout.strip().split('\n')[-1]) if result.stdout else 0
                quality_results["flake8"] = {"status": "failed", "issues": issues}
                print(f"  ❌ Flake8: {issues} issues found")
                
        except Exception as e:
            quality_results["flake8"] = {"status": "error", "error": str(e)}
            print(f"  ❌ Flake8: Error - {e}")
        
        # 2. Black (formatting check)
        try:
            result = subprocess.run([
                sys.executable, "-m", "black",
                "--check",
                "scripts/",
                "tara_universal_model/"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                quality_results["black"] = {"status": "passed"}
                print("  ✅ Black: Passed")
            else:
                quality_results["black"] = {"status": "failed"}
                print("  ❌ Black: Formatting issues found")
                
        except Exception as e:
            quality_results["black"] = {"status": "error", "error": str(e)}
            print(f"  ❌ Black: Error - {e}")
        
        # 3. Bandit (security check)
        try:
            result = subprocess.run([
                sys.executable, "-m", "bandit",
                "-r", "scripts/",
                "-r", "tara_universal_model/",
                "-f", "json",
                "-o", str(self.reports_dir / "security_report.json")
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                quality_results["bandit"] = {"status": "passed", "issues": 0}
                print("  ✅ Bandit: Passed")
            else:
                quality_results["bandit"] = {"status": "failed"}
                print("  ⚠️ Bandit: Security issues found")
                
        except Exception as e:
            quality_results["bandit"] = {"status": "error", "error": str(e)}
            print(f"  ❌ Bandit: Error - {e}")
        
        self.test_results["quality_metrics"] = quality_results
        return quality_results
    
    def _generate_comprehensive_report(self, start_time):
        """Generate comprehensive test report"""
        execution_time = time.time() - start_time
        
        # Calculate totals
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_errors = 0
        
        for category, results in self.test_results["test_categories"].items():
            total_tests += results["total"]
            total_passed += results["passed"]
            total_failed += results["failed"]
            total_errors += results["errors"]
        
        self.test_results.update({
            "total_tests": total_tests,
            "passed": total_passed,
            "failed": total_failed,
            "errors": total_errors,
            "skipped": total_tests - total_passed - total_failed - total_errors,
            "performance_metrics": {
                "execution_time_seconds": execution_time,
                "tests_per_second": total_tests / execution_time if execution_time > 0 else 0
            }
        })
        
        # Save report
        report_file = self.reports_dir / "tdd_test_report.json"
        with open(report_file, "w") as f:
            json.dump(self.test_results, f, indent=2)
        
        print(f"  ✅ Comprehensive report saved: {report_file}")
    
    def _create_test_dashboard(self):
        """Create interactive test dashboard"""
        dashboard_html = self._generate_dashboard_html()
        dashboard_file = self.reports_dir / "tdd_dashboard.html"
        
        with open(dashboard_file, "w") as f:
            f.write(dashboard_html)
        
        print(f"  ✅ Test dashboard created: {dashboard_file}")
    
    def _generate_dashboard_html(self):
        """Generate dashboard HTML"""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TARA Universal Model - TDD Test Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .content {{
            padding: 30px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #667eea;
        }}
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .stat-label {{
            color: #666;
            margin-top: 5px;
        }}
        .test-categories {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .category-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #e9ecef;
        }}
        .category-title {{
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
        }}
        .progress-bar {{
            background: #e9ecef;
            border-radius: 10px;
            height: 20px;
            overflow: hidden;
            margin: 10px 0;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #28a745, #20c997);
            transition: width 0.3s ease;
        }}
        .quality-section {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .quality-title {{
            font-size: 1.3em;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
        }}
        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            border-top: 1px solid #e9ecef;
        }}
        .status-indicator {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }}
        .status-passed {{ background: #28a745; }}
        .status-failed {{ background: #dc3545; }}
        .status-warning {{ background: #ffc107; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧪 TARA Universal Model</h1>
            <p>TDD Test Dashboard & Quality Assurance</p>
        </div>
        
        <div class="content">
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number" id="total-tests">{self.test_results.get('total_tests', 0)}</div>
                    <div class="stat-label">Total Tests</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="passed-tests">{self.test_results.get('passed', 0)}</div>
                    <div class="stat-label">Passed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="failed-tests">{self.test_results.get('failed', 0)}</div>
                    <div class="stat-label">Failed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="coverage-percent">{self.test_results.get('coverage', 0):.1f}%</div>
                    <div class="stat-label">Code Coverage</div>
                </div>
            </div>
            
            <div class="test-categories">
                {self._generate_category_cards()}
            </div>
            
            <div class="quality-section">
                <div class="quality-title">Code Quality Metrics</div>
                {self._generate_quality_metrics()}
            </div>
        </div>
        
        <div class="footer">
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>TARA Universal Model - Trinity Architecture Implementation</p>
        </div>
    </div>
</body>
</html>
        """
    
    def _generate_category_cards(self):
        """Generate category cards HTML"""
        cards_html = ""
        
        for category, results in self.test_results.get("test_categories", {}).items():
            total = results.get("total", 0)
            passed = results.get("passed", 0)
            failed = results.get("failed", 0)
            errors = results.get("errors", 0)
            
            progress = (passed / total * 100) if total > 0 else 0
            status_class = "status-passed" if failed == 0 and errors == 0 else "status-warning" if passed > 0 else "status-failed"
            
            cards_html += f"""
                <div class="category-card">
                    <div class="category-title">
                        <span class="status-indicator {status_class}"></span>
                        {category.replace('_', ' ').title()}
                    </div>
                    <div>Tests: {total}</div>
                    <div>Passed: {passed}</div>
                    <div>Failed: {failed}</div>
                    <div>Errors: {errors}</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {progress}%"></div>
                    </div>
                </div>
            """
        
        return cards_html
    
    def _generate_quality_metrics(self):
        """Generate quality metrics HTML"""
        metrics_html = '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">'
        
        for tool, result in self.test_results.get("quality_metrics", {}).items():
            status = result.get("status", "unknown")
            status_color = "#28a745" if status == "passed" else "#dc3545" if status == "failed" else "#ffc107"
            
            metrics_html += f"""
                <div style="background: white; padding: 15px; border-radius: 5px; border: 1px solid #e9ecef;">
                    <div style="font-weight: bold; margin-bottom: 5px;">{tool.title()}</div>
                    <div style="color: {status_color}; font-size: 1.2em;">{status.title()}</div>
                </div>
            """
        
        metrics_html += '</div>'
        return metrics_html

def main():
    """Main function"""
    runner = TDDRunner()
    success = runner.run_all_tests()
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎯 TDD TEST SUITE SUMMARY")
    print("=" * 60)
    
    results = runner.test_results
    print(f"Total Tests: {results['total_tests']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Errors: {results['errors']}")
    print(f"Code Coverage: {results['coverage']:.1f}%")
    print(f"Execution Time: {results['performance_metrics']['execution_time_seconds']:.2f} seconds")
    
    if results['total_tests'] > 0:
        success_rate = (results['passed'] / results['total_tests']) * 100
        print(f"Success Rate: {success_rate:.1f}%")
    
    print("=" * 60)
    
    print("\n📊 Reports Generated:")
    print("  - tests/reports/tdd_test_report.json (Comprehensive test report)")
    print("  - tests/reports/tdd_dashboard.html (Interactive dashboard)")
    print("  - tests/reports/coverage_html/ (HTML coverage report)")
    print("  - tests/reports/security_report.json (Security analysis)")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 