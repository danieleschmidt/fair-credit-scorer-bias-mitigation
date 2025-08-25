#!/usr/bin/env python3
"""
Production readiness validation for the autonomous SDLC implementation.
Validates production deployment configuration and infrastructure.
"""

import sys
import os
import subprocess
import yaml
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class ProductionReadinessValidator:
    """Validates production deployment readiness."""
    
    def __init__(self):
        self.results = {
            "docker": {"tests": [], "passed": 0, "total": 0},
            "kubernetes": {"tests": [], "passed": 0, "total": 0},
            "monitoring": {"tests": [], "passed": 0, "total": 0},
            "security": {"tests": [], "passed": 0, "total": 0},
            "overall": {"passed": 0, "total": 0}
        }
    
    def test_docker_configuration(self):
        """Test Docker configuration and build readiness."""
        print("\n🐳 DOCKER CONFIGURATION VALIDATION")
        category = "docker"
        
        # Test 1: Dockerfile exists and is valid
        dockerfile_path = Path("deployment/Dockerfile")
        if dockerfile_path.exists():
            self._record_test(category, "Dockerfile Exists", True)
            print("✅ Dockerfile found at deployment/Dockerfile")
            
            # Check key Dockerfile practices
            content = dockerfile_path.read_text()
            
            # Check for security best practices
            if "USER" in content and "RUN useradd" in content:
                self._record_test(category, "Non-root User Security", True)
                print("✅ Non-root user configuration found")
            else:
                self._record_test(category, "Non-root User Security", False)
            
            # Check for health checks
            if "HEALTHCHECK" in content:
                self._record_test(category, "Health Check Configuration", True)
                print("✅ Health check configured")
            else:
                self._record_test(category, "Health Check Configuration", False)
                
        else:
            self._record_test(category, "Dockerfile Exists", False)
        
        # Test 2: Docker compose for production exists
        compose_path = Path("deployment/advanced/docker-compose.production.yml")
        if compose_path.exists():
            self._record_test(category, "Production Docker Compose", True)
            print("✅ Production Docker Compose configuration found")
        else:
            self._record_test(category, "Production Docker Compose", False)
        
        # Test 3: Test Docker build capability
        try:
            # Check if Docker is available
            result = subprocess.run(
                ["docker", "--version"], 
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0:
                self._record_test(category, "Docker Runtime Available", True)
                print(f"✅ Docker runtime: {result.stdout.strip()}")
            else:
                self._record_test(category, "Docker Runtime Available", False)
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self._record_test(category, "Docker Runtime Available", False)
    
    def test_kubernetes_configuration(self):
        """Test Kubernetes deployment configuration."""
        print("\n☸️ KUBERNETES CONFIGURATION VALIDATION")
        category = "kubernetes"
        
        k8s_files = [
            "deployment/k8s/deployment.yaml",
            "deployment/k8s/service.yaml",
            "deployment/k8s/configmap.yaml",
            "deployment/k8s/secrets.yaml",
            "deployment/k8s/ingress.yaml"
        ]
        
        valid_manifests = 0
        
        for k8s_file in k8s_files:
            file_path = Path(k8s_file)
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        content = yaml.safe_load_all(f)
                        manifests = list(content)
                        
                    if manifests and any(manifests):
                        valid_manifests += 1
                        print(f"✅ Valid K8s manifest: {file_path.name}")
                    else:
                        print(f"❌ Invalid K8s manifest: {file_path.name}")
                        
                except yaml.YAMLError as e:
                    print(f"❌ YAML error in {file_path.name}: {e}")
                    
            else:
                print(f"❌ Missing K8s manifest: {file_path.name}")
        
        if valid_manifests >= 4:  # At least 4 valid manifests
            self._record_test(category, "Kubernetes Manifests", True)
        else:
            self._record_test(category, "Kubernetes Manifests", False)
        
        # Test deployment-specific configuration
        deployment_path = Path("deployment/k8s/deployment.yaml")
        if deployment_path.exists():
            try:
                with open(deployment_path, 'r') as f:
                    deployment = yaml.safe_load(f)
                
                # Check security configurations
                spec = deployment.get('spec', {}).get('template', {}).get('spec', {})
                
                if spec.get('securityContext', {}).get('runAsNonRoot'):
                    self._record_test(category, "K8s Security Context", True)
                    print("✅ Kubernetes security context configured")
                else:
                    self._record_test(category, "K8s Security Context", False)
                
                # Check resource limits
                containers = spec.get('containers', [])
                if containers and containers[0].get('resources', {}).get('limits'):
                    self._record_test(category, "K8s Resource Limits", True)
                    print("✅ Kubernetes resource limits configured")
                else:
                    self._record_test(category, "K8s Resource Limits", False)
                    
            except Exception as e:
                print(f"❌ Error parsing deployment.yaml: {e}")
                self._record_test(category, "K8s Security Context", False)
                self._record_test(category, "K8s Resource Limits", False)
        
        # Test HPA configuration
        hpa_path = Path("deployment/k8s/hpa.yaml")
        if hpa_path.exists():
            self._record_test(category, "Horizontal Pod Autoscaler", True)
            print("✅ HPA configuration found")
        else:
            self._record_test(category, "Horizontal Pod Autoscaler", False)
    
    def test_monitoring_configuration(self):
        """Test monitoring and observability configuration."""
        print("\n📊 MONITORING CONFIGURATION VALIDATION")
        category = "monitoring"
        
        # Test Prometheus configuration
        prometheus_path = Path("deployment/monitoring/prometheus.yml")
        if prometheus_path.exists():
            self._record_test(category, "Prometheus Configuration", True)
            print("✅ Prometheus configuration found")
        else:
            self._record_test(category, "Prometheus Configuration", False)
        
        # Test alert rules
        alerts_path = Path("deployment/monitoring/alert-rules.yml")
        if alerts_path.exists():
            self._record_test(category, "Alert Rules Configuration", True)
            print("✅ Alert rules configuration found")
        else:
            self._record_test(category, "Alert Rules Configuration", False)
        
        # Test application health check endpoint
        try:
            from health_check import get_health_status
            
            health = get_health_status()
            if health and 'overall_health_score' in health:
                self._record_test(category, "Application Health Endpoint", True)
                print(f"✅ Health endpoint working: {health['overall_health_score']:.2f}")
            else:
                self._record_test(category, "Application Health Endpoint", False)
                
        except Exception as e:
            self._record_test(category, "Application Health Endpoint", False, str(e))
        
        # Test metrics collection
        try:
            from metrics_server import MetricsServer
            
            # Test that metrics server can be initialized
            server = MetricsServer()
            self._record_test(category, "Metrics Server Configuration", True)
            print("✅ Metrics server configuration valid")
            
        except Exception as e:
            self._record_test(category, "Metrics Server Configuration", False, str(e))
    
    def test_security_configuration(self):
        """Test security configuration and compliance."""
        print("\n🔒 SECURITY CONFIGURATION VALIDATION")
        category = "security"
        
        # Test security scanning results
        security_report = Path("security_report.json")
        if security_report.exists():
            try:
                with open(security_report, 'r') as f:
                    report = json.load(f)
                
                # Check for high severity issues
                metrics = report.get('metrics', {})
                high_severity = metrics.get('_totals', {}).get('SEVERITY.HIGH', 0)
                
                if high_severity == 0:
                    self._record_test(category, "Security Scan Results", True)
                    print("✅ No high severity security issues found")
                else:
                    self._record_test(category, "Security Scan Results", False, f"{high_severity} high severity issues")
                    
            except Exception as e:
                self._record_test(category, "Security Scan Results", False, str(e))
        else:
            self._record_test(category, "Security Scan Results", False, "Security report not found")
        
        # Test secrets management
        secrets_path = Path("deployment/k8s/secrets.yaml")
        if secrets_path.exists():
            self._record_test(category, "Secrets Management", True)
            print("✅ Kubernetes secrets configuration found")
        else:
            self._record_test(category, "Secrets Management", False)
        
        # Test TLS configuration
        ingress_path = Path("deployment/k8s/ingress.yaml")
        if ingress_path.exists():
            try:
                with open(ingress_path, 'r') as f:
                    ingress = yaml.safe_load(f)
                
                spec = ingress.get('spec', {})
                if spec.get('tls'):
                    self._record_test(category, "TLS Configuration", True)
                    print("✅ TLS configuration found in ingress")
                else:
                    self._record_test(category, "TLS Configuration", False)
                    
            except Exception as e:
                self._record_test(category, "TLS Configuration", False, str(e))
        else:
            self._record_test(category, "TLS Configuration", False)
        
        # Test production configuration security
        prod_config = Path("deployment/config/production.yaml")
        if prod_config.exists():
            try:
                with open(prod_config, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Check that debug mode is disabled
                if not config.get('debug', True):  # Default True, so False means explicitly set
                    self._record_test(category, "Production Security Config", True)
                    print("✅ Production security configuration validated")
                else:
                    self._record_test(category, "Production Security Config", False)
                    
            except Exception as e:
                self._record_test(category, "Production Security Config", False, str(e))
        else:
            self._record_test(category, "Production Security Config", False)
    
    def _record_test(self, category, test_name, passed, error=None):
        """Record test result."""
        self.results[category]["tests"].append({
            "name": test_name,
            "passed": passed,
            "error": error
        })
        
        if passed:
            self.results[category]["passed"] += 1
        
        self.results[category]["total"] += 1
        self.results["overall"]["total"] += 1
        
        if passed:
            self.results["overall"]["passed"] += 1
    
    def print_summary(self):
        """Print production readiness summary."""
        print("\n" + "=" * 80)
        print("🚀 PRODUCTION READINESS VALIDATION SUMMARY")
        print("=" * 80)
        
        for category, data in self.results.items():
            if category == "overall":
                continue
                
            print(f"\n📋 {category.upper()}:")
            print(f"   Tests Passed: {data['passed']}/{data['total']}")
            
            if data['total'] > 0:
                percentage = (data['passed'] / data['total']) * 100
                print(f"   Success Rate: {percentage:.1f}%")
            
            for test in data['tests']:
                status = "✅" if test['passed'] else "❌"
                print(f"   {status} {test['name']}")
                if not test['passed'] and test['error']:
                    print(f"      Error: {test['error'][:100]}...")
        
        total_passed = self.results["overall"]["passed"]
        total_tests = self.results["overall"]["total"]
        
        if total_tests > 0:
            overall_percentage = (total_passed / total_tests) * 100
            print(f"\n🎯 OVERALL PRODUCTION READINESS: {overall_percentage:.1f}%")
            print(f"   Total Tests: {total_passed}/{total_tests}")
            
            if overall_percentage >= 90:
                print("\n🎉 EXCELLENT: Production deployment ready!")
                print("✅ AUTONOMOUS SDLC: PRODUCTION-READY")
                return True
            elif overall_percentage >= 80:
                print("\n🟡 GOOD: Production deployment mostly ready")
                print("✅ AUTONOMOUS SDLC: NEARLY PRODUCTION-READY")
                return True
            elif overall_percentage >= 70:
                print("\n⚠️ FAIR: Production deployment needs improvement")
                print("⚠️ AUTONOMOUS SDLC: PARTIAL PRODUCTION READINESS")
                return False
            else:
                print("\n❌ POOR: Production deployment not ready")
                print("❌ AUTONOMOUS SDLC: NOT PRODUCTION-READY")
                return False
        
        return False
    
    def run_production_validation(self):
        """Run all production readiness tests."""
        print("🧪 STARTING PRODUCTION READINESS VALIDATION")
        print("🎯 TARGET: 80%+ PRODUCTION READINESS")
        
        self.test_docker_configuration()
        self.test_kubernetes_configuration()
        self.test_monitoring_configuration() 
        self.test_security_configuration()
        
        return self.print_summary()

if __name__ == "__main__":
    validator = ProductionReadinessValidator()
    success = validator.run_production_validation()
    
    # Save detailed results
    results_file = Path("production_readiness_report.json")
    with open(results_file, 'w') as f:
        json.dump(validator.results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed results saved to: {results_file}")
    
    exit(0 if success else 1)