#!/usr/bin/env python3
"""
Production Deployment Orchestrator.

This script orchestrates the complete production deployment process including:
- Infrastructure provisioning
- Service deployment
- Health validation  
- Monitoring setup
- Rollback capabilities
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'deployment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DeploymentOrchestrator:
    """Orchestrates production deployments with comprehensive validation."""
    
    def __init__(self, config_path: str = "deployment/config/production.yaml"):
        """Initialize deployment orchestrator."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.deployment_id = f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.state_file = Path(f"deployment_state_{self.deployment_id}.json")
        self.rollback_info = {}
        
        logger.info(f"Deployment orchestrator initialized: {self.deployment_id}")

    def _load_config(self) -> Dict[str, Any]:
        """Load deployment configuration."""
        try:
            with open(self.config_path) as f:
                config = yaml.safe_load(f)
            
            # Validate required configuration
            required_keys = ['kubernetes', 'monitoring', 'database', 'security']
            for key in required_keys:
                if key not in config:
                    raise ValueError(f"Missing required configuration section: {key}")
            
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            sys.exit(1)

    def _save_state(self, state: Dict[str, Any]):
        """Save deployment state for rollback purposes."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            logger.info(f"Deployment state saved: {self.state_file}")
        except Exception as e:
            logger.error(f"Failed to save deployment state: {e}")

    def _run_command(self, command: List[str], check: bool = True, timeout: int = 300) -> Tuple[bool, str]:
        """Run a shell command with logging and error handling."""
        logger.info(f"Executing: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=check
            )
            
            if result.stdout:
                logger.info(f"STDOUT: {result.stdout}")
            if result.stderr:
                logger.warning(f"STDERR: {result.stderr}")
            
            return True, result.stdout
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed with exit code {e.returncode}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            return False, e.stderr
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out after {timeout} seconds")
            return False, "Command timeout"
        except Exception as e:
            logger.error(f"Unexpected error running command: {e}")
            return False, str(e)

    def validate_prerequisites(self) -> bool:
        """Validate deployment prerequisites."""
        logger.info("🔍 Validating deployment prerequisites...")
        
        checks = [
            ("kubectl", ["kubectl", "version", "--client"]),
            ("helm", ["helm", "version"]),
            ("docker", ["docker", "--version"]),
            ("aws-cli", ["aws", "--version"]),
        ]
        
        all_passed = True
        
        for tool, command in checks:
            success, output = self._run_command(command, check=False)
            if success:
                logger.info(f"✅ {tool} available")
            else:
                logger.error(f"❌ {tool} not available or not working")
                all_passed = False
        
        # Check Kubernetes connectivity
        success, output = self._run_command(
            ["kubectl", "cluster-info"], 
            check=False
        )
        if success:
            logger.info("✅ Kubernetes cluster connectivity verified")
        else:
            logger.error("❌ Cannot connect to Kubernetes cluster")
            all_passed = False
        
        # Validate image availability
        image_tag = os.getenv('IMAGE_TAG', 'latest')
        registry = self.config.get('registry', 'ghcr.io')
        image_name = self.config.get('image_name', 'fair-credit-scorer')
        full_image = f"{registry}/{image_name}:{image_tag}"
        
        success, output = self._run_command(
            ["docker", "manifest", "inspect", full_image],
            check=False
        )
        if success:
            logger.info(f"✅ Container image available: {full_image}")
        else:
            logger.error(f"❌ Container image not available: {full_image}")
            all_passed = False
        
        return all_passed

    def setup_infrastructure(self) -> bool:
        """Setup and validate infrastructure components."""
        logger.info("🏗️ Setting up infrastructure components...")
        
        # Apply Kubernetes manifests in order
        manifest_order = [
            "deployment/k8s/namespace.yaml",
            "deployment/k8s/secrets.yaml", 
            "deployment/k8s/configmap.yaml",
            "deployment/k8s/database.yaml",
        ]
        
        for manifest in manifest_order:
            if not Path(manifest).exists():
                logger.error(f"Manifest file not found: {manifest}")
                return False
            
            success, output = self._run_command([
                "kubectl", "apply", "-f", manifest
            ])
            
            if not success:
                logger.error(f"Failed to apply manifest: {manifest}")
                return False
            
            logger.info(f"✅ Applied manifest: {manifest}")
        
        # Wait for database to be ready
        logger.info("⏳ Waiting for database to be ready...")
        success, output = self._run_command([
            "kubectl", "wait", "--for=condition=ready", "pod",
            "-l", "app.kubernetes.io/name=postgres",
            "-n", "fair-credit-system",
            "--timeout=600s"
        ])
        
        if not success:
            logger.error("Database failed to become ready")
            return False
        
        logger.info("✅ Database is ready")
        return True

    def deploy_application(self) -> bool:
        """Deploy the main application."""
        logger.info("🚀 Deploying main application...")
        
        # Update image tag in deployment manifest
        image_tag = os.getenv('IMAGE_TAG', 'latest')
        registry = self.config.get('registry', 'ghcr.io')
        image_name = self.config.get('image_name', 'fair-credit-scorer')
        full_image = f"{registry}/{image_name}:{image_tag}"
        
        # Read deployment manifest
        deployment_manifest = "deployment/k8s/main-application.yaml"
        with open(deployment_manifest, 'r') as f:
            content = f.read()
        
        # Replace image placeholder
        content = content.replace(
            "image: fair-credit-scorer:1.0.0",
            f"image: {full_image}"
        )
        
        # Write temporary manifest
        temp_manifest = Path("deployment_temp.yaml")
        with open(temp_manifest, 'w') as f:
            f.write(content)
        
        try:
            # Apply the deployment
            success, output = self._run_command([
                "kubectl", "apply", "-f", str(temp_manifest)
            ])
            
            if not success:
                logger.error("Failed to apply application deployment")
                return False
            
            # Wait for rollout to complete
            logger.info("⏳ Waiting for deployment rollout...")
            success, output = self._run_command([
                "kubectl", "rollout", "status", 
                "deployment/fair-credit-scorer",
                "-n", "fair-credit-system",
                "--timeout=600s"
            ])
            
            if not success:
                logger.error("Application deployment rollout failed")
                return False
            
            logger.info("✅ Application deployed successfully")
            
            # Store rollback information
            self.rollback_info['previous_image'] = self._get_current_image()
            self.rollback_info['deployment_time'] = datetime.now(timezone.utc).isoformat()
            
            return True
            
        finally:
            # Clean up temporary manifest
            if temp_manifest.exists():
                temp_manifest.unlink()

    def _get_current_image(self) -> Optional[str]:
        """Get currently deployed image."""
        success, output = self._run_command([
            "kubectl", "get", "deployment", "fair-credit-scorer",
            "-n", "fair-credit-system",
            "-o", "jsonpath={.spec.template.spec.containers[0].image}"
        ], check=False)
        
        return output.strip() if success else None

    def validate_deployment(self) -> bool:
        """Validate deployment health and functionality."""
        logger.info("🔍 Validating deployment...")
        
        # Check pod readiness
        success, output = self._run_command([
            "kubectl", "get", "pods",
            "-l", "app.kubernetes.io/name=fair-credit-scorer",
            "-n", "fair-credit-system",
            "-o", "jsonpath={.items[*].status.phase}"
        ])
        
        if not success:
            logger.error("Failed to check pod status")
            return False
        
        phases = output.strip().split()
        if not all(phase == "Running" for phase in phases):
            logger.error(f"Not all pods are running: {phases}")
            return False
        
        logger.info("✅ All pods are running")
        
        # Health check endpoint validation
        return self._validate_health_endpoints()

    def _validate_health_endpoints(self) -> bool:
        """Validate application health endpoints."""
        logger.info("🏥 Validating health endpoints...")
        
        # Port forward to access the service locally
        port_forward_process = None
        try:
            port_forward_process = subprocess.Popen([
                "kubectl", "port-forward",
                "service/fair-credit-service",
                "8080:8000",
                "-n", "fair-credit-system"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Wait for port forward to be established
            time.sleep(5)
            
            # Test health endpoints
            health_endpoints = [
                ("liveness", "http://localhost:8080/health/live"),
                ("readiness", "http://localhost:8080/health/ready"),
                ("startup", "http://localhost:8080/health/startup"),
                ("metrics", "http://localhost:8080/metrics")
            ]
            
            for endpoint_name, url in health_endpoints:
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        logger.info(f"✅ {endpoint_name} endpoint healthy")
                    else:
                        logger.error(f"❌ {endpoint_name} endpoint returned {response.status_code}")
                        return False
                except requests.RequestException as e:
                    logger.error(f"❌ {endpoint_name} endpoint failed: {e}")
                    return False
            
            return True
            
        finally:
            if port_forward_process:
                port_forward_process.terminate()
                port_forward_process.wait()

    def setup_monitoring(self) -> bool:
        """Setup monitoring and alerting."""
        logger.info("📊 Setting up monitoring and alerting...")
        
        # Apply monitoring manifests
        monitoring_manifests = [
            "deployment/k8s/monitoring/prometheus.yaml",
            "deployment/k8s/monitoring/grafana.yaml",
            "deployment/k8s/monitoring/alertmanager.yaml"
        ]
        
        for manifest in monitoring_manifests:
            if Path(manifest).exists():
                success, output = self._run_command([
                    "kubectl", "apply", "-f", manifest
                ])
                if success:
                    logger.info(f"✅ Applied monitoring manifest: {manifest}")
                else:
                    logger.warning(f"⚠️ Failed to apply monitoring manifest: {manifest}")
        
        # Validate monitoring endpoints
        return self._validate_monitoring_endpoints()

    def _validate_monitoring_endpoints(self) -> bool:
        """Validate monitoring endpoints."""
        logger.info("📈 Validating monitoring endpoints...")
        
        # Check if Prometheus is scraping metrics
        try:
            # This would typically check if Prometheus is collecting metrics
            # For now, we'll just check if the monitoring pods are running
            success, output = self._run_command([
                "kubectl", "get", "pods",
                "-l", "app=prometheus",
                "-n", "fair-credit-system",
                "-o", "jsonpath={.items[*].status.phase}"
            ], check=False)
            
            if success and "Running" in output:
                logger.info("✅ Monitoring components are running")
                return True
            else:
                logger.warning("⚠️ Monitoring components may not be fully ready")
                return True  # Don't fail deployment for monitoring issues
                
        except Exception as e:
            logger.warning(f"⚠️ Could not validate monitoring: {e}")
            return True  # Don't fail deployment for monitoring issues

    def run_smoke_tests(self) -> bool:
        """Run smoke tests to validate basic functionality."""
        logger.info("🧪 Running smoke tests...")
        
        # Run basic API tests
        test_commands = [
            ["python", "-m", "pytest", "tests/smoke/", "-v", "--tb=short"],
            ["python", "scripts/api_validation.py", "--environment=production"],
            ["python", "scripts/fairness_validation.py", "--quick-check"]
        ]
        
        for command in test_commands:
            success, output = self._run_command(command, check=False, timeout=300)
            if not success:
                logger.error(f"Smoke test failed: {' '.join(command)}")
                return False
        
        logger.info("✅ All smoke tests passed")
        return True

    def rollback(self) -> bool:
        """Rollback to previous deployment."""
        logger.error("🔄 Initiating deployment rollback...")
        
        try:
            # Use kubectl rollout undo
            success, output = self._run_command([
                "kubectl", "rollout", "undo",
                "deployment/fair-credit-scorer",
                "-n", "fair-credit-system"
            ])
            
            if not success:
                logger.error("Failed to initiate rollback")
                return False
            
            # Wait for rollback to complete
            success, output = self._run_command([
                "kubectl", "rollout", "status",
                "deployment/fair-credit-scorer", 
                "-n", "fair-credit-system",
                "--timeout=300s"
            ])
            
            if not success:
                logger.error("Rollback failed to complete")
                return False
            
            logger.info("✅ Rollback completed successfully")
            
            # Validate rollback
            if self._validate_health_endpoints():
                logger.info("✅ Rollback validation passed")
                return True
            else:
                logger.error("❌ Rollback validation failed")
                return False
                
        except Exception as e:
            logger.error(f"Rollback failed with exception: {e}")
            return False

    def cleanup(self):
        """Cleanup deployment artifacts."""
        logger.info("🧹 Cleaning up deployment artifacts...")
        
        # Remove state file if deployment succeeded
        if self.state_file.exists():
            self.state_file.unlink()
            logger.info("State file cleaned up")

    def deploy(self) -> bool:
        """Execute complete deployment process."""
        logger.info(f"🚀 Starting production deployment: {self.deployment_id}")
        
        deployment_state = {
            'deployment_id': self.deployment_id,
            'start_time': datetime.now(timezone.utc).isoformat(),
            'status': 'started'
        }
        
        try:
            # Step 1: Validate prerequisites
            if not self.validate_prerequisites():
                deployment_state['status'] = 'failed_prerequisites'
                self._save_state(deployment_state)
                return False
            
            deployment_state['prerequisites_passed'] = True
            
            # Step 2: Setup infrastructure
            if not self.setup_infrastructure():
                deployment_state['status'] = 'failed_infrastructure'
                self._save_state(deployment_state)
                return False
            
            deployment_state['infrastructure_ready'] = True
            
            # Step 3: Deploy application
            if not self.deploy_application():
                deployment_state['status'] = 'failed_application'
                self._save_state(deployment_state)
                logger.error("Application deployment failed - attempting rollback")
                self.rollback()
                return False
            
            deployment_state['application_deployed'] = True
            
            # Step 4: Validate deployment
            if not self.validate_deployment():
                deployment_state['status'] = 'failed_validation'
                self._save_state(deployment_state)
                logger.error("Deployment validation failed - attempting rollback")
                self.rollback()
                return False
            
            deployment_state['deployment_validated'] = True
            
            # Step 5: Setup monitoring
            if not self.setup_monitoring():
                logger.warning("Monitoring setup failed - continuing with deployment")
                deployment_state['monitoring_warning'] = True
            else:
                deployment_state['monitoring_ready'] = True
            
            # Step 6: Run smoke tests
            if not self.run_smoke_tests():
                deployment_state['status'] = 'failed_smoke_tests'
                self._save_state(deployment_state)
                logger.error("Smoke tests failed - attempting rollback")
                self.rollback()
                return False
            
            deployment_state['smoke_tests_passed'] = True
            deployment_state['status'] = 'completed'
            deployment_state['end_time'] = datetime.now(timezone.utc).isoformat()
            
            self._save_state(deployment_state)
            
            logger.info("🎉 Production deployment completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Deployment failed with exception: {e}")
            deployment_state['status'] = 'failed_exception'
            deployment_state['error'] = str(e)
            self._save_state(deployment_state)
            
            # Attempt rollback on critical failure
            logger.error("Critical failure - attempting rollback")
            self.rollback()
            return False

def main():
    """Main entry point for deployment orchestrator."""
    parser = argparse.ArgumentParser(
        description="Production Deployment Orchestrator for Fair Credit Scorer"
    )
    
    parser.add_argument(
        "--config",
        default="deployment/config/production.yaml",
        help="Path to deployment configuration file"
    )
    
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Validate prerequisites without deploying"
    )
    
    parser.add_argument(
        "--rollback-only",
        action="store_true", 
        help="Only perform rollback operation"
    )
    
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip smoke tests (use with caution)"
    )
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = DeploymentOrchestrator(args.config)
    
    try:
        if args.rollback_only:
            success = orchestrator.rollback()
            sys.exit(0 if success else 1)
        
        if args.dry_run:
            success = orchestrator.validate_prerequisites()
            logger.info(f"Dry run {'passed' if success else 'failed'}")
            sys.exit(0 if success else 1)
        
        # Full deployment
        success = orchestrator.deploy()
        
        if success:
            logger.info("🎉 Deployment completed successfully!")
            orchestrator.cleanup()
        else:
            logger.error("💥 Deployment failed!")
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.warning("Deployment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()