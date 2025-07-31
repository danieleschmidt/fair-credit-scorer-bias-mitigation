#!/usr/bin/env python3
"""
Container security scanning automation for Docker images.
Part of advanced SDLC enhancement suite.
"""

import json
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime
import yaml

@dataclass
class SecurityVulnerability:
    """Container security vulnerability data."""
    vulnerability_id: str
    severity: str
    package_name: str
    installed_version: str
    fixed_version: Optional[str]
    description: str
    cvss_score: Optional[float]
    references: List[str]

@dataclass
class SecurityScanReport:
    """Container security scan results."""
    image_name: str
    scan_timestamp: str
    total_vulnerabilities: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    vulnerabilities: List[SecurityVulnerability]
    compliance_score: float
    recommendations: List[str]

class ContainerSecurityScanner:
    """Automated container security scanning with multiple tools."""
    
    def __init__(self, output_dir: str = "security-reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.config_file = Path("container_security_config.yml")
        self.ensure_config()
        
    def ensure_config(self):
        """Ensure container security configuration exists."""
        if not self.config_file.exists():
            default_config = {
                'scanners': {
                    'trivy': {
                        'enabled': True,
                        'severity_threshold': 'MEDIUM',
                        'scan_types': ['os', 'library'],
                        'ignore_unfixed': False
                    },
                    'grype': {
                        'enabled': False,  # Optional secondary scanner
                        'output_format': 'json'
                    },
                    'docker_bench': {
                        'enabled': True,
                        'categories': ['container_images', 'container_runtime']
                    }
                },
                'thresholds': {
                    'fail_on_critical': True,
                    'fail_on_high': False,
                    'max_critical': 0,
                    'max_high': 5,
                    'max_medium': 20,
                    'compliance_threshold': 80.0
                },
                'notification': {
                    'slack_webhook': None,
                    'email_recipients': [],
                    'notify_on_new_critical': True
                },
                'dockerfile_rules': {
                    'require_non_root_user': True,
                    'require_health_check': True,
                    'prohibit_latest_tag': True,
                    'require_specific_versions': True,
                    'scan_secrets': True
                }
            }
            
            with open(self.config_file, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
                
    def load_config(self) -> Dict[str, Any]:
        """Load container security configuration."""
        with open(self.config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def scan_with_trivy(self, image_name: str) -> SecurityScanReport:
        """Scan container image using Trivy."""
        config = self.load_config()
        trivy_config = config['scanners']['trivy']
        
        if not trivy_config.get('enabled', True):
            print("Trivy scanning is disabled")
            return None
            
        try:
            # Run Trivy scan
            cmd = [
                'trivy', 'image',
                '--format', 'json',
                '--severity', 'CRITICAL,HIGH,MEDIUM,LOW',
                image_name
            ]
            
            if trivy_config.get('ignore_unfixed', False):
                cmd.append('--ignore-unfixed')
                
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                print(f"Trivy scan failed: {result.stderr}")
                return None
                
            scan_data = json.loads(result.stdout)
            return self._parse_trivy_results(image_name, scan_data)
            
        except subprocess.TimeoutExpired:
            print("Trivy scan timed out")
            return None
        except Exception as e:
            print(f"Error running Trivy scan: {e}")
            return None
    
    def _parse_trivy_results(self, image_name: str, scan_data: Dict) -> SecurityScanReport:
        """Parse Trivy scan results into structured format."""
        vulnerabilities = []
        severity_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        
        for result in scan_data.get('Results', []):
            for vuln in result.get('Vulnerabilities', []):
                severity = vuln.get('Severity', 'UNKNOWN')
                if severity in severity_counts:
                    severity_counts[severity] += 1
                    
                vulnerability = SecurityVulnerability(
                    vulnerability_id=vuln.get('VulnerabilityID', ''),
                    severity=severity,
                    package_name=vuln.get('PkgName', ''),
                    installed_version=vuln.get('InstalledVersion', ''),
                    fixed_version=vuln.get('FixedVersion'),
                    description=vuln.get('Description', ''),
                    cvss_score=self._extract_cvss_score(vuln),
                    references=vuln.get('References', [])
                )
                vulnerabilities.append(vulnerability)
        
        total_vulns = sum(severity_counts.values())
        compliance_score = self._calculate_compliance_score(severity_counts, total_vulns)
        recommendations = self._generate_recommendations(severity_counts, vulnerabilities)
        
        return SecurityScanReport(
            image_name=image_name,
            scan_timestamp=datetime.now().isoformat(),
            total_vulnerabilities=total_vulns,
            critical_count=severity_counts['CRITICAL'],
            high_count=severity_counts['HIGH'],
            medium_count=severity_counts['MEDIUM'],
            low_count=severity_counts['LOW'],
            vulnerabilities=vulnerabilities,
            compliance_score=compliance_score,
            recommendations=recommendations
        )
    
    def scan_dockerfile(self, dockerfile_path: str = "Dockerfile") -> Dict[str, Any]:
        """Scan Dockerfile for security best practices."""
        config = self.load_config()
        dockerfile_rules = config.get('dockerfile_rules', {})
        
        dockerfile = Path(dockerfile_path)
        if not dockerfile.exists():
            return {'error': f"Dockerfile not found: {dockerfile_path}"}
            
        issues = []
        recommendations = []
        
        try:
            with open(dockerfile, 'r') as f:
                content = f.read()
                lines = content.split('\n')
                
            # Check for non-root user
            if dockerfile_rules.get('require_non_root_user', True):
                if 'USER ' not in content or content.count('USER root') > 0:
                    issues.append({
                        'rule': 'non_root_user',
                        'severity': 'HIGH',
                        'message': 'Container should not run as root user',
                        'recommendation': 'Add USER directive with non-root user'
                    })
            
            # Check for health check
            if dockerfile_rules.get('require_health_check', True):
                if 'HEALTHCHECK' not in content:
                    issues.append({
                        'rule': 'health_check',
                        'severity': 'MEDIUM',
                        'message': 'Container lacks health check',
                        'recommendation': 'Add HEALTHCHECK instruction'
                    })
            
            # Check for latest tag usage
            if dockerfile_rules.get('prohibit_latest_tag', True):
                for line in lines:
                    if line.strip().startswith('FROM') and ':latest' in line:
                        issues.append({
                            'rule': 'latest_tag',
                            'severity': 'MEDIUM',
                            'message': f'Using :latest tag: {line.strip()}',
                            'recommendation': 'Use specific version tags instead of :latest'
                        })
            
            # Check for secrets
            if dockerfile_rules.get('scan_secrets', True):
                secret_patterns = [
                    r'password', r'secret', r'key', r'token', r'pwd',
                    r'api[_-]?key', r'auth[_-]?token'
                ]
                
                for i, line in enumerate(lines, 1):
                    line_lower = line.lower()
                    for pattern in secret_patterns:
                        if pattern in line_lower and '=' in line:
                            issues.append({
                                'rule': 'potential_secret',
                                'severity': 'CRITICAL',
                                'message': f'Potential secret in line {i}: {line.strip()}',
                                'recommendation': 'Use build args or environment variables for secrets'
                            })
                            
            return {
                'dockerfile': dockerfile_path,
                'scan_timestamp': datetime.now().isoformat(),
                'total_issues': len(issues),
                'issues': issues,
                'compliance_passed': len([i for i in issues if i['severity'] == 'CRITICAL']) == 0
            }
            
        except Exception as e:
            return {'error': f"Error scanning Dockerfile: {e}"}
    
    def run_docker_bench_security(self) -> Dict[str, Any]:
        """Run Docker Bench Security checks."""
        try:
            # Check if Docker Bench Security is available
            result = subprocess.run(
                ['docker', 'run', '--rm', '--net', 'host', '--pid', 'host', '--userns', 'host',
                 '--cap-add', 'audit_control', '-e', 'DOCKER_CONTENT_TRUST=$DOCKER_CONTENT_TRUST',
                 '-v', '/var/lib:/var/lib:ro', '-v', '/var/run/docker.sock:/var/run/docker.sock:ro',
                 '-v', '/usr/lib/systemd:/usr/lib/systemd:ro', '-v', '/etc:/etc:ro',
                 'docker/docker-bench-security'],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            return {
                'scan_timestamp': datetime.now().isoformat(),
                'exit_code': result.returncode,
                'output': result.stdout,
                'errors': result.stderr,
                'passed': result.returncode == 0
            }
            
        except subprocess.TimeoutExpired:
            return {'error': 'Docker Bench Security scan timed out'}
        except Exception as e:
            return {'error': f'Error running Docker Bench Security: {e}'}
    
    def generate_security_report(self, image_name: str, scan_results: SecurityScanReport, 
                               dockerfile_results: Dict, bench_results: Dict) -> str:
        """Generate comprehensive security report."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_data = {
            'meta': {
                'image_name': image_name,
                'scan_timestamp': timestamp,
                'report_version': '1.0'
            },
            'vulnerability_scan': asdict(scan_results) if scan_results else None,
            'dockerfile_analysis': dockerfile_results,
            'docker_bench_security': bench_results,
            'overall_assessment': self._generate_overall_assessment(
                scan_results, dockerfile_results, bench_results
            )
        }
        
        # Save JSON report
        json_file = self.output_dir / f"security_report_{image_name.replace('/', '_')}_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Generate HTML report
        html_file = self.output_dir / f"security_report_{image_name.replace('/', '_')}_{timestamp}.html"
        self._generate_html_security_report(report_data, html_file)
        
        # Check if scan should fail build
        should_fail = self._should_fail_build(scan_results, dockerfile_results)
        
        print(f"🔒 Security report generated: {json_file}")
        if scan_results:
            print(f"🚨 Critical vulnerabilities: {scan_results.critical_count}")
            print(f"⚠️  High vulnerabilities: {scan_results.high_count}")
            print(f"📊 Compliance score: {scan_results.compliance_score:.1f}%")
            
        if should_fail:
            print("❌ Security scan FAILED - build should be blocked")
            return "FAILED"
        else:
            print("✅ Security scan PASSED")
            return "PASSED"
    
    def _extract_cvss_score(self, vuln: Dict) -> Optional[float]:
        """Extract CVSS score from vulnerability data."""
        cvss = vuln.get('CVSS', {})
        if isinstance(cvss, dict):
            for version in ['v3', 'v2']:
                if version in cvss and 'Score' in cvss[version]:
                    return float(cvss[version]['Score'])
        return None
    
    def _calculate_compliance_score(self, severity_counts: Dict, total_vulns: int) -> float:
        """Calculate compliance score based on vulnerability severity."""
        if total_vulns == 0:
            return 100.0
            
        # Weighted scoring
        weights = {'CRITICAL': 10, 'HIGH': 5, 'MEDIUM': 2, 'LOW': 1}
        weighted_score = sum(severity_counts[sev] * weights[sev] for sev in weights)
        max_possible = total_vulns * weights['CRITICAL']
        
        return max(0, 100 - (weighted_score / max_possible * 100))
    
    def _generate_recommendations(self, severity_counts: Dict, vulnerabilities: List) -> List[str]:
        """Generate security recommendations based on scan results."""
        recommendations = []
        
        if severity_counts['CRITICAL'] > 0:
            recommendations.append("🚨 CRITICAL: Address all critical vulnerabilities immediately")
            recommendations.append("🔄 Update base image to latest security patches")
            
        if severity_counts['HIGH'] > 5:
            recommendations.append("⚠️  Consider updating packages with high vulnerabilities")
            
        # Find packages with multiple vulnerabilities
        package_counts = {}
        for vuln in vulnerabilities:
            pkg = vuln.package_name
            package_counts[pkg] = package_counts.get(pkg, 0) + 1
            
        problematic_packages = [pkg for pkg, count in package_counts.items() if count > 3]
        if problematic_packages:
            recommendations.append(f"📦 Consider replacing packages: {', '.join(problematic_packages[:5])}")
            
        if severity_counts['CRITICAL'] == 0 and severity_counts['HIGH'] < 3:
            recommendations.append("✅ Good security posture - maintain regular scanning")
            
        return recommendations
    
    def _generate_overall_assessment(self, scan_results: SecurityScanReport, 
                                   dockerfile_results: Dict, bench_results: Dict) -> Dict:
        """Generate overall security assessment."""
        assessment = {
            'security_grade': 'A',
            'risk_level': 'LOW',
            'actionable_items': [],
            'compliance_status': 'PASS'
        }
        
        # Assess vulnerability scan
        if scan_results:
            if scan_results.critical_count > 0:
                assessment['security_grade'] = 'F'
                assessment['risk_level'] = 'CRITICAL'
                assessment['compliance_status'] = 'FAIL'
            elif scan_results.high_count > 10:
                assessment['security_grade'] = 'D'
                assessment['risk_level'] = 'HIGH'
            elif scan_results.high_count > 5:
                assessment['security_grade'] = 'C'
                assessment['risk_level'] = 'MEDIUM'
            elif scan_results.medium_count > 20:
                assessment['security_grade'] = 'B'
                assessment['risk_level'] = 'LOW'
                
        # Assess Dockerfile issues
        dockerfile_critical = len([i for i in dockerfile_results.get('issues', []) 
                                 if i.get('severity') == 'CRITICAL'])
        if dockerfile_critical > 0:
            assessment['risk_level'] = 'HIGH'
            assessment['compliance_status'] = 'FAIL'
            
        return assessment
    
    def _should_fail_build(self, scan_results: SecurityScanReport, dockerfile_results: Dict) -> bool:
        """Determine if build should fail based on security thresholds."""
        config = self.load_config()
        thresholds = config.get('thresholds', {})
        
        if not scan_results:
            return False
            
        # Check critical vulnerabilities
        if thresholds.get('fail_on_critical', True) and scan_results.critical_count > 0:
            return True
            
        # Check high vulnerabilities
        if thresholds.get('fail_on_high', False) and scan_results.high_count > thresholds.get('max_high', 5):
            return True
            
        # Check Dockerfile critical issues
        dockerfile_critical = len([i for i in dockerfile_results.get('issues', []) 
                                 if i.get('severity') == 'CRITICAL'])
        if dockerfile_critical > 0:
            return True
            
        # Check compliance threshold
        if scan_results.compliance_score < thresholds.get('compliance_threshold', 80.0):
            return True
            
        return False
    
    def _generate_html_security_report(self, report_data: Dict, output_file: Path):
        """Generate HTML security report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Container Security Report - {report_data['meta']['image_name']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .critical {{ background-color: #ffebee; border-left: 5px solid #f44336; }}
                .high {{ background-color: #fff3e0; border-left: 5px solid #ff9800; }}
                .medium {{ background-color: #f3e5f5; border-left: 5px solid #9c27b0; }}
                .low {{ background-color: #e8f5e8; border-left: 5px solid #4caf50; }}
                .metric {{ margin: 10px 0; padding: 15px; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .summary {{ background-color: #f5f5f5; padding: 20px; border-radius: 5px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>🔒 Container Security Report</h1>
            <div class="summary">
                <h2>📊 Summary</h2>
                <p><strong>Image:</strong> {report_data['meta']['image_name']}</p>
                <p><strong>Scan Time:</strong> {report_data['meta']['scan_timestamp']}</p>
        """
        
        if report_data.get('vulnerability_scan'):
            vuln_data = report_data['vulnerability_scan']
            html_content += f"""
                <p><strong>Total Vulnerabilities:</strong> {vuln_data['total_vulnerabilities']}</p>
                <p><strong>Critical:</strong> {vuln_data['critical_count']} | 
                   <strong>High:</strong> {vuln_data['high_count']} | 
                   <strong>Medium:</strong> {vuln_data['medium_count']} | 
                   <strong>Low:</strong> {vuln_data['low_count']}</p>
                <p><strong>Compliance Score:</strong> {vuln_data['compliance_score']:.1f}%</p>
            """
            
        html_content += """
            </div>
            
            <h2>🎯 Recommendations</h2>
            <ul>
        """
        
        if report_data.get('vulnerability_scan', {}).get('recommendations'):
            for rec in report_data['vulnerability_scan']['recommendations']:
                html_content += f"<li>{rec}</li>"
                
        html_content += """
            </ul>
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)

def main():
    """Main container security scanning execution."""
    scanner = ContainerSecurityScanner()
    
    # Determine image name
    image_name = sys.argv[1] if len(sys.argv) > 1 else "fair-credit-scorer:latest"
    
    print(f"🔍 Scanning container security for: {image_name}")
    
    # Run scans
    print("📋 Running vulnerability scan...")
    vuln_results = scanner.scan_with_trivy(image_name)
    
    print("🐳 Scanning Dockerfile...")
    dockerfile_results = scanner.scan_dockerfile()
    
    print("🔧 Running Docker Bench Security...")
    bench_results = scanner.run_docker_bench_security()
    
    # Generate report
    result = scanner.generate_security_report(image_name, vuln_results, dockerfile_results, bench_results)
    
    # Exit with appropriate code
    sys.exit(1 if result == "FAILED" else 0)

if __name__ == "__main__":
    main()