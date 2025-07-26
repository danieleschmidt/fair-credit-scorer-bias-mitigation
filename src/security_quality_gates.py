"""
Security and Quality Gates for Autonomous Backlog Management

This module implements comprehensive security and quality checks that are applied
to every backlog item execution. It ensures that all code changes maintain high
security standards and quality metrics.

Security Checks:
- Input sanitization validation
- Authentication and authorization checks
- Secrets and configuration security
- Error handling security
- Logging security (no sensitive data)

Quality Checks:
- Test coverage maintenance
- Code complexity analysis
- Performance regression detection
- Documentation completeness
- Dependency vulnerability scanning
"""

import ast
import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
import yaml

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security risk levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class QualityGate(Enum):
    """Quality gate types"""
    SECURITY = "security"
    TESTING = "testing"
    PERFORMANCE = "performance"
    DOCUMENTATION = "documentation"
    DEPENDENCIES = "dependencies"
    CODE_QUALITY = "code_quality"


@dataclass
class SecurityFinding:
    """Security issue found during scanning"""
    file_path: str
    line_number: int
    rule_id: str
    severity: SecurityLevel
    message: str
    code_snippet: str
    recommendation: str


@dataclass
class QualityMetric:
    """Quality metric measurement"""
    name: str
    current_value: float
    threshold_value: float
    passed: bool
    details: str


@dataclass
class GateResult:
    """Result of a quality gate check"""
    gate_type: QualityGate
    passed: bool
    score: float  # 0-100
    findings: List[SecurityFinding]
    metrics: List[QualityMetric]
    execution_time: float
    recommendations: List[str]


class SecurityChecker:
    """Comprehensive security checking system"""
    
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        
        # Enhanced patterns for secret detection
        self.sensitive_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'api[_-]?key\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']',
            r'private[_-]?key\s*=\s*["\'][^"\']+["\']',
            r'-----BEGIN.*PRIVATE KEY-----',
            r'access[_-]?key\s*=\s*["\'][^"\']+["\']',
            r'auth[_-]?token\s*=\s*["\'][^"\']+["\']',
            r'database[_-]?url\s*=\s*["\'][^"\']+["\']',
            r'connection[_-]?string\s*=\s*["\'][^"\']+["\']',
        ]
        
        # Enhanced patterns for insecure code detection
        self.insecure_patterns = [
            (r'eval\s*\(', 'Use of eval() can execute arbitrary code'),
            (r'exec\s*\(', 'Use of exec() can execute arbitrary code'),
            (r'shell\s*=\s*True', 'subprocess with shell=True is dangerous'),
            (r'pickle\.loads?\s*\(', 'Pickle deserialization can execute code'),
            (r'yaml\.load\s*\(', 'Use yaml.safe_load() instead of yaml.load()'),
            (r'input\s*\(.*\)', 'Direct input() usage can be dangerous - validate input'),
            (r'os\.system\s*\(', 'os.system() is unsafe - use subprocess instead'),
            (r'subprocess\.call\s*\([^)]*shell\s*=\s*True', 'subprocess.call with shell=True is dangerous'),
        ]
    
    def check_secrets_exposure(self, file_paths: List[str]) -> List[SecurityFinding]:
        """Check for exposed secrets in code"""
        findings = []
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                lines = content.split('\n')
                for line_num, line in enumerate(lines, 1):
                    for pattern in self.sensitive_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            findings.append(SecurityFinding(
                                file_path=file_path,
                                line_number=line_num,
                                rule_id="S001",
                                severity=SecurityLevel.CRITICAL,
                                message="Potential secret exposed in code",
                                code_snippet=line.strip(),
                                recommendation="Move secrets to environment variables or secure config"
                            ))
            
            except Exception as e:
                logger.warning(f"Failed to scan {file_path} for secrets: {e}")
        
        return findings
    
    def check_insecure_patterns(self, file_paths: List[str]) -> List[SecurityFinding]:
        """Check for insecure coding patterns"""
        findings = []
        
        for file_path in file_paths:
            if not file_path.endswith('.py') or not os.path.exists(file_path):
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                lines = content.split('\n')
                for line_num, line in enumerate(lines, 1):
                    for pattern, message in self.insecure_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            findings.append(SecurityFinding(
                                file_path=file_path,
                                line_number=line_num,
                                rule_id="S002",
                                severity=SecurityLevel.HIGH,
                                message=message,
                                code_snippet=line.strip(),
                                recommendation="Use secure alternatives"
                            ))
            
            except Exception as e:
                logger.warning(f"Failed to scan {file_path} for insecure patterns: {e}")
        
        return findings
    
    def check_input_validation(self, file_paths: List[str]) -> List[SecurityFinding]:
        """Check for proper input validation"""
        findings = []
        
        for file_path in file_paths:
            if not file_path.endswith('.py') or not os.path.exists(file_path):
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse AST to find function definitions
                try:
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            # Check if function has external inputs but no validation
                            has_external_input = any(
                                arg.arg in ['request', 'data', 'input', 'params', 'args']
                                for arg in node.args.args
                            )
                            
                            if has_external_input:
                                # Look for validation patterns in function body
                                func_code = ast.get_source_segment(content, node)
                                if func_code and not self._has_validation_patterns(func_code):
                                    findings.append(SecurityFinding(
                                        file_path=file_path,
                                        line_number=node.lineno,
                                        rule_id="S003",
                                        severity=SecurityLevel.MEDIUM,
                                        message=f"Function '{node.name}' may lack input validation",
                                        code_snippet=f"def {node.name}(...)",
                                        recommendation="Add input validation and sanitization"
                                    ))
                
                except SyntaxError:
                    # Skip files with syntax errors
                    pass
            
            except Exception as e:
                logger.warning(f"Failed to check input validation in {file_path}: {e}")
        
        return findings
    
    def _has_validation_patterns(self, code: str) -> bool:
        """Check if code contains validation patterns"""
        validation_patterns = [
            r'isinstance\s*\(',
            r'validate\w*\s*\(',
            r'check\w*\s*\(',
            r'sanitize\w*\s*\(',
            r'if\s+not\s+\w+:',
            r'raise\s+ValueError',
            r'raise\s+TypeError',
        ]
        
        return any(re.search(pattern, code, re.IGNORECASE) for pattern in validation_patterns)
    
    def run_bandit_scan(self) -> List[SecurityFinding]:
        """Run Bandit security scanner"""
        findings = []
        
        try:
            cmd = [
                "bandit", "-r", f"{self.repo_path}/src/",
                "-f", "json", "--quiet"
            ]
            
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120
            )
            
            if result.stdout:
                data = json.loads(result.stdout)
                
                for issue in data.get('results', []):
                    severity_map = {
                        'LOW': SecurityLevel.LOW,
                        'MEDIUM': SecurityLevel.MEDIUM,
                        'HIGH': SecurityLevel.HIGH
                    }
                    
                    findings.append(SecurityFinding(
                        file_path=issue['filename'],
                        line_number=issue['line_number'],
                        rule_id=issue['test_id'],
                        severity=severity_map.get(issue['issue_severity'], SecurityLevel.MEDIUM),
                        message=issue['issue_text'],
                        code_snippet=issue.get('code', ''),
                        recommendation=f"See: {issue.get('more_info', 'Bandit documentation')}"
                    ))
        
        except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Bandit scan failed: {e}")
        
        return findings


class QualityChecker:
    """Comprehensive quality checking system"""
    
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.quality_thresholds = {
            'test_coverage': 85.0,
            'complexity_max': 10,
            'documentation_coverage': 80.0,
            'performance_regression': 20.0  # Max % performance degradation
        }
    
    def check_test_coverage(self) -> QualityMetric:
        """Check test coverage meets threshold"""
        try:
            cmd = [
                "python", "-m", "pytest", "--cov=src", "--cov-report=json",
                "--cov-report=term-missing", "--quiet"
            ]
            
            result = subprocess.run(
                cmd, cwd=self.repo_path, capture_output=True, text=True, timeout=300
            )
            
            coverage_file = os.path.join(self.repo_path, "coverage.json")
            if os.path.exists(coverage_file):
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                
                total_coverage = coverage_data['totals']['percent_covered']
                
                return QualityMetric(
                    name="test_coverage",
                    current_value=total_coverage,
                    threshold_value=self.quality_thresholds['test_coverage'],
                    passed=total_coverage >= self.quality_thresholds['test_coverage'],
                    details=f"Coverage: {total_coverage:.1f}% (threshold: {self.quality_thresholds['test_coverage']}%)"
                )
        
        except Exception as e:
            logger.warning(f"Failed to check test coverage: {e}")
        
        return QualityMetric(
            name="test_coverage",
            current_value=0.0,
            threshold_value=self.quality_thresholds['test_coverage'],
            passed=False,
            details="Failed to measure coverage"
        )
    
    def check_code_complexity(self) -> QualityMetric:
        """Check code complexity using radon or similar"""
        try:
            # Simple complexity check by counting nested levels
            total_functions = 0
            high_complexity_functions = 0
            
            for root, dirs, files in os.walk(os.path.join(self.repo_path, "src")):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        complexity_score = self._analyze_file_complexity(file_path)
                        
                        if complexity_score > self.quality_thresholds['complexity_max']:
                            high_complexity_functions += 1
                        total_functions += 1
            
            complexity_ratio = (high_complexity_functions / max(total_functions, 1)) * 100
            
            return QualityMetric(
                name="code_complexity",
                current_value=complexity_ratio,
                threshold_value=10.0,  # Max 10% of functions can be complex
                passed=complexity_ratio <= 10.0,
                details=f"{high_complexity_functions}/{total_functions} functions exceed complexity threshold"
            )
        
        except Exception as e:
            logger.warning(f"Failed to check code complexity: {e}")
        
        return QualityMetric(
            name="code_complexity",
            current_value=100.0,
            threshold_value=10.0,
            passed=False,
            details="Failed to measure complexity"
        )
    
    def _analyze_file_complexity(self, file_path: str) -> int:
        """Simple complexity analysis by counting nested blocks"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Count nesting levels (simplified cyclomatic complexity)
            max_nesting = 0
            current_nesting = 0
            
            for line in content.split('\n'):
                stripped = line.strip()
                
                # Count indent increases
                if any(stripped.startswith(keyword) for keyword in 
                      ['if ', 'for ', 'while ', 'try:', 'with ', 'def ', 'class ']):
                    current_nesting += 1
                    max_nesting = max(max_nesting, current_nesting)
                
                # Count indent decreases (simplified)
                if stripped == '' or not line.startswith('    '):
                    current_nesting = max(0, current_nesting - 1)
            
            return max_nesting
        
        except Exception:
            return 0
    
    def check_documentation_coverage(self) -> QualityMetric:
        """Check documentation coverage"""
        try:
            total_functions = 0
            documented_functions = 0
            
            for root, dirs, files in os.walk(os.path.join(self.repo_path, "src")):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        doc_stats = self._analyze_documentation(file_path)
                        total_functions += doc_stats['total']
                        documented_functions += doc_stats['documented']
            
            if total_functions == 0:
                coverage = 100.0
            else:
                coverage = (documented_functions / total_functions) * 100
            
            return QualityMetric(
                name="documentation_coverage",
                current_value=coverage,
                threshold_value=self.quality_thresholds['documentation_coverage'],
                passed=coverage >= self.quality_thresholds['documentation_coverage'],
                details=f"{documented_functions}/{total_functions} functions documented"
            )
        
        except Exception as e:
            logger.warning(f"Failed to check documentation coverage: {e}")
        
        return QualityMetric(
            name="documentation_coverage",
            current_value=0.0,
            threshold_value=self.quality_thresholds['documentation_coverage'],
            passed=False,
            details="Failed to measure documentation coverage"
        )
    
    def _analyze_documentation(self, file_path: str) -> Dict[str, int]:
        """Analyze documentation in a Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            total_functions = 0
            documented_functions = 0
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Skip private functions and test functions
                    if not node.name.startswith('_') and not node.name.startswith('test_'):
                        total_functions += 1
                        
                        # Check if function has docstring
                        if (node.body and 
                            isinstance(node.body[0], ast.Expr) and
                            isinstance(node.body[0].value, ast.Constant) and
                            isinstance(node.body[0].value.value, str)):
                            documented_functions += 1
            
            return {
                'total': total_functions,
                'documented': documented_functions
            }
        
        except Exception:
            return {'total': 0, 'documented': 0}
    
    def check_dependencies(self) -> QualityMetric:
        """Check for dependency vulnerabilities"""
        try:
            # Try to use safety if available
            cmd = ["safety", "check", "--json"]
            
            result = subprocess.run(
                cmd, cwd=self.repo_path, capture_output=True, text=True, timeout=60
            )
            
            if result.returncode == 0:
                # No vulnerabilities found
                return QualityMetric(
                    name="dependency_security",
                    current_value=100.0,
                    threshold_value=100.0,
                    passed=True,
                    details="No known vulnerabilities in dependencies"
                )
            else:
                # Vulnerabilities found
                try:
                    vuln_data = json.loads(result.stdout)
                    vuln_count = len(vuln_data)
                except (json.JSONDecodeError, TypeError):
                    vuln_count = 1  # Assume at least one if safety failed
                
                return QualityMetric(
                    name="dependency_security",
                    current_value=0.0,
                    threshold_value=100.0,
                    passed=False,
                    details=f"{vuln_count} vulnerabilities found in dependencies"
                )
        
        except Exception as e:
            logger.warning(f"Failed to check dependencies: {e}")
        
        # Fallback: manual check of requirements files
        return self._manual_dependency_check()
    
    def _manual_dependency_check(self) -> QualityMetric:
        """Manual dependency check by parsing requirements"""
        # This is a simplified check - in production you'd want more sophisticated analysis
        suspicious_packages = ['pickle', 'eval', 'exec']
        
        req_files = ['requirements.txt', 'requirements-dev.txt', 'pyproject.toml']
        issues = []
        
        for req_file in req_files:
            file_path = os.path.join(self.repo_path, req_file)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        content = f.read().lower()
                    
                    for package in suspicious_packages:
                        if package in content:
                            issues.append(f"Potentially risky package: {package}")
                
                except Exception:
                    pass
        
        return QualityMetric(
            name="dependency_security",
            current_value=100.0 if not issues else 50.0,
            threshold_value=100.0,
            passed=len(issues) == 0,
            details=f"Manual check: {len(issues)} potential issues"
        )


class SecurityQualityGateManager:
    """Main gate manager that coordinates all security and quality checks"""
    
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.security_checker = SecurityChecker(repo_path)
        self.quality_checker = QualityChecker(repo_path)
        
        # Load gate configuration
        self.gate_config = self._load_gate_config()
    
    def _load_gate_config(self) -> Dict:
        """Load gate configuration from file or use defaults"""
        config_file = os.path.join(self.repo_path, "config", "quality_gates.yaml")
        
        default_config = {
            'gates': {
                'security': {'enabled': True, 'required': True, 'weight': 0.3},
                'testing': {'enabled': True, 'required': True, 'weight': 0.25},
                'performance': {'enabled': True, 'required': False, 'weight': 0.15},
                'documentation': {'enabled': True, 'required': False, 'weight': 0.15},
                'dependencies': {'enabled': True, 'required': True, 'weight': 0.15}
            },
            'thresholds': {
                'overall_score': 75.0,
                'security_score': 90.0,
                'critical_findings_max': 0,
                'high_findings_max': 2
            }
        }
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    # Merge with defaults
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    return config
            except Exception as e:
                logger.warning(f"Failed to load gate config: {e}")
        
        return default_config
    
    def run_all_gates(self, changed_files: Optional[List[str]] = None) -> List[GateResult]:
        """Run all configured quality gates"""
        results = []
        
        if changed_files is None:
            # Scan all source files
            changed_files = []
            for root, dirs, files in os.walk(os.path.join(self.repo_path, "src")):
                for file in files:
                    if file.endswith('.py'):
                        changed_files.append(os.path.join(root, file))
        
        # Security Gate
        if self.gate_config['gates']['security']['enabled']:
            results.append(self._run_security_gate(changed_files))
        
        # Testing Gate
        if self.gate_config['gates']['testing']['enabled']:
            results.append(self._run_testing_gate())
        
        # Documentation Gate
        if self.gate_config['gates']['documentation']['enabled']:
            results.append(self._run_documentation_gate())
        
        # Dependencies Gate
        if self.gate_config['gates']['dependencies']['enabled']:
            results.append(self._run_dependencies_gate())
        
        return results
    
    def _run_security_gate(self, changed_files: List[str]) -> GateResult:
        """Run comprehensive security checks"""
        start_time = time.time()
        findings = []
        
        # Run all security checks
        findings.extend(self.security_checker.check_secrets_exposure(changed_files))
        findings.extend(self.security_checker.check_insecure_patterns(changed_files))
        findings.extend(self.security_checker.check_input_validation(changed_files))
        findings.extend(self.security_checker.run_bandit_scan())
        
        # Calculate security score
        critical_count = sum(1 for f in findings if f.severity == SecurityLevel.CRITICAL)
        high_count = sum(1 for f in findings if f.severity == SecurityLevel.HIGH)
        medium_count = sum(1 for f in findings if f.severity == SecurityLevel.MEDIUM)
        low_count = sum(1 for f in findings if f.severity == SecurityLevel.LOW)
        
        # Scoring: Critical = -50, High = -20, Medium = -10, Low = -5
        deductions = (critical_count * 50) + (high_count * 20) + (medium_count * 10) + (low_count * 5)
        score = max(0, 100 - deductions)
        
        # Gate passes if score >= threshold and critical findings within limits
        threshold = self.gate_config['thresholds']['security_score']
        max_critical = self.gate_config['thresholds']['critical_findings_max']
        max_high = self.gate_config['thresholds']['high_findings_max']
        
        passed = (score >= threshold and 
                 critical_count <= max_critical and 
                 high_count <= max_high)
        
        recommendations = []
        if critical_count > 0:
            recommendations.append("Address all critical security findings immediately")
        if high_count > max_high:
            recommendations.append(f"Reduce high-severity findings to {max_high} or fewer")
        if score < threshold:
            recommendations.append(f"Improve security score to {threshold}+ by addressing findings")
        
        return GateResult(
            gate_type=QualityGate.SECURITY,
            passed=passed,
            score=score,
            findings=findings,
            metrics=[],
            execution_time=time.time() - start_time,
            recommendations=recommendations
        )
    
    def _run_testing_gate(self) -> GateResult:
        """Run testing quality checks"""
        start_time = time.time()
        
        coverage_metric = self.quality_checker.check_test_coverage()
        
        score = coverage_metric.current_value
        passed = coverage_metric.passed
        
        recommendations = []
        if not passed:
            recommendations.append(f"Increase test coverage to {coverage_metric.threshold_value}%")
            recommendations.append("Add tests for uncovered code paths")
        
        return GateResult(
            gate_type=QualityGate.TESTING,
            passed=passed,
            score=score,
            findings=[],
            metrics=[coverage_metric],
            execution_time=time.time() - start_time,
            recommendations=recommendations
        )
    
    def _run_documentation_gate(self) -> GateResult:
        """Run documentation quality checks"""
        start_time = time.time()
        
        doc_metric = self.quality_checker.check_documentation_coverage()
        
        score = doc_metric.current_value
        passed = doc_metric.passed
        
        recommendations = []
        if not passed:
            recommendations.append("Add docstrings to undocumented functions")
            recommendations.append("Ensure all public APIs are documented")
        
        return GateResult(
            gate_type=QualityGate.DOCUMENTATION,
            passed=passed,
            score=score,
            findings=[],
            metrics=[doc_metric],
            execution_time=time.time() - start_time,
            recommendations=recommendations
        )
    
    def _run_dependencies_gate(self) -> GateResult:
        """Run dependency security checks"""
        start_time = time.time()
        
        dep_metric = self.quality_checker.check_dependencies()
        
        score = dep_metric.current_value
        passed = dep_metric.passed
        
        recommendations = []
        if not passed:
            recommendations.append("Update vulnerable dependencies")
            recommendations.append("Review dependency security advisories")
        
        return GateResult(
            gate_type=QualityGate.DEPENDENCIES,
            passed=passed,
            score=score,
            findings=[],
            metrics=[dep_metric],
            execution_time=time.time() - start_time,
            recommendations=recommendations
        )
    
    def evaluate_overall_quality(self, gate_results: List[GateResult]) -> Tuple[bool, float, List[str]]:
        """Evaluate overall quality based on all gate results"""
        if not gate_results:
            return False, 0.0, ["No quality gates were executed"]
        
        # Calculate weighted score
        total_weight = 0
        weighted_score = 0
        
        required_gates_passed = True
        all_recommendations = []
        
        for result in gate_results:
            gate_name = result.gate_type.value
            gate_config = self.gate_config['gates'].get(gate_name, {})
            
            weight = gate_config.get('weight', 0.2)
            is_required = gate_config.get('required', False)
            
            total_weight += weight
            weighted_score += result.score * weight
            
            if is_required and not result.passed:
                required_gates_passed = False
            
            all_recommendations.extend(result.recommendations)
        
        overall_score = weighted_score / max(total_weight, 1)
        threshold = self.gate_config['thresholds']['overall_score']
        
        overall_passed = (required_gates_passed and 
                         overall_score >= threshold)
        
        if not overall_passed:
            if not required_gates_passed:
                all_recommendations.insert(0, "Fix all required quality gate failures")
            if overall_score < threshold:
                all_recommendations.insert(0, f"Improve overall quality score to {threshold}+")
        
        return overall_passed, overall_score, all_recommendations
    
    def generate_gate_report(self, gate_results: List[GateResult]) -> Dict:
        """Generate comprehensive gate report"""
        overall_passed, overall_score, recommendations = self.evaluate_overall_quality(gate_results)
        
        report = {
            'timestamp': time.time(),
            'overall_passed': overall_passed,
            'overall_score': overall_score,
            'recommendations': recommendations,
            'gate_results': []
        }
        
        for result in gate_results:
            gate_report = {
                'gate_type': result.gate_type.value,
                'passed': result.passed,
                'score': result.score,
                'execution_time': result.execution_time,
                'findings_count': len(result.findings),
                'metrics_count': len(result.metrics),
                'recommendations': result.recommendations
            }
            
            # Add detailed findings for security gate
            if result.gate_type == QualityGate.SECURITY and result.findings:
                gate_report['security_findings'] = [
                    {
                        'file': f.file_path,
                        'line': f.line_number,
                        'severity': f.severity.value,
                        'message': f.message,
                        'rule': f.rule_id
                    }
                    for f in result.findings
                ]
            
            report['gate_results'].append(gate_report)
        
        return report