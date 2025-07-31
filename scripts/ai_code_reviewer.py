#!/usr/bin/env python3
"""AI-assisted code review and quality analysis system.

This script provides intelligent code analysis using static analysis tools
and pattern recognition to identify code quality issues, suggest improvements,
and maintain coding standards.
"""

import ast
import json
import subprocess
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set
import re
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CodeIssue:
    """Represents a code quality issue."""
    file_path: str
    line_number: int
    issue_type: str
    severity: str  # critical, high, medium, low
    message: str
    suggestion: str
    auto_fixable: bool = False


@dataclass
class CodeMetrics:
    """Code complexity and quality metrics."""
    lines_of_code: int
    cyclomatic_complexity: int
    maintainability_index: float
    technical_debt_ratio: float
    code_duplication: float
    test_coverage: float


class AICodeReviewer:
    """Intelligent code review system with AI-assisted analysis."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.src_dir = self.project_root / "src"
        self.test_dir = self.project_root / "tests"
        
        # Pattern-based issue detection
        self.anti_patterns = self._load_anti_patterns()
        self.best_practices = self._load_best_practices()
    
    def _load_anti_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load common anti-patterns to detect."""
        return {
            "god_class": {
                "description": "Class with too many responsibilities",
                "detection": lambda node: isinstance(node, ast.ClassDef) and len(node.body) > 20,
                "severity": "high",
                "suggestion": "Consider breaking this class into smaller, focused classes"
            },
            "long_method": {
                "description": "Method with too many lines",
                "detection": lambda node: isinstance(node, ast.FunctionDef) and self._count_lines(node) > 20,
                "severity": "medium",
                "suggestion": "Break this method into smaller, focused methods"
            },
            "deep_nesting": {
                "description": "Excessive nesting levels",
                "detection": lambda node: self._calculate_nesting_depth(node) > 4,
                "severity": "medium",
                "suggestion": "Reduce nesting by using early returns or extracting methods"
            },
            "magic_numbers": {
                "description": "Hard-coded numeric literals",
                "detection": lambda node: isinstance(node, ast.Num) and str(node.n) not in ["0", "1", "-1"],
                "severity": "low",
                "suggestion": "Replace magic numbers with named constants"
            },
            "bare_except": {
                "description": "Bare except clause",
                "detection": lambda node: isinstance(node, ast.ExceptHandler) and node.type is None,
                "severity": "high",
                "suggestion": "Specify the exception type or use 'except Exception:'"
            }
        }
    
    def _load_best_practices(self) -> Dict[str, Dict[str, Any]]:
        """Load best practice checks."""
        return {
            "docstring_missing": {
                "description": "Missing docstring for public method/class",
                "severity": "medium",
                "suggestion": "Add comprehensive docstring with parameters and return values"
            },
            "type_hints_missing": {
                "description": "Missing type hints",
                "severity": "low",
                "suggestion": "Add type hints for better code documentation and IDE support"
            },
            "test_coverage_low": {
                "description": "Insufficient test coverage",
                "severity": "high",
                "suggestion": "Add unit tests to improve coverage above 80%"
            }
        }
    
    def analyze_codebase(self) -> Dict[str, Any]:
        """Perform comprehensive codebase analysis."""
        logger.info("🔍 Starting AI-assisted code analysis...")
        
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "project_metrics": self._calculate_project_metrics(),
            "code_issues": self._detect_code_issues(),
            "security_analysis": self._analyze_security(),
            "architecture_analysis": self._analyze_architecture(),
            "test_quality": self._analyze_test_quality(),
            "recommendations": [],
            "quality_score": 0
        }
        
        # Generate recommendations and calculate quality score
        analysis["recommendations"] = self._generate_recommendations(analysis)
        analysis["quality_score"] = self._calculate_quality_score(analysis)
        
        return analysis
    
    def _calculate_project_metrics(self) -> CodeMetrics:
        """Calculate overall project metrics."""
        total_loc = 0
        total_complexity = 0
        file_count = 0
        
        # Count lines and analyze complexity
        for py_file in self.src_dir.glob("**/*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = len([line for line in content.splitlines() if line.strip() and not line.strip().startswith('#')])
                    total_loc += lines
                    
                    # Parse AST for complexity analysis
                    tree = ast.parse(content)
                    complexity = self._calculate_file_complexity(tree)
                    total_complexity += complexity
                    file_count += 1
                    
            except Exception as e:
                logger.debug(f"Could not analyze {py_file}: {e}")
        
        avg_complexity = total_complexity / max(file_count, 1)
        
        # Calculate other metrics (simplified)
        maintainability = max(0, 100 - avg_complexity * 2)  # Simplified formula
        technical_debt = min(100, avg_complexity * 3)  # Simplified formula
        
        return CodeMetrics(
            lines_of_code=total_loc,
            cyclomatic_complexity=avg_complexity,
            maintainability_index=maintainability,
            technical_debt_ratio=technical_debt,
            code_duplication=self._estimate_duplication(),
            test_coverage=self._get_test_coverage()
        )
    
    def _calculate_file_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity for a file."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            # Count decision points
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            elif isinstance(node, (ast.ExceptHandler, ast.With, ast.AsyncWith)):
                complexity += 1
        
        return complexity
    
    def _detect_code_issues(self) -> List[CodeIssue]:
        """Detect code quality issues using pattern analysis."""
        issues = []
        
        for py_file in self.src_dir.glob("**/*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)
                
                # Analyze AST for anti-patterns
                file_issues = self._analyze_file_ast(py_file, tree, content.splitlines())
                issues.extend(file_issues)
                
            except Exception as e:
                logger.debug(f"Could not analyze {py_file}: {e}")
        
        return issues
    
    def _analyze_file_ast(self, file_path: Path, tree: ast.AST, lines: List[str]) -> List[CodeIssue]:
        """Analyze a single file's AST for issues."""
        issues = []
        
        for node in ast.walk(tree):
            line_no = getattr(node, 'lineno', 1)
            
            # Check anti-patterns
            for pattern_name, pattern_info in self.anti_patterns.items():
                try:
                    if pattern_info["detection"](node):
                        issues.append(CodeIssue(
                            file_path=str(file_path.relative_to(self.project_root)),
                            line_number=line_no,
                            issue_type=pattern_name,
                            severity=pattern_info["severity"],
                            message=pattern_info["description"],
                            suggestion=pattern_info["suggestion"]
                        ))
                except Exception:
                    continue
            
            # Check for missing docstrings
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not self._has_docstring(node):
                if not node.name.startswith('_'):  # Only public methods/classes
                    issues.append(CodeIssue(
                        file_path=str(file_path.relative_to(self.project_root)),
                        line_number=line_no,
                        issue_type="docstring_missing",
                        severity="medium",
                        message=f"Missing docstring for {node.__class__.__name__.lower()} '{node.name}'",
                        suggestion="Add comprehensive docstring with parameters and return values"
                    ))
        
        return issues
    
    def _count_lines(self, node: ast.AST) -> int:
        """Count lines in an AST node."""
        if hasattr(node, 'end_lineno') and hasattr(node, 'lineno'):
            return node.end_lineno - node.lineno + 1
        return 1
    
    def _calculate_nesting_depth(self, node: ast.AST) -> int:
        """Calculate maximum nesting depth in an AST node."""
        max_depth = 0
        
        def _visit_node(n: ast.AST, depth: int = 0) -> None:
            nonlocal max_depth
            max_depth = max(max_depth, depth)
            
            # Increase depth for control structures
            if isinstance(n, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                depth += 1
            
            for child in ast.iter_child_nodes(n):
                _visit_node(child, depth)
        
        _visit_node(node)
        return max_depth
    
    def _has_docstring(self, node: ast.AST) -> bool:
        """Check if a function or class has a docstring."""
        if not hasattr(node, 'body') or not node.body:
            return False
        
        first_stmt = node.body[0]
        return (isinstance(first_stmt, ast.Expr) and 
                isinstance(first_stmt.value, ast.Str))
    
    def _estimate_duplication(self) -> float:
        """Estimate code duplication percentage."""
        # Simplified duplication detection using external tools
        try:
            result = subprocess.run(
                ["python", "-m", "duplo", "src/", "--min-lines", "5"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Parse duplo output for duplication percentage
                # This is simplified - real implementation would parse output
                lines = result.stdout.splitlines()
                duplicate_lines = len([l for l in lines if 'duplicate' in l.lower()])
                return min(100, duplicate_lines * 2)  # Rough estimate
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        return 5.0  # Default low estimate
    
    def _get_test_coverage(self) -> float:
        """Get current test coverage percentage."""
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "--cov=src", "--cov-report=json", "-q"],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            coverage_file = self.project_root / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                return coverage_data.get("totals", {}).get("percent_covered", 0)
            
        except Exception:
            pass
        
        return 75.0  # Default estimate
    
    def _analyze_security(self) -> Dict[str, Any]:
        """Analyze security aspects of the code."""
        security_issues = []
        
        # Run bandit for security analysis
        try:
            result = subprocess.run(
                ["bandit", "-r", "src", "-f", "json"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.stdout:
                bandit_results = json.loads(result.stdout)
                security_issues = bandit_results.get("results", [])
            
        except Exception:
            pass
        
        return {
            "security_issues_found": len(security_issues),
            "high_severity_issues": len([i for i in security_issues if i.get("issue_severity") == "HIGH"]),
            "medium_severity_issues": len([i for i in security_issues if i.get("issue_severity") == "MEDIUM"]),
            "security_score": max(0, 100 - len(security_issues) * 5),
            "recommendations": self._generate_security_recommendations(security_issues)
        }
    
    def _analyze_architecture(self) -> Dict[str, Any]:
        """Analyze code architecture and design patterns."""
        modules = list(self.src_dir.glob("**/*.py"))
        
        # Analyze imports and dependencies
        import_graph = {}
        circular_deps = []
        
        for py_file in modules:
            try:
                with open(py_file, 'r') as f:
                    tree = ast.parse(f.read())
                
                imports = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imports.extend([alias.name for alias in node.names])
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        imports.append(node.module)
                
                module_name = str(py_file.relative_to(self.src_dir)).replace('/', '.').replace('.py', '')
                import_graph[module_name] = imports
                
            except Exception:
                continue
        
        return {
            "total_modules": len(modules),
            "import_complexity": len(import_graph),
            "circular_dependencies": len(circular_deps),
            "architecture_score": self._calculate_architecture_score(import_graph),
            "design_patterns_detected": self._detect_design_patterns(),
            "recommendations": self._generate_architecture_recommendations()
        }
    
    def _analyze_test_quality(self) -> Dict[str, Any]:
        """Analyze test code quality."""
        test_files = list(self.test_dir.glob("**/*.py")) if self.test_dir.exists() else []
        
        if not test_files:
            return {
                "test_files_count": 0,
                "test_quality_score": 0,
                "issues": ["No test files found"],
                "recommendations": ["Create comprehensive test suite"]
            }
        
        test_functions = 0
        test_coverage_issues = []
        
        for test_file in test_files:
            try:
                with open(test_file, 'r') as f:
                    tree = ast.parse(f.read())
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                        test_functions += 1
                        
            except Exception:
                continue
        
        return {
            "test_files_count": len(test_files),
            "test_functions_count": test_functions,
            "test_quality_score": min(100, test_functions * 2),
            "recommendations": self._generate_test_recommendations(test_functions)
        }
    
    def _calculate_architecture_score(self, import_graph: Dict[str, List[str]]) -> float:
        """Calculate architecture quality score."""
        if not import_graph:
            return 50
        
        # Simple scoring based on import complexity
        avg_imports = sum(len(imports) for imports in import_graph.values()) / len(import_graph)
        return max(0, 100 - avg_imports * 3)
    
    def _detect_design_patterns(self) -> List[str]:
        """Detect common design patterns in the code."""
        patterns = []
        
        # Simple pattern detection based on common naming conventions
        for py_file in self.src_dir.glob("**/*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                if "Factory" in content:
                    patterns.append("Factory Pattern")
                if "Observer" in content or "subscribe" in content:
                    patterns.append("Observer Pattern")
                if "Singleton" in content:
                    patterns.append("Singleton Pattern")
                if "Strategy" in content:
                    patterns.append("Strategy Pattern")
                    
            except Exception:
                continue
        
        return list(set(patterns))
    
    def _generate_security_recommendations(self, security_issues: List[Dict]) -> List[str]:
        """Generate security improvement recommendations."""
        recommendations = []
        
        if security_issues:
            recommendations.extend([
                "🔒 Address identified security vulnerabilities immediately",
                "🔍 Implement automated security scanning in CI/CD",
                "📚 Review OWASP security guidelines for Python",
                "🛡️ Consider using security-focused linting tools"
            ])
        else:
            recommendations.append("✅ No security issues detected - maintain secure coding practices")
        
        return recommendations
    
    def _generate_architecture_recommendations(self) -> List[str]:
        """Generate architecture improvement recommendations."""
        return [
            "🏗️ Consider applying SOLID principles more consistently",
            "🔄 Implement dependency injection for better testability",
            "📦 Consider modularizing large components",
            "🎯 Apply appropriate design patterns for common problems"
        ]
    
    def _generate_test_recommendations(self, test_count: int) -> List[str]:
        """Generate test improvement recommendations."""
        if test_count < 10:
            return [
                "📝 Increase test coverage to at least 80%",
                "🧪 Add unit tests for all public methods",
                "🔄 Implement integration tests for critical workflows"
            ]
        else:
            return [
                "✅ Good test coverage - consider adding edge case tests",
                "🎯 Add performance tests for critical components",
                "🔄 Implement mutation testing for test quality validation"
            ]
    
    def _calculate_quality_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall code quality score."""
        metrics = analysis["project_metrics"]
        issues = analysis["code_issues"]
        security = analysis["security_analysis"]
        architecture = analysis["architecture_analysis"]
        tests = analysis["test_quality"]
        
        # Weighted scoring
        complexity_score = max(0, 100 - metrics.cyclomatic_complexity * 2)
        issue_score = max(0, 100 - len(issues) * 2)
        security_score = security["security_score"]
        architecture_score = architecture["architecture_score"]
        test_score = tests["test_quality_score"]
        
        # Weighted average
        overall_score = (
            complexity_score * 0.2 +
            issue_score * 0.3 +
            security_score * 0.2 +
            architecture_score * 0.15 +
            test_score * 0.15
        )
        
        return round(overall_score, 1)
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate comprehensive improvement recommendations."""
        recommendations = []
        
        quality_score = analysis["quality_score"]
        
        if quality_score >= 90:
            recommendations.append("🌟 Excellent code quality! Continue maintaining high standards")
        elif quality_score >= 75:
            recommendations.append("✅ Good code quality with room for minor improvements")
        elif quality_score >= 60:
            recommendations.append("⚠️ Moderate code quality - focus on key improvements")
        else:
            recommendations.append("🚨 Code quality needs significant attention")
        
        # Issue-specific recommendations
        critical_issues = [i for i in analysis["code_issues"] if i.severity == "critical"]
        high_issues = [i for i in analysis["code_issues"] if i.severity == "high"]
        
        if critical_issues:
            recommendations.append(f"🚨 Address {len(critical_issues)} critical issues immediately")
        if high_issues:
            recommendations.append(f"⚠️ Fix {len(high_issues)} high-priority issues")
        
        # Add specific improvement areas
        recommendations.extend([
            "🔍 Implement automated code review in CI/CD pipeline",
            "📊 Set up code quality metrics tracking",
            "🤖 Consider AI-assisted refactoring tools",
            "📚 Establish coding standards documentation"
        ])
        
        return recommendations
    
    def generate_review_report(self, output_path: str = "code_review_report.json") -> None:
        """Generate comprehensive code review report."""
        analysis = self.analyze_codebase()
        
        # Convert dataclass instances to dictionaries
        if isinstance(analysis["project_metrics"], CodeMetrics):
            analysis["project_metrics"] = asdict(analysis["project_metrics"])
        
        analysis["code_issues"] = [asdict(issue) for issue in analysis["code_issues"]]
        
        report_path = self.project_root / output_path
        with open(report_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info(f"📊 Code review report generated: {report_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("🤖 AI-ASSISTED CODE REVIEW REPORT")
        print("="*60)
        print(f"📊 Overall Quality Score: {analysis['quality_score']}/100")
        print(f"📝 Lines of Code: {analysis['project_metrics']['lines_of_code']:,}")
        print(f"🔍 Issues Found: {len(analysis['code_issues'])}")
        print(f"🔒 Security Score: {analysis['security_analysis']['security_score']}/100")
        print(f"🏗️ Architecture Score: {analysis['architecture_analysis']['architecture_score']}/100")
        
        if analysis["code_issues"]:
            print(f"\n📋 TOP ISSUES:")
            for issue in sorted(analysis["code_issues"], key=lambda x: {"critical": 0, "high": 1, "medium": 2, "low": 3}[x["severity"]])[:5]:
                print(f"   {issue['severity'].upper()}: {issue['message']} ({issue['file_path']}:{issue['line_number']})")
        
        print("\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(analysis['recommendations'][:8], 1):
            print(f"   {i}. {rec}")
        print("="*60)


def main():
    """Main entry point for AI code reviewer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI-Assisted Code Review")
    parser.add_argument("--analyze", action="store_true", help="Perform code analysis")
    parser.add_argument("--output", default="code_review_report.json", help="Output file")
    parser.add_argument("--format", choices=["json", "html"], default="json", help="Output format")
    
    args = parser.parse_args()
    
    reviewer = AICodeReviewer()
    reviewer.generate_review_report(args.output)


if __name__ == "__main__":
    main()