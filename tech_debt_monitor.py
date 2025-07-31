#!/usr/bin/env python3
"""
Technical debt monitoring and automated tracking system.
Part of advanced SDLC enhancement suite for repository maturity optimization.
"""

import ast
import json
import subprocess
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import yaml

@dataclass
class TechDebtItem:
    """Represents a single technical debt item."""
    type: str
    severity: str
    location: str
    description: str
    estimated_hours: float
    created_date: str
    priority: int
    category: str
    
@dataclass
class CodeQualityMetrics:
    """Code quality measurements."""
    lines_of_code: int
    cyclomatic_complexity: float
    maintainability_index: float
    code_duplication_pct: float
    test_coverage_pct: float
    tech_debt_ratio: float

class TechnicalDebtAnalyzer:
    """Automated technical debt detection and monitoring."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.debt_file = self.project_root / "tech_debt_registry.json"
        self.config_file = self.project_root / "tech_debt_config.yml"
        self.ensure_config()
        
    def ensure_config(self):
        """Ensure technical debt configuration exists."""
        if not self.config_file.exists():
            default_config = {
                'thresholds': {
                    'cyclomatic_complexity': 10,
                    'function_length': 50,
                    'class_length': 200,
                    'file_length': 500,
                    'parameter_count': 6,
                    'nesting_depth': 4
                },
                'severity_weights': {
                    'critical': 10,
                    'high': 5,
                    'medium': 2,
                    'low': 1
                },
                'categories': [
                    'code_smell',
                    'performance',
                    'security',
                    'maintainability',
                    'testing',
                    'documentation',
                    'architecture'
                ],
                'exclusions': [
                    'tests/*',
                    '__pycache__/*',
                    '*.pyc',
                    'build/*',
                    'dist/*'
                ]
            }
            
            with open(self.config_file, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
                
    def load_config(self) -> Dict[str, Any]:
        """Load technical debt configuration."""
        with open(self.config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def analyze_code_complexity(self) -> List[TechDebtItem]:
        """Analyze code complexity and identify debt items."""
        debt_items = []
        config = self.load_config()
        thresholds = config['thresholds']
        
        for py_file in self.project_root.rglob("*.py"):
            if self._should_exclude_file(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)
                    
                # Analyze functions
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        complexity = self._calculate_cyclomatic_complexity(node)
                        function_length = self._get_function_length(node, content)
                        param_count = len(node.args.args)
                        
                        if complexity > thresholds['cyclomatic_complexity']:
                            debt_items.append(TechDebtItem(
                                type="high_complexity",
                                severity="high" if complexity > thresholds['cyclomatic_complexity'] * 1.5 else "medium",
                                location=f"{py_file}:{node.lineno}:{node.name}",
                                description=f"Function has high cyclomatic complexity ({complexity})",
                                estimated_hours=complexity * 0.5,
                                created_date=datetime.now().isoformat(),
                                priority=8 if complexity > 15 else 5,
                                category="maintainability"
                            ))
                            
                        if function_length > thresholds['function_length']:
                            debt_items.append(TechDebtItem(
                                type="long_function",
                                severity="medium",
                                location=f"{py_file}:{node.lineno}:{node.name}",
                                description=f"Function is too long ({function_length} lines)",
                                estimated_hours=function_length * 0.1,
                                created_date=datetime.now().isoformat(),
                                priority=4,
                                category="code_smell"
                            ))
                            
                        if param_count > thresholds['parameter_count']:
                            debt_items.append(TechDebtItem(
                                type="parameter_overload",
                                severity="low",
                                location=f"{py_file}:{node.lineno}:{node.name}",
                                description=f"Function has too many parameters ({param_count})",
                                estimated_hours=1.0,
                                created_date=datetime.now().isoformat(),
                                priority=3,
                                category="code_smell"
                            ))
                    
                    elif isinstance(node, ast.ClassDef):
                        class_length = self._get_class_length(node, content)
                        if class_length > thresholds['class_length']:
                            debt_items.append(TechDebtItem(
                                type="large_class",
                                severity="medium",
                                location=f"{py_file}:{node.lineno}:{node.name}",
                                description=f"Class is too large ({class_length} lines)",
                                estimated_hours=class_length * 0.05,
                                created_date=datetime.now().isoformat(),
                                priority=5,
                                category="architecture"
                            ))
                            
            except Exception as e:
                print(f"Error analyzing {py_file}: {e}")
                
        return debt_items
    
    def detect_code_duplication(self) -> List[TechDebtItem]:
        """Detect code duplication using AST analysis."""
        debt_items = []
        file_hashes = {}
        
        for py_file in self.project_root.rglob("*.py"):
            if self._should_exclude_file(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)
                    
                # Extract function signatures and bodies for comparison
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        func_hash = self._get_function_hash(node)
                        location = f"{py_file}:{node.lineno}:{node.name}"
                        
                        if func_hash in file_hashes:
                            # Potential duplication found
                            debt_items.append(TechDebtItem(
                                type="code_duplication",
                                severity="medium",
                                location=location,
                                description=f"Potential code duplication with {file_hashes[func_hash]}",
                                estimated_hours=2.0,
                                created_date=datetime.now().isoformat(),
                                priority=6,
                                category="maintainability"
                            ))
                        else:
                            file_hashes[func_hash] = location
                            
            except Exception as e:
                print(f"Error analyzing duplication in {py_file}: {e}")
                
        return debt_items
    
    def analyze_test_coverage_gaps(self) -> List[TechDebtItem]:
        """Identify functions/classes lacking test coverage."""
        debt_items = []
        
        try:
            # Run coverage analysis
            result = subprocess.run(
                ["python", "-m", "pytest", "--cov=src", "--cov-report=json", "--cov-report=term-missing"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            coverage_file = self.project_root / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                    
                for filename, file_data in coverage_data.get('files', {}).items():
                    coverage_pct = file_data.get('summary', {}).get('percent_covered', 0)
                    missing_lines = file_data.get('missing_lines', [])
                    
                    if coverage_pct < 80:  # Below 80% coverage threshold
                        debt_items.append(TechDebtItem(
                            type="low_test_coverage",
                            severity="high" if coverage_pct < 50 else "medium",
                            location=filename,
                            description=f"Low test coverage ({coverage_pct:.1f}%), missing {len(missing_lines)} lines",
                            estimated_hours=(100 - coverage_pct) * 0.1,
                            created_date=datetime.now().isoformat(),
                            priority=7 if coverage_pct < 50 else 5,
                            category="testing"
                        ))
                        
        except Exception as e:
            print(f"Error analyzing test coverage: {e}")
            
        return debt_items
    
    def analyze_security_issues(self) -> List[TechDebtItem]:
        """Analyze security-related technical debt using bandit."""
        debt_items = []
        
        try:
            result = subprocess.run(
                ["bandit", "-r", "src", "-f", "json", "-o", "bandit-debt-report.json"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            bandit_file = self.project_root / "bandit-debt-report.json"
            if bandit_file.exists():
                with open(bandit_file, 'r') as f:
                    bandit_data = json.load(f)
                    
                for issue in bandit_data.get('results', []):
                    severity_map = {'HIGH': 'high', 'MEDIUM': 'medium', 'LOW': 'low'}
                    severity = severity_map.get(issue.get('issue_severity', 'MEDIUM'), 'medium')
                    
                    debt_items.append(TechDebtItem(
                        type="security_issue",
                        severity=severity,
                        location=f"{issue.get('filename', '')}:{issue.get('line_number', 0)}",
                        description=f"{issue.get('test_name', '')}: {issue.get('issue_text', '')}",
                        estimated_hours=3.0 if severity == 'high' else 1.5 if severity == 'medium' else 0.5,
                        created_date=datetime.now().isoformat(),
                        priority=9 if severity == 'high' else 6 if severity == 'medium' else 3,
                        category="security"
                    ))
                    
        except Exception as e:
            print(f"Error analyzing security issues: {e}")
            
        return debt_items
    
    def calculate_debt_metrics(self, debt_items: List[TechDebtItem]) -> CodeQualityMetrics:
        """Calculate overall technical debt metrics."""
        total_hours = sum(item.estimated_hours for item in debt_items)
        total_files = len(list(self.project_root.rglob("*.py")))
        total_loc = self._count_lines_of_code()
        
        # Calculate complexity average
        complexity_items = [item for item in debt_items if item.type == "high_complexity"]
        avg_complexity = sum(float(item.description.split('(')[1].split(')')[0]) 
                           for item in complexity_items) / max(len(complexity_items), 1)
        
        # Calculate maintainability index (simplified)
        maintainability = max(0, 100 - (total_hours / max(total_loc / 1000, 1)))
        
        # Calculate duplication percentage
        duplication_items = len([item for item in debt_items if item.type == "code_duplication"])
        duplication_pct = (duplication_items / max(total_files, 1)) * 100
        
        # Calculate tech debt ratio (hours per KLOC)
        debt_ratio = total_hours / max(total_loc / 1000, 1)
        
        return CodeQualityMetrics(
            lines_of_code=total_loc,
            cyclomatic_complexity=avg_complexity,
            maintainability_index=maintainability,
            code_duplication_pct=duplication_pct,
            test_coverage_pct=80.0,  # Placeholder - should integrate with actual coverage
            tech_debt_ratio=debt_ratio
        )
    
    def generate_debt_report(self, debt_items: List[TechDebtItem], metrics: CodeQualityMetrics):
        """Generate comprehensive technical debt report."""
        timestamp = datetime.now().isoformat()
        
        report = {
            'timestamp': timestamp,
            'metrics': asdict(metrics),
            'debt_items': [asdict(item) for item in debt_items],
            'summary': {
                'total_items': len(debt_items),
                'total_hours': sum(item.estimated_hours for item in debt_items),
                'by_severity': self._group_by_severity(debt_items),
                'by_category': self._group_by_category(debt_items),
                'top_priorities': sorted(debt_items, key=lambda x: x.priority, reverse=True)[:10]
            }
        }
        
        # Save to registry
        with open(self.debt_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        # Generate HTML report
        self._generate_html_report(report)
        
        print(f"📊 Technical debt report generated: {self.debt_file}")
        print(f"📈 Total debt items: {len(debt_items)}")
        print(f"⏱️  Estimated hours: {sum(item.estimated_hours for item in debt_items):.1f}")
        print(f"🔧 Maintainability index: {metrics.maintainability_index:.1f}")
        
    def _should_exclude_file(self, file_path: Path) -> bool:
        """Check if file should be excluded from analysis."""
        config = self.load_config()
        exclusions = config.get('exclusions', [])
        
        for pattern in exclusions:
            if file_path.match(pattern):
                return True
        return False
    
    def _calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity for a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.With, ast.AsyncWith):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
                
        return complexity
    
    def _get_function_length(self, node: ast.FunctionDef, content: str) -> int:
        """Get the number of lines in a function."""
        lines = content.split('\n')
        if hasattr(node, 'end_lineno') and node.end_lineno:
            return node.end_lineno - node.lineno + 1
        else:
            # Fallback: count non-empty lines
            func_lines = lines[node.lineno-1:]
            count = 0
            for line in func_lines:
                if line.strip():
                    count += 1
                elif count > 0 and not line.strip():
                    break
            return count
    
    def _get_class_length(self, node: ast.ClassDef, content: str) -> int:
        """Get the number of lines in a class."""
        lines = content.split('\n')
        if hasattr(node, 'end_lineno') and node.end_lineno:
            return node.end_lineno - node.lineno + 1
        else:
            return len([line for line in lines[node.lineno-1:] if line.strip()])
    
    def _get_function_hash(self, node: ast.FunctionDef) -> str:
        """Generate a hash for function structure comparison."""
        # Simplified structural hash
        structure = {
            'args': len(node.args.args),
            'returns': bool(node.returns),
            'decorators': len(node.decorator_list),
            'body_types': [type(stmt).__name__ for stmt in node.body[:5]]  # First 5 statements
        }
        return str(hash(str(structure)))
    
    def _count_lines_of_code(self) -> int:
        """Count total lines of code in the project."""
        total_lines = 0
        for py_file in self.project_root.rglob("*.py"):
            if self._should_exclude_file(py_file):
                continue
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    total_lines += len([line for line in f if line.strip()])
            except Exception:
                continue
        return total_lines
    
    def _group_by_severity(self, debt_items: List[TechDebtItem]) -> Dict[str, int]:
        """Group debt items by severity."""
        groups = {}
        for item in debt_items:
            groups[item.severity] = groups.get(item.severity, 0) + 1
        return groups
    
    def _group_by_category(self, debt_items: List[TechDebtItem]) -> Dict[str, int]:
        """Group debt items by category."""
        groups = {}
        for item in debt_items:
            groups[item.category] = groups.get(item.category, 0) + 1
        return groups
    
    def _generate_html_report(self, report: Dict[str, Any]):
        """Generate HTML technical debt report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Technical Debt Report - {report['timestamp'][:10]}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ margin: 10px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .critical {{ background-color: #ffebee; border-color: #f44336; }}
                .high {{ background-color: #fff3e0; border-color: #ff9800; }}
                .medium {{ background-color: #f3e5f5; border-color: #9c27b0; }}
                .low {{ background-color: #e8f5e8; border-color: #4caf50; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .summary {{ background-color: #f5f5f5; padding: 20px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>🔧 Technical Debt Report</h1>
            <div class="summary">
                <h2>📊 Summary</h2>
                <p><strong>Total Items:</strong> {report['summary']['total_items']}</p>
                <p><strong>Estimated Hours:</strong> {report['summary']['total_hours']:.1f}</p>
                <p><strong>Lines of Code:</strong> {report['metrics']['lines_of_code']:,}</p>
                <p><strong>Maintainability Index:</strong> {report['metrics']['maintainability_index']:.1f}</p>
                <p><strong>Tech Debt Ratio:</strong> {report['metrics']['tech_debt_ratio']:.2f} hours/KLOC</p>
            </div>
            
            <h2>🚨 Top Priority Items</h2>
            <table>
                <tr><th>Type</th><th>Severity</th><th>Location</th><th>Description</th><th>Hours</th></tr>
        """
        
        for item in report['summary']['top_priorities']:
            if isinstance(item, dict):
                html_content += f"""
                    <tr class="{item['severity']}">
                        <td>{item['type']}</td>
                        <td>{item['severity'].title()}</td>
                        <td>{item['location']}</td>
                        <td>{item['description']}</td>
                        <td>{item['estimated_hours']:.1f}</td>
                    </tr>
                """
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        html_file = self.project_root / "tech_debt_report.html"
        with open(html_file, 'w') as f:
            f.write(html_content)

def main():
    """Main technical debt analysis execution."""
    analyzer = TechnicalDebtAnalyzer()
    
    print("🔍 Analyzing technical debt...")
    
    # Collect all debt items
    all_debt_items = []
    all_debt_items.extend(analyzer.analyze_code_complexity())
    all_debt_items.extend(analyzer.detect_code_duplication())
    all_debt_items.extend(analyzer.analyze_test_coverage_gaps())
    all_debt_items.extend(analyzer.analyze_security_issues())
    
    # Calculate metrics
    metrics = analyzer.calculate_debt_metrics(all_debt_items)
    
    # Generate report
    analyzer.generate_debt_report(all_debt_items, metrics)
    
    # Provide recommendations
    print("\n=== 🎯 Recommendations ===")
    high_priority = [item for item in all_debt_items if item.severity == 'high']
    if high_priority:
        print(f"🚨 Address {len(high_priority)} high-priority items first")
        
    security_items = [item for item in all_debt_items if item.category == 'security']
    if security_items:
        print(f"🔒 {len(security_items)} security-related items need attention")
        
    if metrics.maintainability_index < 70:
        print("📉 Maintainability index is below recommended threshold (70)")
        
    if metrics.tech_debt_ratio > 5:
        print("⚠️  Technical debt ratio is high - consider refactoring sprint")

if __name__ == "__main__":
    main()