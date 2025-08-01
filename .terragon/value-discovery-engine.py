#!/usr/bin/env python3
"""
Autonomous Value Discovery Engine for Advanced SDLC Repositories

Implements continuous value discovery using WSJF + ICE + Technical Debt scoring
with adaptive prioritization and autonomous execution capabilities.
"""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
import re
import ast


class ValueDiscoveryEngine:
    """Advanced value discovery and prioritization system"""
    
    def __init__(self, repo_path: Path = Path(".")):
        self.repo_path = repo_path
        self.config_path = repo_path / ".terragon" / "config.yaml"
        self.metrics_path = repo_path / ".terragon" / "value-metrics.json"
        self.load_config()
        
    def load_config(self):
        """Load configuration from terragon config file"""
        if self.config_path.exists():
            with open(self.config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()
    
    def _default_config(self) -> Dict:
        """Default configuration for advanced repositories"""
        return {
            "scoring": {
                "weights": {"wsjf": 0.5, "ice": 0.1, "technicalDebt": 0.3, "security": 0.1},
                "thresholds": {"minScore": 15, "maxRisk": 0.7}
            },
            "repository": {"maturity": "advanced"}
        }
    
    def discover_value_opportunities(self) -> List[Dict]:
        """Comprehensive signal harvesting from multiple sources"""
        opportunities = []
        
        # Git history analysis
        opportunities.extend(self._analyze_git_history())
        
        # Static analysis integration
        opportunities.extend(self._run_static_analysis())
        
        # Test and coverage analysis
        opportunities.extend(self._analyze_test_gaps())
        
        # Performance monitoring
        opportunities.extend(self._analyze_performance_metrics())
        
        # Security scanning results
        opportunities.extend(self._analyze_security_posture())
        
        # Documentation and compliance gaps
        opportunities.extend(self._analyze_documentation_gaps())
        
        return opportunities
    
    def _analyze_git_history(self) -> List[Dict]:
        """Extract value signals from Git history"""
        opportunities = []
        
        try:
            # Find TODO/FIXME/HACK comments
            result = subprocess.run(
                ["git", "grep", "-n", "-i", "todo\\|fixme\\|hack\\|xxx"],
                cwd=self.repo_path, capture_output=True, text=True
            )
            
            if result.returncode == 0:
                for line in result.stdout.split('\n')[:10]:  # Limit to prevent spam
                    if line.strip():
                        parts = line.split(':', 2)
                        if len(parts) >= 3:
                            opportunities.append({
                                "title": f"Address technical debt comment in {parts[0]}",
                                "type": "technical-debt",
                                "source": "git-grep",
                                "file": parts[0],
                                "line": parts[1],
                                "description": parts[2].strip(),
                                "effort_hours": 2,
                                "priority": "medium"
                            })
        except Exception:
            pass
        
        return opportunities
    
    def _run_static_analysis(self) -> List[Dict]:
        """Run static analysis tools and extract improvement opportunities"""
        opportunities = []
        
        # Run ruff for code style issues
        try:
            result = subprocess.run(
                ["ruff", "check", "--output-format=json", "."],
                cwd=self.repo_path, capture_output=True, text=True
            )
            
            if result.stdout:
                issues = json.loads(result.stdout)
                issue_counts = {}
                
                for issue in issues[:20]:  # Limit to prevent spam
                    rule = issue.get("code", "unknown")
                    if rule not in issue_counts:
                        issue_counts[rule] = 0
                    issue_counts[rule] += 1
                
                # Create opportunities for frequent issues
                for rule, count in issue_counts.items():
                    if count >= 3:
                        opportunities.append({
                            "title": f"Fix {count} instances of {rule} code style issues",
                            "type": "code-quality",
                            "source": "ruff",
                            "description": f"Multiple {rule} violations found",
                            "effort_hours": min(count * 0.5, 8),
                            "priority": "low" if count < 5 else "medium"
                        })
        except Exception:
            pass
        
        return opportunities
    
    def _analyze_test_gaps(self) -> List[Dict]:
        """Analyze test coverage and identify testing opportunities"""
        opportunities = []
        
        # Check for files with low coverage
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "--cov=src", "--cov-report=json", "--tb=no", "-q"],
                cwd=self.repo_path, capture_output=True, text=True
            )
            
            cov_path = self.repo_path / "coverage.json"
            if cov_path.exists():
                with open(cov_path) as f:
                    coverage_data = json.load(f)
                
                for filename, file_data in coverage_data.get("files", {}).items():
                    coverage_pct = file_data.get("summary", {}).get("percent_covered", 100)
                    missing_lines = len(file_data.get("missing_lines", []))
                    
                    if coverage_pct < 80 and missing_lines > 10:
                        opportunities.append({
                            "title": f"Improve test coverage for {Path(filename).name}",
                            "type": "testing",
                            "source": "coverage-analysis",
                            "file": filename,
                            "description": f"Coverage: {coverage_pct:.1f}%, {missing_lines} missing lines",
                            "effort_hours": min(missing_lines * 0.1, 6),
                            "priority": "medium" if coverage_pct < 60 else "low"
                        })
        except Exception:
            pass
        
        return opportunities
    
    def _analyze_performance_metrics(self) -> List[Dict]:
        """Analyze performance data for optimization opportunities"""
        opportunities = []
        
        # Look for performance benchmark results
        perf_files = list(self.repo_path.glob("**/*performance*.json"))
        perf_files.extend(list(self.repo_path.glob("**/*benchmark*.json")))
        
        for perf_file in perf_files[:5]:  # Limit to prevent spam
            try:
                with open(perf_file) as f:
                    perf_data = json.load(f)
                
                # Look for slow operations (>1 second)
                if isinstance(perf_data, dict):
                    for key, value in perf_data.items():
                        if isinstance(value, (int, float)) and value > 1.0:
                            opportunities.append({
                                "title": f"Optimize slow operation: {key}",
                                "type": "performance",
                                "source": "performance-analysis",
                                "description": f"Operation takes {value:.2f} seconds",
                                "effort_hours": 4,
                                "priority": "medium" if value > 5.0 else "low"
                            })
            except Exception:
                continue
        
        return opportunities
    
    def _analyze_security_posture(self) -> List[Dict]:
        """Analyze security scanning results"""
        opportunities = []
        
        # Run bandit security scan
        try:
            result = subprocess.run(
                ["bandit", "-r", "src/", "-f", "json"],
                cwd=self.repo_path, capture_output=True, text=True
            )
            
            if result.stdout:
                security_data = json.loads(result.stdout)
                issues = security_data.get("results", [])
                
                high_severity = [i for i in issues if i.get("issue_severity") == "HIGH"]
                medium_severity = [i for i in issues if i.get("issue_severity") == "MEDIUM"]
                
                if high_severity:
                    opportunities.append({
                        "title": f"Fix {len(high_severity)} high-severity security issues",
                        "type": "security",
                        "source": "bandit",
                        "description": "Critical security vulnerabilities found",
                        "effort_hours": len(high_severity) * 2,
                        "priority": "high"
                    })
                
                if medium_severity:
                    opportunities.append({
                        "title": f"Address {len(medium_severity)} medium-severity security issues",
                        "type": "security",
                        "source": "bandit",
                        "description": "Moderate security issues found",
                        "effort_hours": len(medium_severity) * 1,
                        "priority": "medium"
                    })
        except Exception:
            pass
        
        return opportunities
    
    def _analyze_documentation_gaps(self) -> List[Dict]:
        """Identify documentation and compliance gaps"""
        opportunities = []
        
        # Check for missing docstrings
        python_files = list(self.repo_path.glob("src/**/*.py"))
        missing_docstrings = 0
        
        for py_file in python_files:
            try:
                with open(py_file) as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                        if not ast.get_docstring(node):
                            missing_docstrings += 1
            except Exception:
                continue
        
        if missing_docstrings > 5:
            opportunities.append({
                "title": f"Add docstrings to {missing_docstrings} functions/classes",
                "type": "documentation",
                "source": "docstring-analysis",
                "description": "Improve code documentation",
                "effort_hours": missing_docstrings * 0.25,
                "priority": "low"
            })
        
        return opportunities
    
    def calculate_composite_score(self, opportunity: Dict) -> float:
        """Calculate WSJF + ICE + Technical Debt composite score"""
        
        # WSJF Components
        business_value = self._score_business_value(opportunity)
        time_criticality = self._score_time_criticality(opportunity)
        risk_reduction = self._score_risk_reduction(opportunity)
        job_size = opportunity.get("effort_hours", 4)
        
        wsjf = (business_value + time_criticality + risk_reduction) / max(job_size, 0.5)
        
        # ICE Components  
        impact = self._score_impact(opportunity)
        confidence = self._score_confidence(opportunity)
        ease = 10 - min(job_size, 10)  # Inverse of effort
        
        ice = impact * confidence * ease
        
        # Technical Debt Score
        debt_score = self._score_technical_debt(opportunity)
        
        # Apply weights
        weights = self.config["scoring"]["weights"]
        composite = (
            weights.get("wsjf", 0.5) * self._normalize_score(wsjf, 0, 30) +
            weights.get("ice", 0.1) * self._normalize_score(ice, 0, 1000) +
            weights.get("technicalDebt", 0.3) * self._normalize_score(debt_score, 0, 100) +
            weights.get("security", 0.1) * (2.0 if opportunity.get("type") == "security" else 1.0)
        ) * 100
        
        return round(composite, 2)
    
    def _score_business_value(self, opp: Dict) -> float:
        """Score business value impact (1-10)"""
        type_scores = {
            "security": 9,
            "performance": 7,
            "testing": 6,
            "technical-debt": 5,
            "code-quality": 4,
            "documentation": 3
        }
        return type_scores.get(opp.get("type", ""), 5)
    
    def _score_time_criticality(self, opp: Dict) -> float:
        """Score time criticality (1-10)"""
        priority_scores = {"high": 8, "medium": 5, "low": 2}
        return priority_scores.get(opp.get("priority", "medium"), 5)
    
    def _score_risk_reduction(self, opp: Dict) -> float:
        """Score risk reduction value (1-10)"""
        type_scores = {
            "security": 9,
            "testing": 7,
            "performance": 6,
            "technical-debt": 5,
            "code-quality": 3,
            "documentation": 2
        }
        return type_scores.get(opp.get("type", ""), 4)
    
    def _score_impact(self, opp: Dict) -> float:
        """Score overall impact (1-10)"""
        return self._score_business_value(opp)
    
    def _score_confidence(self, opp: Dict) -> float:
        """Score execution confidence (1-10)"""
        effort = opp.get("effort_hours", 4)
        if effort <= 2:
            return 9
        elif effort <= 4:
            return 7
        elif effort <= 8:
            return 5
        else:
            return 3
    
    def _score_technical_debt(self, opp: Dict) -> float:
        """Score technical debt impact (0-100)"""
        if opp.get("type") == "technical-debt":
            return 80
        elif opp.get("type") == "security":
            return 90
        elif opp.get("type") == "testing":
            return 60
        elif opp.get("type") == "performance":
            return 70
        else:
            return 30
    
    def _normalize_score(self, score: float, min_val: float, max_val: float) -> float:
        """Normalize score to 0-1 range"""
        return max(0, min(1, (score - min_val) / (max_val - min_val)))
    
    def prioritize_opportunities(self, opportunities: List[Dict]) -> List[Dict]:
        """Score and prioritize all opportunities"""
        
        for opp in opportunities:
            opp["composite_score"] = self.calculate_composite_score(opp)
            opp["timestamp"] = datetime.now().isoformat()
        
        # Sort by composite score descending
        return sorted(opportunities, key=lambda x: x["composite_score"], reverse=True)
    
    def generate_backlog(self, opportunities: List[Dict]) -> str:
        """Generate markdown backlog with prioritized opportunities"""
        
        backlog = f"""# 🎯 Autonomous Value Discovery Backlog

Last Updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Repository Maturity: {self.config["repository"]["maturity"].title()}

## 📊 Discovery Summary

- **Total Opportunities Found**: {len(opportunities)}
- **High Priority (Score ≥ 70)**: {len([o for o in opportunities if o["composite_score"] >= 70])}
- **Medium Priority (Score 40-69)**: {len([o for o in opportunities if 40 <= o["composite_score"] < 70])}
- **Low Priority (Score < 40)**: {len([o for o in opportunities if o["composite_score"] < 40])}

## 🚀 Next Best Value Item

"""
        
        if opportunities:
            top = opportunities[0]
            backlog += f"""**{top["title"]}**
- **Composite Score**: {top["composite_score"]}
- **Type**: {top["type"].title()}
- **Source**: {top["source"]}
- **Estimated Effort**: {top["effort_hours"]} hours
- **Description**: {top["description"]}

"""
        
        backlog += "## 📋 Top Value Opportunities\n\n"
        backlog += "| Rank | Score | Type | Title | Effort (hrs) | Priority |\n"
        backlog += "|------|-------|------|-------|--------------|----------|\n"
        
        for i, opp in enumerate(opportunities[:15], 1):
            backlog += f"| {i} | {opp['composite_score']} | {opp['type']} | {opp['title'][:50]}... | {opp['effort_hours']} | {opp['priority']} |\n"
        
        backlog += f"\n## 🔍 Discovery Sources\n\n"
        
        sources = {}
        for opp in opportunities:
            source = opp.get("source", "unknown")
            sources[source] = sources.get(source, 0) + 1
        
        for source, count in sources.items():
            pct = (count / len(opportunities)) * 100 if opportunities else 0
            backlog += f"- **{source}**: {count} items ({pct:.1f}%)\n"
        
        backlog += f"\n---\n*Generated by Terragon Autonomous Value Discovery Engine*"
        
        return backlog
    
    def save_metrics(self, opportunities: List[Dict]):
        """Save value metrics to JSON for tracking"""
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "total_opportunities": len(opportunities),
            "opportunities": opportunities[:50],  # Save top 50
            "summary_stats": {
                "avg_score": sum(o["composite_score"] for o in opportunities) / len(opportunities) if opportunities else 0,
                "high_priority_count": len([o for o in opportunities if o["composite_score"] >= 70]),
                "total_effort_hours": sum(o["effort_hours"] for o in opportunities),
            },
            "discovery_sources": {
                source: len([o for o in opportunities if o.get("source") == source])
                for source in set(o.get("source", "unknown") for o in opportunities)
            }
        }
        
        self.metrics_path.parent.mkdir(exist_ok=True)
        with open(self.metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)


def main():
    """Run autonomous value discovery"""
    engine = ValueDiscoveryEngine()
    
    print("🔍 Discovering value opportunities...")
    opportunities = engine.discover_value_opportunities()
    
    print(f"📊 Found {len(opportunities)} opportunities")
    prioritized = engine.prioritize_opportunities(opportunities)
    
    print("📋 Generating backlog...")
    backlog = engine.generate_backlog(prioritized)
    
    # Save backlog
    backlog_path = Path("AUTONOMOUS_BACKLOG.md")
    with open(backlog_path, "w") as f:
        f.write(backlog)
    
    print(f"✅ Backlog saved to {backlog_path}")
    
    # Save metrics
    engine.save_metrics(prioritized)
    print(f"📈 Metrics saved to {engine.metrics_path}")
    
    if prioritized:
        print(f"\n🎯 Next best value item:")
        print(f"   {prioritized[0]['title']}")
        print(f"   Score: {prioritized[0]['composite_score']}")
        print(f"   Effort: {prioritized[0]['effort_hours']} hours")


if __name__ == "__main__":
    main()