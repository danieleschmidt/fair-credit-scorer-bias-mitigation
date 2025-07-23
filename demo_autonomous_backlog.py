#!/usr/bin/env python3
"""
Demonstration of the Autonomous Backlog Management System

This script demonstrates the core functionality of the autonomous backlog
management system without requiring external dependencies that might not
be installed.
"""

import datetime
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class TaskType(Enum):
    """Types of backlog tasks"""
    FEATURE = "Feature"
    BUG = "Bug" 
    REFACTOR = "Refactor"
    SECURITY = "Security"
    DOC = "Doc"
    TEST = "Test"
    TECH_DEBT = "Tech_Debt"


class TaskStatus(Enum):
    """Task lifecycle states"""
    NEW = "NEW"
    REFINED = "REFINED"
    READY = "READY"
    DOING = "DOING"
    PR = "PR"
    BLOCKED = "BLOCKED"
    MERGED = "MERGED"
    DONE = "DONE"


@dataclass
class BacklogItem:
    """Simplified backlog item for demonstration"""
    id: str
    title: str
    description: str
    task_type: TaskType
    
    # WSJF Components (1,2,3,5,8,13 scale)
    business_value: int = 1
    time_criticality: int = 1
    risk_reduction: int = 1
    effort: int = 1
    
    # Metadata
    status: TaskStatus = TaskStatus.NEW
    acceptance_criteria: List[str] = field(default_factory=list)
    created_date: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    # Computed fields
    cost_of_delay: float = field(init=False)
    wsjf_score: float = field(init=False)
    final_score: float = field(init=False)
    
    def __post_init__(self):
        """Calculate derived scoring fields"""
        self.cost_of_delay = self.business_value + self.time_criticality + self.risk_reduction
        self.wsjf_score = self.cost_of_delay / max(self.effort, 1)
        
        # Simple aging multiplier (would be more sophisticated in real system)
        days_old = (datetime.datetime.now() - self.created_date).days
        aging_multiplier = min(1 + (days_old / 30), 2.0)
        
        self.final_score = self.wsjf_score * aging_multiplier
    
    def is_ready(self) -> bool:
        """Check if task is ready for execution"""
        return (self.status == TaskStatus.READY and 
                len(self.acceptance_criteria) > 0)


def create_sample_backlog() -> List[BacklogItem]:
    """Create sample backlog items for demonstration"""
    
    items = [
        BacklogItem(
            id="security_fix_1",
            title="Fix SQL injection vulnerability in user auth",
            description="Sanitize user inputs in authentication module to prevent SQL injection attacks",
            task_type=TaskType.SECURITY,
            business_value=13,
            time_criticality=13,
            risk_reduction=13,
            effort=3,
            status=TaskStatus.READY,
            acceptance_criteria=[
                "All user inputs are parameterized",
                "SQL injection tests pass",
                "Security scan shows no vulnerabilities",
                "Code review completed"
            ]
        ),
        
        BacklogItem(
            id="performance_bug",
            title="Fix memory leak in data processing",
            description="Memory usage grows unbounded during large dataset processing",
            task_type=TaskType.BUG,
            business_value=8,
            time_criticality=8,
            risk_reduction=8,
            effort=5,
            status=TaskStatus.READY,
            acceptance_criteria=[
                "Memory usage stabilizes during processing",
                "Processing completes without OOM errors",
                "Performance tests pass",
                "Memory profiling shows no leaks"
            ]
        ),
        
        BacklogItem(
            id="user_dashboard",
            title="Create user analytics dashboard", 
            description="Build responsive dashboard showing user activity metrics and insights",
            task_type=TaskType.FEATURE,
            business_value=8,
            time_criticality=5,
            risk_reduction=2,
            effort=8,
            status=TaskStatus.REFINED,
            acceptance_criteria=[]  # Not ready - missing acceptance criteria
        ),
        
        BacklogItem(
            id="test_coverage",
            title="Increase test coverage to 90%",
            description="Add unit and integration tests to achieve 90% code coverage",
            task_type=TaskType.TEST,
            business_value=6,
            time_criticality=4,
            risk_reduction=7,
            effort=6,
            status=TaskStatus.READY,
            acceptance_criteria=[
                "Coverage reaches 90% or higher",
                "All critical paths are tested",
                "Tests run in under 2 minutes",
                "No flaky tests in test suite"
            ]
        ),
        
        BacklogItem(
            id="api_docs",
            title="Update API documentation",
            description="Comprehensive update of API documentation with examples",
            task_type=TaskType.DOC,
            business_value=3,
            time_criticality=2,
            risk_reduction=2,
            effort=4,
            status=TaskStatus.NEW,
            acceptance_criteria=[]
        ),
        
        BacklogItem(
            id="blocked_feature",
            title="Advanced ML model integration",
            description="Integrate new machine learning model for improved predictions",
            task_type=TaskType.FEATURE,
            business_value=13,
            time_criticality=3,
            risk_reduction=5,
            effort=13,
            status=TaskStatus.BLOCKED,  # Would be blocked
            acceptance_criteria=[
                "ML model is trained and validated",
                "Integration tests pass",
                "Performance meets benchmarks"
            ]
        )
    ]
    
    return items


def demonstrate_wsjf_scoring(backlog: List[BacklogItem]):
    """Demonstrate WSJF scoring and ranking"""
    print("🎯 WSJF SCORING DEMONSTRATION")
    print("=" * 60)
    
    # Sort by final score
    sorted_items = sorted(backlog, key=lambda x: x.final_score, reverse=True)
    
    print(f"{'Rank':<4} {'ID':<15} {'Type':<8} {'WSJF':<6} {'Final':<6} {'Status'}")
    print("-" * 60)
    
    for i, item in enumerate(sorted_items, 1):
        print(f"{i:<4} {item.id:<15} {item.task_type.value:<8} "
              f"{item.wsjf_score:<6.2f} {item.final_score:<6.2f} {item.status.value}")
    
    print("\n📊 SCORING BREAKDOWN")
    print("=" * 60)
    
    for item in sorted_items[:3]:  # Show top 3
        print(f"\n🏆 {item.title}")
        print(f"   Business Value: {item.business_value}")
        print(f"   Time Criticality: {item.time_criticality}")
        print(f"   Risk Reduction: {item.risk_reduction}")
        print(f"   Effort: {item.effort}")
        print(f"   → Cost of Delay: {item.cost_of_delay}")
        print(f"   → WSJF Score: {item.wsjf_score:.2f}")
        print(f"   → Final Score: {item.final_score:.2f}")


def demonstrate_task_discovery():
    """Demonstrate task discovery from code comments"""
    print("\n🔍 TASK DISCOVERY DEMONSTRATION")
    print("=" * 60)
    
    # Simulate discovered tasks from code comments
    discovered_tasks = [
        {
            'file': 'src/auth.py',
            'line': 42,
            'type': 'TODO',
            'content': 'TODO: Implement rate limiting for login attempts',
            'priority': 8
        },
        {
            'file': 'src/data_processor.py', 
            'line': 156,
            'type': 'FIXME',
            'content': 'FIXME: Handle edge case when dataset is empty',
            'priority': 5
        },
        {
            'file': 'src/utils.py',
            'line': 23,
            'type': 'SECURITY',
            'content': 'SECURITY: Validate all user inputs before processing',
            'priority': 13
        },
        {
            'file': 'tests/test_api.py',
            'line': 89,
            'type': 'NOTE',
            'content': 'NOTE: Add integration test for new endpoint',
            'priority': 3
        }
    ]
    
    print("Discovered tasks from code comments:")
    print()
    
    for task in discovered_tasks:
        print(f"📁 {task['file']}:{task['line']}")
        print(f"   {task['type']} (Priority: {task['priority']})")
        print(f"   {task['content']}")
        print()


def demonstrate_execution_cycle(backlog: List[BacklogItem]):
    """Demonstrate the execution cycle"""
    print("\n⚡ EXECUTION CYCLE DEMONSTRATION")
    print("=" * 60)
    
    # Get ready items
    ready_items = [item for item in backlog if item.is_ready()]
    
    print(f"Ready items found: {len(ready_items)}")
    print()
    
    if ready_items:
        # Sort by priority
        ready_items.sort(key=lambda x: x.final_score, reverse=True)
        next_item = ready_items[0]
        
        print(f"🎯 Next item to execute: {next_item.title}")
        print(f"   ID: {next_item.id}")
        print(f"   Type: {next_item.task_type.value}")
        print(f"   Priority Score: {next_item.final_score:.2f}")
        print(f"   Acceptance Criteria: {len(next_item.acceptance_criteria)} items")
        print()
        
        print("🔄 TDD Micro-Cycle Phases:")
        phases = [
            "1. RED: Write failing test",
            "2. GREEN: Implement minimal code", 
            "3. REFACTOR: Improve design",
            "4. SECURITY: Apply security checks",
            "5. CI: Run full pipeline",
            "6. DOCS: Update documentation"
        ]
        
        for phase in phases:
            print(f"   {phase}")
        
        # Simulate status transition
        next_item.status = TaskStatus.DOING
        print(f"\n✅ Item status updated to: {next_item.status.value}")
    
    else:
        print("❌ No ready items found. Items need:")
        print("   - Status = READY")
        print("   - Acceptance criteria defined")
        print("   - No blocking issues")


def demonstrate_quality_gates():
    """Demonstrate quality gate checking"""
    print("\n🔐 QUALITY GATES DEMONSTRATION")
    print("=" * 60)
    
    # Simulate quality gate results
    gates = [
        {
            'name': 'Security Gate',
            'score': 95.0,
            'passed': True,
            'findings': [
                'No critical security issues found',
                'All inputs properly validated',
                'No secrets detected in code'
            ]
        },
        {
            'name': 'Testing Gate',
            'score': 88.0,
            'passed': True,
            'findings': [
                'Test coverage: 88% (target: 85%)',
                'All tests passing',
                'No flaky tests detected'
            ]
        },
        {
            'name': 'Documentation Gate',
            'score': 76.0,
            'passed': True,
            'findings': [
                'API documentation coverage: 76%',
                'Most public functions documented',
                '3 functions missing docstrings'
            ]
        }
    ]
    
    overall_score = sum(gate['score'] for gate in gates) / len(gates)
    all_passed = all(gate['passed'] for gate in gates)
    
    print(f"Overall Quality Score: {overall_score:.1f}")
    print(f"All Gates Passed: {'✅ YES' if all_passed else '❌ NO'}")
    print()
    
    for gate in gates:
        status = "✅ PASS" if gate['passed'] else "❌ FAIL"
        print(f"{gate['name']}: {gate['score']:.1f}% {status}")
        for finding in gate['findings']:
            print(f"   • {finding}")
        print()


def demonstrate_metrics_reporting(backlog: List[BacklogItem]):
    """Demonstrate metrics and reporting"""
    print("\n📊 METRICS & REPORTING DEMONSTRATION")
    print("=" * 60)
    
    # Calculate backlog metrics
    total_items = len(backlog)
    by_status = {}
    by_type = {}
    
    for item in backlog:
        status = item.status.value
        task_type = item.task_type.value
        
        by_status[status] = by_status.get(status, 0) + 1
        by_type[task_type] = by_type.get(task_type, 0) + 1
    
    ready_count = by_status.get('READY', 0)
    blocked_count = by_status.get('BLOCKED', 0)
    
    # WSJF distribution
    high_priority = len([item for item in backlog if item.final_score > 5])
    medium_priority = len([item for item in backlog if 2 <= item.final_score <= 5])
    low_priority = len([item for item in backlog if item.final_score < 2])
    
    print("📈 BACKLOG HEALTH METRICS")
    print(f"   Total Items: {total_items}")
    print(f"   Ready to Work: {ready_count}")
    print(f"   Blocked: {blocked_count}")
    print(f"   Health Score: {((ready_count / max(total_items, 1)) * 100):.1f}%")
    print()
    
    print("🎲 PRIORITY DISTRIBUTION")
    print(f"   High Priority (>5): {high_priority}")
    print(f"   Medium Priority (2-5): {medium_priority}")
    print(f"   Low Priority (<2): {low_priority}")
    print()
    
    print("📋 STATUS DISTRIBUTION")
    for status, count in by_status.items():
        print(f"   {status}: {count}")
    print()
    
    print("🏷️ TYPE DISTRIBUTION")
    for task_type, count in by_type.items():
        print(f"   {task_type}: {count}")


def generate_status_dashboard(backlog: List[BacklogItem]) -> str:
    """Generate a text-based status dashboard"""
    ready_count = len([item for item in backlog if item.is_ready()])
    blocked_count = len([item for item in backlog if item.status == TaskStatus.BLOCKED])
    total_items = len(backlog)
    
    dashboard = f"""
🎯 AUTONOMOUS BACKLOG STATUS DASHBOARD
{'=' * 50}

📊 BACKLOG OVERVIEW
  Total Items: {total_items}
  Ready to Work: {ready_count}
  Blocked: {blocked_count}
  Health Score: {((ready_count / max(total_items, 1)) * 100):.1f}%

⚡ NEXT ACTIONS
  • {'Execute ready items' if ready_count > 0 else 'Refine items to ready state'}
  • {'Unblock items' if blocked_count > 0 else 'Continue execution'}
  • {'Discover new tasks' if total_items < 10 else 'Focus on execution'}

🏆 TOP PRIORITY ITEMS
"""
    
    # Add top 3 items
    sorted_items = sorted(backlog, key=lambda x: x.final_score, reverse=True)
    for i, item in enumerate(sorted_items[:3], 1):
        status_emoji = "🚀" if item.is_ready() else "📝" if item.status == TaskStatus.REFINED else "🆕"
        dashboard += f"  {i}. {status_emoji} {item.title} (Score: {item.final_score:.1f})\n"
    
    dashboard += f"\nLast Updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    
    return dashboard


def main():
    """Main demonstration function"""
    print("🚀 AUTONOMOUS BACKLOG MANAGEMENT SYSTEM DEMO")
    print("=" * 80)
    print()
    print("This demonstration shows the core functionality of the autonomous")
    print("backlog management system without requiring external dependencies.")
    print()
    
    # Create sample backlog
    backlog = create_sample_backlog()
    
    # Demonstrate different aspects
    demonstrate_wsjf_scoring(backlog)
    demonstrate_task_discovery()
    demonstrate_execution_cycle(backlog)
    demonstrate_quality_gates()
    demonstrate_metrics_reporting(backlog)
    
    # Generate status dashboard
    print("\n📋 GENERATED STATUS DASHBOARD")
    print("=" * 60)
    dashboard = generate_status_dashboard(backlog)
    print(dashboard)
    
    print("\n🎉 DEMONSTRATION COMPLETE!")
    print("=" * 60)
    print("This shows how the autonomous system:")
    print("✅ Scores and ranks backlog items using WSJF methodology")
    print("✅ Discovers new tasks from code comments and other sources")
    print("✅ Executes items using TDD micro-cycles")
    print("✅ Applies quality gates and security checks")
    print("✅ Generates comprehensive metrics and reports")
    print()
    print("The full system includes:")
    print("• Continuous execution loops")
    print("• Integration with CI/CD pipelines")
    print("• Advanced trend analysis and predictions")
    print("• Visualization charts and dashboards")
    print("• Configuration management and customization")
    print()
    print("See DOCS/AUTONOMOUS_BACKLOG_SYSTEM.md for complete documentation.")


if __name__ == "__main__":
    main()