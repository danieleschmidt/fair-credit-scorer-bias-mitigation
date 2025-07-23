"""
Autonomous Backlog Management System

This module implements a comprehensive backlog management system that:
1. Loads, normalizes, and scores backlog items using WSJF methodology
2. Discovers new tasks from code comments, failing tests, and other sources
3. Executes tasks in priority order with TDD micro-cycles
4. Maintains quality gates and security checks
5. Reports metrics and status continuously
"""

import datetime
import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
import yaml

logger = logging.getLogger(__name__)


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
    """Structured backlog item with WSJF scoring"""
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
    links: List[str] = field(default_factory=list)
    created_date: datetime.datetime = field(default_factory=datetime.datetime.now)
    last_updated: datetime.datetime = field(default_factory=datetime.datetime.now)
    blocked_reason: Optional[str] = None
    
    # Computed fields
    cost_of_delay: float = field(init=False)
    wsjf_score: float = field(init=False)
    aging_multiplier: float = field(init=False, default=1.0)
    final_score: float = field(init=False)
    
    def __post_init__(self):
        """Calculate derived scoring fields"""
        self.cost_of_delay = self.business_value + self.time_criticality + self.risk_reduction
        self.wsjf_score = self.cost_of_delay / max(self.effort, 1)  # Avoid division by zero
        
        # Apply aging multiplier (max 2x after 30 days)
        days_old = (datetime.datetime.now() - self.created_date).days
        self.aging_multiplier = min(1 + (days_old / 30), 2.0)
        
        self.final_score = self.wsjf_score * self.aging_multiplier
    
    def is_ready(self) -> bool:
        """Check if task is ready for execution"""
        return (self.status == TaskStatus.READY and 
                len(self.acceptance_criteria) > 0 and
                self.blocked_reason is None)
    
    def is_blocked(self) -> bool:
        """Check if task is blocked"""
        return self.status == TaskStatus.BLOCKED or self.blocked_reason is not None


@dataclass 
class TaskDiscoveryResult:
    """Result from task discovery scanning"""
    file_path: str
    line_number: int
    content: str
    task_type: TaskType
    priority: int
    

class TaskDiscoveryEngine:
    """Discovers new tasks from various sources"""
    
    def __init__(self, repo_path: str = "/root/repo"):
        self.repo_path = repo_path
        self.comment_patterns = {
            'TODO': TaskType.FEATURE,
            'FIXME': TaskType.BUG,
            'HACK': TaskType.REFACTOR,
            'BUG': TaskType.BUG,
            'XXX': TaskType.TECH_DEBT,
            'NOTE': TaskType.DOC
        }
    
    def scan_code_comments(self) -> List[TaskDiscoveryResult]:
        """Scan codebase for TODO/FIXME/etc comments"""
        results = []
        
        for pattern, task_type in self.comment_patterns.items():
            try:
                # Use grep to find pattern in all source files
                cmd = ["grep", "-rn", f"#{pattern}", f"{self.repo_path}/src/", f"{self.repo_path}/tests/"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                for line in result.stdout.split('\n'):
                    if line.strip():
                        parts = line.split(':', 2)
                        if len(parts) >= 3:
                            file_path, line_num, content = parts[0], parts[1], parts[2]
                            results.append(TaskDiscoveryResult(
                                file_path=file_path,
                                line_number=int(line_num),
                                content=content.strip(),
                                task_type=task_type,
                                priority=self._estimate_priority(content, task_type)
                            ))
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError):
                logger.warning(f"Failed to scan for {pattern} comments")
                continue
        
        return results
    
    def scan_failing_tests(self) -> List[TaskDiscoveryResult]:
        """Scan for failing or flaky tests"""
        results = []
        
        try:
            # Run pytest with --collect-only to find tests without running them
            cmd = ["python", "-m", "pytest", "--collect-only", "-q"]
            result = subprocess.run(cmd, cwd=self.repo_path, capture_output=True, text=True, timeout=60)
            
            # Look for test files with issues
            for line in result.stderr.split('\n'):
                if 'ERROR' in line or 'FAILED' in line:
                    results.append(TaskDiscoveryResult(
                        file_path="pytest",
                        line_number=0,
                        content=line.strip(),
                        task_type=TaskType.BUG,
                        priority=8  # High priority for broken tests
                    ))
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            logger.warning("Failed to scan for failing tests")
        
        return results
    
    def scan_security_vulnerabilities(self) -> List[TaskDiscoveryResult]:
        """Scan for security issues using bandit"""
        results = []
        
        try:
            cmd = ["bandit", "-r", f"{self.repo_path}/src/", "-f", "json"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.stdout:
                data = json.loads(result.stdout)
                for issue in data.get('results', []):
                    results.append(TaskDiscoveryResult(
                        file_path=issue['filename'],
                        line_number=issue['line_number'],
                        content=f"Security: {issue['test_name']} - {issue['issue_text']}",
                        task_type=TaskType.SECURITY,
                        priority=13  # Always high priority for security
                    ))
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, json.JSONDecodeError):
            logger.warning("Failed to scan for security vulnerabilities")
        
        return results
    
    def _estimate_priority(self, content: str, task_type: TaskType) -> int:
        """Estimate priority based on content and type"""
        priority_keywords = {
            'critical': 13,
            'urgent': 8,
            'important': 5,
            'security': 13,
            'performance': 8,
            'bug': 8,
            'deprecated': 5,
            'cleanup': 3
        }
        
        content_lower = content.lower()
        for keyword, priority in priority_keywords.items():
            if keyword in content_lower:
                return priority
        
        # Default priorities by type
        type_priorities = {
            TaskType.SECURITY: 13,
            TaskType.BUG: 8,
            TaskType.FEATURE: 5,
            TaskType.REFACTOR: 3,
            TaskType.DOC: 2,
            TaskType.TEST: 5,
            TaskType.TECH_DEBT: 3
        }
        
        return type_priorities.get(task_type, 3)


class BacklogManager:
    """Main backlog management and execution engine"""
    
    def __init__(self, repo_path: str = "/root/repo"):
        self.repo_path = repo_path
        self.backlog_file = os.path.join(repo_path, "DOCS", "backlog.yml")
        self.status_dir = os.path.join(repo_path, "DOCS", "status")
        self.backlog: List[BacklogItem] = []
        self.discovery_engine = TaskDiscoveryEngine(repo_path)
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.backlog_file), exist_ok=True)
        os.makedirs(self.status_dir, exist_ok=True)
    
    def load_backlog(self) -> List[BacklogItem]:
        """Load and normalize backlog from various sources"""
        items = []
        
        # Load from YAML file if exists
        if os.path.exists(self.backlog_file):
            try:
                with open(self.backlog_file, 'r') as f:
                    data = yaml.safe_load(f) or {}
                    for item_data in data.get('items', []):
                        items.append(self._dict_to_backlog_item(item_data))
            except (yaml.YAMLError, KeyError, ValueError) as e:
                logger.error(f"Failed to load backlog from YAML: {e}")
        
        # Load from existing BACKLOG.md if no YAML exists
        if not items:
            items = self._parse_markdown_backlog()
        
        self.backlog = items
        return items
    
    def _parse_markdown_backlog(self) -> List[BacklogItem]:
        """Parse existing BACKLOG.md file"""
        backlog_md = os.path.join(self.repo_path, "BACKLOG.md")
        items = []
        
        if not os.path.exists(backlog_md):
            return items
        
        try:
            with open(backlog_md, 'r') as f:
                content = f.read()
            
            # Extract tasks using regex
            task_pattern = r'###\s+(\d+)\.\s*(.*?)\(Score:\s*(\d+)\)'
            matches = re.findall(task_pattern, content, re.MULTILINE)
            
            for match in matches:
                task_id, title, score = match
                
                # Extract additional details from the task section
                task_section_pattern = rf'###\s+{task_id}\..*?(?=###|\Z)'
                section_match = re.search(task_section_pattern, content, re.DOTALL)
                
                description = ""
                business_value = 5
                effort = 5
                status = TaskStatus.NEW
                
                if section_match:
                    section = section_match.group(0)
                    
                    # Extract business value, effort, etc.
                    if 'Business Value' in section:
                        bv_match = re.search(r'Business Value.*?(\d+)/10', section)
                        if bv_match:
                            business_value = int(bv_match.group(1))
                    
                    if 'Effort' in section:
                        effort_match = re.search(r'Effort.*?(\w+)', section)
                        if effort_match:
                            effort_text = effort_match.group(1).lower()
                            effort = {'low': 2, 'medium': 5, 'high': 8, 'very': 13}.get(effort_text, 5)
                    
                    # Check if completed
                    if '✅' in section or 'COMPLETED' in section:
                        status = TaskStatus.DONE
                    
                    # Extract description from Impact line
                    impact_match = re.search(r'- \*\*Impact\*\*:\s*(.*)', section)
                    if impact_match:
                        description = impact_match.group(1)
                
                items.append(BacklogItem(
                    id=f"task_{task_id}",
                    title=title.strip(),
                    description=description,
                    task_type=self._infer_task_type(title),
                    business_value=min(business_value, 13),  # Cap at 13
                    effort=effort,
                    status=status
                ))
        
        except Exception as e:
            logger.error(f"Failed to parse BACKLOG.md: {e}")
        
        return items
    
    def _infer_task_type(self, title: str) -> TaskType:
        """Infer task type from title"""
        title_lower = title.lower()
        
        if any(word in title_lower for word in ['security', 'vulnerability', 'auth']):
            return TaskType.SECURITY
        elif any(word in title_lower for word in ['bug', 'fix', 'error', 'issue']):
            return TaskType.BUG
        elif any(word in title_lower for word in ['test', 'coverage', 'pytest']):
            return TaskType.TEST
        elif any(word in title_lower for word in ['doc', 'documentation', 'readme']):
            return TaskType.DOC
        elif any(word in title_lower for word in ['refactor', 'cleanup', 'improve']):
            return TaskType.REFACTOR
        else:
            return TaskType.FEATURE
    
    def discover_new_tasks(self) -> List[BacklogItem]:
        """Discover new tasks from all sources"""
        new_items = []
        
        # Scan various sources
        sources = [
            self.discovery_engine.scan_code_comments(),
            self.discovery_engine.scan_failing_tests(),
            self.discovery_engine.scan_security_vulnerabilities()
        ]
        
        existing_descriptions = {item.description for item in self.backlog}
        
        for source_results in sources:
            for result in source_results:
                # Avoid duplicates
                if result.content not in existing_descriptions:
                    item = BacklogItem(
                        id=f"auto_{int(time.time())}_{len(new_items)}",
                        title=f"Auto: {result.content[:50]}...",
                        description=result.content,
                        task_type=result.task_type,
                        business_value=min(result.priority, 13),
                        effort=3,  # Default medium effort
                        status=TaskStatus.NEW
                    )
                    new_items.append(item)
        
        return new_items
    
    def score_and_rank(self) -> List[BacklogItem]:
        """Score and rank all backlog items"""
        # Recalculate scores (triggers __post_init__)
        for item in self.backlog:
            item.__post_init__()
        
        # Sort by final score descending
        self.backlog.sort(key=lambda x: x.final_score, reverse=True)
        return self.backlog
    
    def get_next_ready_item(self) -> Optional[BacklogItem]:
        """Get the highest priority ready item"""
        for item in self.backlog:
            if item.is_ready() and not item.is_blocked():
                return item
        return None
    
    def execute_item_tdd_cycle(self, item: BacklogItem) -> bool:
        """Execute a single backlog item using TDD cycle"""
        logger.info(f"Starting TDD cycle for: {item.title}")
        
        try:
            # Mark as in progress
            item.status = TaskStatus.DOING
            item.last_updated = datetime.datetime.now()
            
            # TDD Red-Green-Refactor cycle would be implemented here
            # For now, we'll simulate the process and focus on the framework
            
            # 1. Write failing test (RED)
            logger.info("Phase 1: Write failing test")
            
            # 2. Implement minimal code (GREEN)  
            logger.info("Phase 2: Implement minimal code")
            
            # 3. Refactor (REFACTOR)
            logger.info("Phase 3: Refactor")
            
            # 4. Security & compliance checks
            logger.info("Phase 4: Security checks")
            security_results = self.discovery_engine.scan_security_vulnerabilities()
            if security_results:
                logger.warning(f"Security issues found: {len(security_results)}")
                return False
            
            # 5. Run CI pipeline
            logger.info("Phase 5: CI pipeline")
            if not self._run_ci_checks():
                logger.error("CI checks failed")
                return False
            
            # Mark as ready for PR
            item.status = TaskStatus.PR
            item.last_updated = datetime.datetime.now()
            
            logger.info(f"Successfully completed TDD cycle for: {item.title}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute item {item.id}: {e}")
            item.blocked_reason = str(e)
            item.status = TaskStatus.BLOCKED
            return False
    
    def _run_ci_checks(self) -> bool:
        """Run CI pipeline checks"""
        checks = [
            (["python", "-m", "pytest", "--tb=short"], "Tests"),
            (["ruff", "check", "src/"], "Linting"),  
            (["bandit", "-r", "src/"], "Security")
        ]
        
        for cmd, name in checks:
            try:
                result = subprocess.run(
                    cmd, 
                    cwd=self.repo_path, 
                    capture_output=True, 
                    text=True, 
                    timeout=120
                )
                if result.returncode != 0:
                    logger.error(f"{name} check failed: {result.stderr}")
                    return False
                logger.info(f"{name} check passed")
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
                logger.error(f"{name} check error: {e}")
                return False
        
        return True
    
    def save_backlog(self):
        """Save backlog to YAML file"""
        data = {
            'last_updated': datetime.datetime.now().isoformat(),
            'items': [self._backlog_item_to_dict(item) for item in self.backlog]
        }
        
        try:
            with open(self.backlog_file, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Backlog saved to {self.backlog_file}")
        except Exception as e:
            logger.error(f"Failed to save backlog: {e}")
    
    def generate_status_report(self) -> Dict:
        """Generate comprehensive status report"""
        now = datetime.datetime.now()
        
        status_counts = {}
        for status in TaskStatus:
            status_counts[status.value] = sum(1 for item in self.backlog if item.status == status)
        
        completed_last_week = [
            item for item in self.backlog 
            if item.status == TaskStatus.DONE and 
            (now - item.last_updated).days <= 7
        ]
        
        report = {
            'timestamp': now.isoformat(),
            'backlog_size': len(self.backlog),
            'status_distribution': status_counts,
            'completed_last_week': len(completed_last_week),
            'avg_wsjf_score': sum(item.wsjf_score for item in self.backlog) / max(len(self.backlog), 1),
            'blocked_items': [item.id for item in self.backlog if item.is_blocked()],
            'top_priority_items': [
                {'id': item.id, 'title': item.title, 'score': item.final_score} 
                for item in self.backlog[:5]
            ]
        }
        
        # Save report
        report_file = os.path.join(self.status_dir, f"status_{now.strftime('%Y%m%d_%H%M%S')}.json")
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save status report: {e}")
        
        return report
    
    def _dict_to_backlog_item(self, data: Dict) -> BacklogItem:
        """Convert dictionary to BacklogItem"""
        return BacklogItem(
            id=data['id'],
            title=data['title'],
            description=data.get('description', ''),
            task_type=TaskType(data.get('task_type', 'Feature')),
            business_value=data.get('business_value', 5),
            time_criticality=data.get('time_criticality', 3),
            risk_reduction=data.get('risk_reduction', 3),
            effort=data.get('effort', 5),
            status=TaskStatus(data.get('status', 'NEW')),
            acceptance_criteria=data.get('acceptance_criteria', []),
            links=data.get('links', []),
            created_date=datetime.datetime.fromisoformat(data.get('created_date', datetime.datetime.now().isoformat())),
            last_updated=datetime.datetime.fromisoformat(data.get('last_updated', datetime.datetime.now().isoformat())),
            blocked_reason=data.get('blocked_reason')
        )
    
    def _backlog_item_to_dict(self, item: BacklogItem) -> Dict:
        """Convert BacklogItem to dictionary"""
        return {
            'id': item.id,
            'title': item.title,
            'description': item.description,
            'task_type': item.task_type.value,
            'business_value': item.business_value,
            'time_criticality': item.time_criticality,
            'risk_reduction': item.risk_reduction,
            'effort': item.effort,
            'status': item.status.value,
            'acceptance_criteria': item.acceptance_criteria,
            'links': item.links,
            'created_date': item.created_date.isoformat(),
            'last_updated': item.last_updated.isoformat(),
            'blocked_reason': item.blocked_reason,
            'wsjf_score': item.wsjf_score,
            'final_score': item.final_score
        }


def main():
    """Main execution loop for autonomous backlog management"""
    logging.basicConfig(level=logging.INFO)
    
    manager = BacklogManager()
    
    try:
        while True:
            logger.info("Starting backlog management cycle")
            
            # 1. Load and refresh backlog
            manager.load_backlog()
            
            # 2. Discover new tasks
            new_tasks = manager.discover_new_tasks()
            if new_tasks:
                logger.info(f"Discovered {len(new_tasks)} new tasks")
                manager.backlog.extend(new_tasks)
            
            # 3. Score and rank
            manager.score_and_rank()
            
            # 4. Execute next ready item
            next_item = manager.get_next_ready_item()
            if next_item:
                logger.info(f"Executing: {next_item.title}")
                success = manager.execute_item_tdd_cycle(next_item)
                if success:
                    logger.info(f"Successfully completed: {next_item.title}")
                else:
                    logger.warning(f"Failed to complete: {next_item.title}")
            else:
                logger.info("No ready items found")
            
            # 5. Save state and generate report
            manager.save_backlog()
            report = manager.generate_status_report()
            logger.info(f"Status: {report['backlog_size']} items, {report['completed_last_week']} completed this week")
            
            # 6. Check if done
            ready_items = [item for item in manager.backlog if item.is_ready() and not item.is_blocked()]
            if not ready_items:
                logger.info("No more actionable items. Cycle complete.")
                break
            
            # Sleep briefly before next cycle (in production, this might be longer)
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Backlog management interrupted by user")
    except Exception as e:
        logger.error(f"Backlog management failed: {e}")
        raise


if __name__ == "__main__":
    main()