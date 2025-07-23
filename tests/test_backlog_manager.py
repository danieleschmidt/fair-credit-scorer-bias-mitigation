"""
Comprehensive tests for the autonomous backlog management system.

Tests cover:
- Backlog item scoring and ranking (WSJF)
- Task discovery from various sources  
- TDD execution cycles
- Security and quality gates
- Status reporting and metrics
"""

import datetime
import json
import os
import tempfile
import pytest
from unittest.mock import Mock, patch, MagicMock
import yaml

from src.backlog_manager import (
    BacklogItem, BacklogManager, TaskDiscoveryEngine, TaskDiscoveryResult,
    TaskType, TaskStatus
)


class TestBacklogItem:
    """Test BacklogItem scoring and lifecycle"""
    
    def test_backlog_item_wsjf_calculation(self):
        """Test WSJF score calculation"""
        item = BacklogItem(
            id="test_1",
            title="Test Task",
            description="Test description",
            task_type=TaskType.FEATURE,
            business_value=8,
            time_criticality=5,
            risk_reduction=3,
            effort=4
        )
        
        assert item.cost_of_delay == 16  # 8 + 5 + 3
        assert item.wsjf_score == 4.0    # 16 / 4
        assert item.aging_multiplier == 1.0  # New item
        assert item.final_score == 4.0   # 4.0 * 1.0
    
    def test_aging_multiplier_calculation(self):
        """Test aging multiplier for old items"""
        old_date = datetime.datetime.now() - datetime.timedelta(days=45)
        
        item = BacklogItem(
            id="test_old",
            title="Old Task", 
            description="Old task",
            task_type=TaskType.BUG,
            business_value=5,
            effort=5,
            created_date=old_date
        )
        
        # Should be capped at 2x after 30+ days
        assert item.aging_multiplier == 2.0
        assert item.final_score == 2.0  # 1.0 * 2.0
    
    def test_zero_effort_handling(self):
        """Test handling of zero effort to avoid division by zero"""
        item = BacklogItem(
            id="test_zero",
            title="Zero Effort",
            description="Test",
            task_type=TaskType.DOC,
            business_value=5,
            effort=0  # This should be handled gracefully
        )
        
        assert item.wsjf_score == 5.0  # Should use max(effort, 1)
    
    def test_is_ready_validation(self):
        """Test ready state validation"""
        item = BacklogItem(
            id="test_ready",
            title="Ready Task",
            description="Test",
            task_type=TaskType.FEATURE,
            status=TaskStatus.READY,
            acceptance_criteria=["Criterion 1", "Criterion 2"]
        )
        
        assert item.is_ready() is True
        
        # Test not ready cases
        item.status = TaskStatus.NEW
        assert item.is_ready() is False
        
        item.status = TaskStatus.READY
        item.acceptance_criteria = []
        assert item.is_ready() is False
        
        item.acceptance_criteria = ["Criterion 1"]
        item.blocked_reason = "Waiting for dependency"
        assert item.is_ready() is False
    
    def test_is_blocked_validation(self):
        """Test blocked state validation"""
        item = BacklogItem(
            id="test_blocked",
            title="Blocked Task",
            description="Test",
            task_type=TaskType.FEATURE
        )
        
        assert item.is_blocked() is False
        
        item.status = TaskStatus.BLOCKED
        assert item.is_blocked() is True
        
        item.status = TaskStatus.NEW
        item.blocked_reason = "Missing requirements"
        assert item.is_blocked() is True


class TestTaskDiscoveryEngine:
    """Test task discovery from various sources"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.engine = TaskDiscoveryEngine(self.temp_dir)
    
    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('subprocess.run')
    def test_scan_code_comments(self, mock_run):
        """Test scanning for TODO/FIXME comments"""
        # Mock grep output
        mock_run.return_value.stdout = (
            "/path/to/file.py:42:# TODO: Implement feature X\n"
            "/path/to/file.py:56:# FIXME: Handle edge case\n"
            "/path/to/other.py:12:# HACK: Temporary workaround\n"
        )
        mock_run.return_value.returncode = 0
        
        results = self.engine.scan_code_comments()
        
        assert len(results) >= 3  # May find duplicates across patterns
        
        # Check that different comment types are detected
        found_types = {r.task_type for r in results}
        assert TaskType.FEATURE in found_types or TaskType.BUG in found_types
    
    @patch('subprocess.run')
    def test_scan_failing_tests(self, mock_run):
        """Test scanning for failing tests"""
        mock_run.return_value.stderr = (
            "ERROR collecting tests/test_module.py::test_function\n"
            "FAILED tests/test_other.py::test_broken - AssertionError\n"
        )
        mock_run.return_value.returncode = 1
        
        results = self.engine.scan_failing_tests()
        
        assert len(results) == 2
        assert all(r.task_type == TaskType.BUG for r in results)
        assert all(r.priority == 8 for r in results)  # High priority for broken tests
    
    @patch('subprocess.run') 
    def test_scan_security_vulnerabilities(self, mock_run):
        """Test scanning for security issues"""
        mock_bandit_output = {
            "results": [
                {
                    "filename": "/path/to/vulnerable.py",
                    "line_number": 25,
                    "test_name": "B101",
                    "issue_text": "Use of assert detected"
                },
                {
                    "filename": "/path/to/other.py", 
                    "line_number": 67,
                    "test_name": "B602",
                    "issue_text": "subprocess call with shell=True"
                }
            ]
        }
        
        mock_run.return_value.stdout = json.dumps(mock_bandit_output)
        mock_run.return_value.returncode = 0
        
        results = self.engine.scan_security_vulnerabilities()
        
        assert len(results) == 2
        assert all(r.task_type == TaskType.SECURITY for r in results)
        assert all(r.priority == 13 for r in results)  # Always high priority
    
    def test_estimate_priority_by_keywords(self):
        """Test priority estimation based on content keywords"""
        test_cases = [
            ("Critical security issue", TaskType.SECURITY, 13),
            ("Urgent bug fix needed", TaskType.BUG, 8),
            ("Performance improvement", TaskType.REFACTOR, 8),
            ("Important feature", TaskType.FEATURE, 5),
            ("Simple cleanup task", TaskType.REFACTOR, 3),
            ("Documentation update", TaskType.DOC, 2)
        ]
        
        for content, task_type, expected_priority in test_cases:
            priority = self.engine._estimate_priority(content, task_type)
            assert priority >= expected_priority or priority == expected_priority


class TestBacklogManager:
    """Test backlog manager functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = BacklogManager(self.temp_dir)
        
        # Create test directories
        os.makedirs(os.path.join(self.temp_dir, "DOCS"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, "src"), exist_ok=True)
    
    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_empty_backlog(self):
        """Test loading when no backlog exists"""
        items = self.manager.load_backlog()
        assert items == []
        assert self.manager.backlog == []
    
    def test_load_backlog_from_yaml(self):
        """Test loading backlog from YAML file"""
        test_data = {
            'items': [
                {
                    'id': 'test_1',
                    'title': 'Test Task',
                    'description': 'Test description',
                    'task_type': 'Feature',
                    'business_value': 8,
                    'effort': 5,
                    'status': 'NEW',
                    'created_date': datetime.datetime.now().isoformat(),
                    'last_updated': datetime.datetime.now().isoformat()
                }
            ]
        }
        
        # Write test YAML
        with open(self.manager.backlog_file, 'w') as f:
            yaml.dump(test_data, f)
        
        items = self.manager.load_backlog()
        
        assert len(items) == 1
        assert items[0].id == 'test_1'
        assert items[0].title == 'Test Task'
        assert items[0].task_type == TaskType.FEATURE
        assert items[0].business_value == 8
    
    def test_parse_markdown_backlog(self):
        """Test parsing existing BACKLOG.md format"""
        # Create sample BACKLOG.md content
        backlog_content = """
# Impact-Ranked Backlog

## High Priority Tasks (Priority Score: 8+)

### 1. ✅ Fix scikit-learn deprecation warnings (Score: 12) - COMPLETED
- **Impact**: High - Ensures future compatibility
- **Effort**: Low (2 hours)
- **Business Value**: 8/10 - Prevents future breaking changes

### 2. Add input validation (Score: 9)
- **Impact**: High - Prevents runtime errors
- **Effort**: Medium (3 hours)
- **Business Value**: 8/10 - Improves robustness
"""
        
        backlog_md = os.path.join(self.temp_dir, "BACKLOG.md")
        with open(backlog_md, 'w') as f:
            f.write(backlog_content)
        
        items = self.manager._parse_markdown_backlog()
        
        assert len(items) == 2
        
        # Check completed task
        completed_item = next((item for item in items if item.status == TaskStatus.DONE), None)
        assert completed_item is not None
        assert "Fix scikit-learn" in completed_item.title
        
        # Check active task
        active_item = next((item for item in items if item.status == TaskStatus.NEW), None)
        assert active_item is not None
        assert "input validation" in active_item.title
    
    def test_task_type_inference(self):
        """Test task type inference from titles"""
        test_cases = [
            ("Fix security vulnerability", TaskType.SECURITY),
            ("Bug fix for login", TaskType.BUG),
            ("Add test coverage", TaskType.TEST),
            ("Update documentation", TaskType.DOC),
            ("Refactor data loader", TaskType.REFACTOR),
            ("New user dashboard", TaskType.FEATURE)
        ]
        
        for title, expected_type in test_cases:
            inferred_type = self.manager._infer_task_type(title)
            assert inferred_type == expected_type
    
    @patch.object(TaskDiscoveryEngine, 'scan_code_comments')
    @patch.object(TaskDiscoveryEngine, 'scan_failing_tests')
    @patch.object(TaskDiscoveryEngine, 'scan_security_vulnerabilities')
    def test_discover_new_tasks(self, mock_security, mock_tests, mock_comments):
        """Test task discovery and deduplication"""
        # Setup existing backlog
        existing_item = BacklogItem(
            id="existing",
            title="Existing Task",
            description="Existing description",
            task_type=TaskType.FEATURE
        )
        self.manager.backlog = [existing_item]
        
        # Mock discovery results
        mock_comments.return_value = [
            TaskDiscoveryResult(
                file_path="/test.py",
                line_number=1,
                content="TODO: New feature",
                task_type=TaskType.FEATURE,
                priority=5
            )
        ]
        mock_tests.return_value = []
        mock_security.return_value = []
        
        new_tasks = self.manager.discover_new_tasks()
        
        assert len(new_tasks) == 1
        assert "New feature" in new_tasks[0].title
        assert new_tasks[0].task_type == TaskType.FEATURE
    
    def test_score_and_rank(self):
        """Test scoring and ranking of backlog items"""
        items = [
            BacklogItem(
                id="low", title="Low Priority", description="Test",
                task_type=TaskType.DOC, business_value=2, effort=5
            ),
            BacklogItem(
                id="high", title="High Priority", description="Test", 
                task_type=TaskType.SECURITY, business_value=13, effort=3
            ),
            BacklogItem(
                id="medium", title="Medium Priority", description="Test",
                task_type=TaskType.FEATURE, business_value=8, effort=5
            )
        ]
        
        self.manager.backlog = items
        ranked_items = self.manager.score_and_rank()
        
        # Should be sorted by final_score descending
        assert ranked_items[0].id == "high"   # Highest score
        assert ranked_items[-1].id == "low"   # Lowest score
        
        # Verify scores are calculated
        for item in ranked_items:
            assert item.wsjf_score > 0
            assert item.final_score > 0
    
    def test_get_next_ready_item(self):
        """Test getting next ready item for execution"""
        items = [
            BacklogItem(
                id="blocked", title="Blocked", description="Test",
                task_type=TaskType.FEATURE, status=TaskStatus.BLOCKED
            ),
            BacklogItem(
                id="not_ready", title="Not Ready", description="Test",
                task_type=TaskType.FEATURE, status=TaskStatus.NEW
            ),
            BacklogItem(
                id="ready", title="Ready", description="Test",
                task_type=TaskType.FEATURE, status=TaskStatus.READY,
                acceptance_criteria=["AC1", "AC2"]
            )
        ]
        
        self.manager.backlog = items
        self.manager.score_and_rank()
        
        next_item = self.manager.get_next_ready_item()
        
        assert next_item is not None
        assert next_item.id == "ready"
    
    @patch('subprocess.run')
    def test_run_ci_checks_success(self, mock_run):
        """Test successful CI pipeline execution"""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stderr = ""
        
        result = self.manager._run_ci_checks()
        assert result is True
        
        # Should run multiple checks
        assert mock_run.call_count >= 3
    
    @patch('subprocess.run')
    def test_run_ci_checks_failure(self, mock_run):
        """Test CI pipeline failure handling"""
        mock_run.return_value.returncode = 1
        mock_run.return_value.stderr = "Test failed"
        
        result = self.manager._run_ci_checks()
        assert result is False
    
    @patch.object(BacklogManager, '_run_ci_checks')
    @patch.object(TaskDiscoveryEngine, 'scan_security_vulnerabilities')
    def test_execute_item_tdd_cycle_success(self, mock_security, mock_ci):
        """Test successful TDD cycle execution"""
        mock_security.return_value = []  # No security issues
        mock_ci.return_value = True      # CI passes
        
        item = BacklogItem(
            id="test_execute",
            title="Test Execution",
            description="Test item",
            task_type=TaskType.FEATURE,
            status=TaskStatus.READY,
            acceptance_criteria=["AC1"]
        )
        
        result = self.manager.execute_item_tdd_cycle(item)
        
        assert result is True
        assert item.status == TaskStatus.PR
        assert item.last_updated is not None
    
    @patch.object(BacklogManager, '_run_ci_checks')
    @patch.object(TaskDiscoveryEngine, 'scan_security_vulnerabilities')
    def test_execute_item_tdd_cycle_failure(self, mock_security, mock_ci):
        """Test TDD cycle failure handling"""
        mock_security.return_value = []  # No security issues
        mock_ci.return_value = False     # CI fails
        
        item = BacklogItem(
            id="test_fail",
            title="Test Failure",
            description="Test item",
            task_type=TaskType.FEATURE,
            status=TaskStatus.READY,
            acceptance_criteria=["AC1"]
        )
        
        result = self.manager.execute_item_tdd_cycle(item)
        
        assert result is False
        assert item.status == TaskStatus.DOING  # Should remain in progress
    
    def test_save_and_load_backlog_roundtrip(self):
        """Test saving and loading backlog maintains data integrity"""
        original_items = [
            BacklogItem(
                id="test_save",
                title="Test Save",
                description="Test description",
                task_type=TaskType.FEATURE,
                business_value=8,
                effort=5,
                status=TaskStatus.READY,
                acceptance_criteria=["AC1", "AC2"]
            )
        ]
        
        self.manager.backlog = original_items
        self.manager.save_backlog()
        
        # Load into new manager
        new_manager = BacklogManager(self.temp_dir)
        loaded_items = new_manager.load_backlog()
        
        assert len(loaded_items) == 1
        loaded_item = loaded_items[0]
        
        assert loaded_item.id == "test_save"
        assert loaded_item.title == "Test Save"
        assert loaded_item.task_type == TaskType.FEATURE
        assert loaded_item.business_value == 8
        assert loaded_item.effort == 5
        assert loaded_item.status == TaskStatus.READY
        assert loaded_item.acceptance_criteria == ["AC1", "AC2"]
    
    def test_generate_status_report(self):
        """Test status report generation"""
        # Create test items with different statuses
        items = [
            BacklogItem(id="1", title="Done", description="Test", 
                       task_type=TaskType.FEATURE, status=TaskStatus.DONE,
                       last_updated=datetime.datetime.now() - datetime.timedelta(days=2)),
            BacklogItem(id="2", title="Blocked", description="Test",
                       task_type=TaskType.BUG, status=TaskStatus.BLOCKED,
                       blocked_reason="Waiting for dependency"),
            BacklogItem(id="3", title="Ready", description="Test",
                       task_type=TaskType.REFACTOR, status=TaskStatus.READY)
        ]
        
        self.manager.backlog = items
        self.manager.score_and_rank()
        
        report = self.manager.generate_status_report()
        
        assert 'timestamp' in report
        assert report['backlog_size'] == 3
        assert report['status_distribution']['DONE'] == 1
        assert report['status_distribution']['BLOCKED'] == 1
        assert report['status_distribution']['READY'] == 1
        assert report['completed_last_week'] == 1
        assert len(report['blocked_items']) == 1
        assert len(report['top_priority_items']) <= 5
        
        # Check if report file was created
        status_files = os.listdir(self.manager.status_dir)
        assert len(status_files) >= 1
        assert any(f.startswith('status_') and f.endswith('.json') for f in status_files)


class TestBacklogManagerIntegration:
    """Integration tests for the complete backlog management workflow"""
    
    def setup_method(self):
        """Setup integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = BacklogManager(self.temp_dir)
        
        # Create realistic test environment structure
        os.makedirs(os.path.join(self.temp_dir, "DOCS"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, "src"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, "tests"), exist_ok=True)
    
    def teardown_method(self):
        """Cleanup integration test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('subprocess.run')
    def test_full_backlog_cycle(self, mock_run):
        """Test complete backlog management cycle"""
        # Mock all subprocess calls to return success
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""
        
        # Create initial backlog with mixed priorities
        initial_items = [
            BacklogItem(
                id="high_priority",
                title="Critical Security Fix", 
                description="Fix auth vulnerability",
                task_type=TaskType.SECURITY,
                business_value=13,
                effort=3,
                status=TaskStatus.READY,
                acceptance_criteria=["Fix implemented", "Tests pass"]
            ),
            BacklogItem(
                id="medium_priority",
                title="Add Feature X",
                description="New user feature",
                task_type=TaskType.FEATURE,
                business_value=8,
                effort=5,
                status=TaskStatus.READY,
                acceptance_criteria=["Feature implemented"]
            ),
            BacklogItem(
                id="blocked_item",
                title="Blocked Task",
                description="Waiting for dependency",
                task_type=TaskType.FEATURE,
                status=TaskStatus.BLOCKED,
                blocked_reason="External API not ready"
            )
        ]
        
        self.manager.backlog = initial_items
        
        # Test scoring and ranking
        ranked = self.manager.score_and_rank()
        assert ranked[0].id == "high_priority"  # Should be first due to security + high value
        
        # Test getting next ready item (should skip blocked)
        next_item = self.manager.get_next_ready_item()
        assert next_item.id == "high_priority"
        
        # Test TDD execution cycle
        with patch.object(self.manager.discovery_engine, 'scan_security_vulnerabilities', return_value=[]):
            success = self.manager.execute_item_tdd_cycle(next_item)
            assert success is True
            assert next_item.status == TaskStatus.PR
        
        # Test status reporting
        report = self.manager.generate_status_report()
        assert report['backlog_size'] == 3
        assert len(report['blocked_items']) == 1
        
        # Test save/load cycle
        self.manager.save_backlog()
        
        new_manager = BacklogManager(self.temp_dir)
        loaded_items = new_manager.load_backlog()
        assert len(loaded_items) == 3
        
        # Verify the PR item was saved correctly
        pr_item = next((item for item in loaded_items if item.status == TaskStatus.PR), None)
        assert pr_item is not None
        assert pr_item.id == "high_priority"


@pytest.fixture
def sample_backlog_items():
    """Fixture providing sample backlog items for testing"""
    return [
        BacklogItem(
            id="security_1",
            title="Fix SQL injection vulnerability",
            description="Sanitize user inputs in auth module",
            task_type=TaskType.SECURITY,
            business_value=13,
            time_criticality=13,
            risk_reduction=13,
            effort=5,
            status=TaskStatus.READY,
            acceptance_criteria=[
                "All user inputs are sanitized",
                "SQL injection tests pass",
                "Security scan shows no vulnerabilities"
            ]
        ),
        BacklogItem(
            id="feature_1", 
            title="Add user dashboard",
            description="Create personalized user dashboard",
            task_type=TaskType.FEATURE,
            business_value=8,
            time_criticality=5,
            risk_reduction=2,
            effort=8,
            status=TaskStatus.NEW
        ),
        BacklogItem(
            id="bug_1",
            title="Fix memory leak in data processor",
            description="Memory usage grows unbounded during large data processing",
            task_type=TaskType.BUG,
            business_value=5,
            time_criticality=8,
            risk_reduction=8,
            effort=3,
            status=TaskStatus.REFINED
        )
    ]


def test_wsjf_scoring_priorities(sample_backlog_items):
    """Test that WSJF scoring correctly prioritizes items"""
    for item in sample_backlog_items:
        item.__post_init__()  # Recalculate scores
    
    # Sort by final score
    sorted_items = sorted(sample_backlog_items, key=lambda x: x.final_score, reverse=True)
    
    # Security item should have highest priority due to maximum values
    assert sorted_items[0].id == "security_1"
    
    # Bug should be higher than feature due to better effort ratio
    bug_item = next(item for item in sorted_items if item.id == "bug_1")
    feature_item = next(item for item in sorted_items if item.id == "feature_1")
    
    assert bug_item.final_score > feature_item.final_score


def test_backlog_yaml_schema_validation():
    """Test that backlog YAML follows expected schema"""
    manager = BacklogManager()
    
    # Create test item
    item = BacklogItem(
        id="schema_test",
        title="Schema Test",
        description="Test schema compliance",
        task_type=TaskType.FEATURE,
        business_value=5,
        effort=3,
        acceptance_criteria=["Criterion 1", "Criterion 2"]
    )
    
    # Convert to dict and back
    item_dict = manager._backlog_item_to_dict(item)
    restored_item = manager._dict_to_backlog_item(item_dict)
    
    # Verify all fields are preserved
    assert restored_item.id == item.id
    assert restored_item.title == item.title
    assert restored_item.description == item.description
    assert restored_item.task_type == item.task_type
    assert restored_item.business_value == item.business_value
    assert restored_item.effort == item.effort
    assert restored_item.acceptance_criteria == item.acceptance_criteria
    assert restored_item.status == item.status