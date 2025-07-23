# Autonomous Backlog Management System - Usage Guide

## Quick Start

### Prerequisites

Ensure you have the required dependencies installed:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Basic Usage

1. **Run the autonomous system** (continuous mode):
   ```bash
   python src/autonomous_backlog_executor.py
   ```

2. **Dry run** (see what would happen without making changes):
   ```bash
   python src/autonomous_backlog_executor.py --dry-run
   ```

3. **Check current status**:
   ```bash
   python src/autonomous_backlog_executor.py --status-only
   ```

## Command Line Options

### Main Execution Modes

| Option | Description | Example |
|--------|-------------|---------|
| `--dry-run` | Simulate execution without making changes | `--dry-run` |
| `--status-only` | Show backlog status and exit | `--status-only` |
| `--max-cycles N` | Limit to N execution cycles | `--max-cycles 5` |
| `--cycle-delay SECONDS` | Delay between cycles | `--cycle-delay 10` |
| `--repo-path PATH` | Repository path | `--repo-path /path/to/repo` |
| `--log-level LEVEL` | Logging level (DEBUG/INFO/WARNING/ERROR) | `--log-level DEBUG` |

### Example Commands

```bash
# Development workflow - limited cycles with detailed logging
python src/autonomous_backlog_executor.py --max-cycles 3 --cycle-delay 5 --log-level DEBUG

# Production run - continuous with moderate logging
python src/autonomous_backlog_executor.py --log-level INFO

# Quick status check
python src/autonomous_backlog_executor.py --status-only

# Safe preview of what would happen
python src/autonomous_backlog_executor.py --dry-run --max-cycles 1
```

## Understanding the Output

### Execution Phases

The system runs in 6 phases per cycle:

1. **🔄 Sync & Refresh**: Load backlog and recalculate priorities
2. **🔍 Task Discovery**: Scan for new TODO/FIXME/security issues
3. **📊 Score & Rank**: Apply WSJF methodology with aging
4. **⚡ Execute Ready Items**: Run TDD cycles on ready tasks
5. **🔐 Quality Gates**: Apply security and quality checks
6. **💾 Save State & Report**: Persist changes and generate metrics

### Status Indicators

| Emoji | Status | Description |
|-------|--------|-------------|
| 🆕 | NEW | Newly discovered item |
| 📝 | REFINED | Item has been analyzed and detailed |
| 🚀 | READY | Ready for execution (has acceptance criteria) |
| ⚡ | DOING | Currently being executed |
| 🔄 | PR | In pull request/review |
| 🚫 | BLOCKED | Blocked by external dependency |
| 🎯 | MERGED | Successfully merged |
| ✅ | DONE | Completed |

### Priority Scoring

Items are scored using WSJF (Weighted Shortest Job First):

```
Cost of Delay = Business Value + Time Criticality + Risk Reduction
WSJF Score = Cost of Delay / Effort
Final Score = WSJF Score × Aging Multiplier
```

**Score Ranges:**
- **20+**: Critical priority (security, blocking bugs)
- **10-20**: High priority (important features, performance)
- **5-10**: Medium priority (enhancements, refactoring)
- **2-5**: Low priority (nice-to-have features)
- **<2**: Very low priority (cleanup, documentation)

## Backlog Management

### Backlog File Structure

The system maintains a structured YAML backlog at `DOCS/backlog.yml`:

```yaml
items:
  - id: 'unique_item_id'
    title: 'Descriptive task title'
    description: 'Detailed description of what needs to be done'
    task_type: 'Feature'  # Feature|Bug|Security|Test|Doc|Refactor|Tech_Debt
    
    # WSJF Scoring (1-13 scale)
    business_value: 8      # Impact on business/users
    time_criticality: 5    # Urgency/deadline pressure
    risk_reduction: 3      # Risk mitigation value
    effort: 5              # Implementation effort
    
    status: 'READY'        # NEW|REFINED|READY|DOING|PR|BLOCKED|MERGED|DONE
    
    acceptance_criteria:   # Required for READY status
      - 'Criterion 1'
      - 'Criterion 2'
    
    links: []              # Related issues, PRs, docs
    created_date: '2025-07-23T12:00:00'
    last_updated: '2025-07-23T12:00:00'
    blocked_reason: null   # Reason if blocked
```

### Adding New Items

#### Method 1: Direct YAML Edit

Edit `DOCS/backlog.yml` and add a new item with proper scoring:

```yaml
- id: 'my_new_feature'
  title: 'Add user authentication'
  description: 'Implement JWT-based user authentication system'
  task_type: 'Feature'
  business_value: 8
  time_criticality: 6
  risk_reduction: 7
  effort: 8
  status: 'NEW'
  acceptance_criteria: []
  links: []
  created_date: '2025-07-23T12:00:00'
  last_updated: '2025-07-23T12:00:00'
  blocked_reason: null
```

#### Method 2: Code Comments (Auto-Discovery)

Add TODO/FIXME comments in your code:

```python
# TODO: Implement caching layer for improved performance
# FIXME: Handle edge case when user input is empty
# SECURITY: Validate all user inputs to prevent injection
```

The system will discover these automatically and convert them to backlog items.

### Refining Items

To make an item ready for execution:

1. **Add Acceptance Criteria**: Clear, testable requirements
2. **Set Proper Status**: Change from NEW → REFINED → READY
3. **Verify Scoring**: Ensure WSJF components are accurate
4. **Remove Blockers**: Address any blocking dependencies

Example of a well-refined item:

```yaml
- id: 'user_dashboard'
  title: 'Create user dashboard with key metrics'
  description: 'Implement a responsive dashboard showing user activity, preferences, and key performance indicators'
  task_type: 'Feature'
  business_value: 8
  time_criticality: 5
  risk_reduction: 2
  effort: 6
  status: 'READY'
  acceptance_criteria:
    - 'Dashboard displays user activity chart'
    - 'Show last 30 days of key metrics'
    - 'Responsive design works on mobile'
    - 'Page loads in under 2 seconds'
    - 'Unit tests cover >90% of component logic'
    - 'Accessibility score >90 (Lighthouse)'
  links:
    - 'https://github.com/org/repo/issues/123'
    - 'https://figma.com/design/dashboard-mockup'
  created_date: '2025-07-23T10:00:00'
  last_updated: '2025-07-23T12:00:00'
  blocked_reason: null
```

## Quality Gates Configuration

### Gate Configuration File

Create `config/quality_gates.yaml` to customize quality requirements:

```yaml
gates:
  security:
    enabled: true
    required: true      # Must pass for item to complete
    weight: 0.3         # Contribution to overall score
  
  testing:
    enabled: true
    required: true
    weight: 0.25
  
  documentation:
    enabled: true
    required: false     # Optional gate
    weight: 0.15

thresholds:
  overall_score: 75.0           # Minimum overall quality score
  security_score: 90.0          # Minimum security gate score
  critical_findings_max: 0      # Max critical security findings
  high_findings_max: 2          # Max high-severity findings
```

### Security Checks

The system automatically checks for:

- **Secrets in code**: API keys, passwords, tokens
- **Insecure patterns**: `eval()`, `shell=True`, unsafe deserialization
- **Input validation**: Missing validation for external inputs
- **Bandit findings**: Static security analysis results

### Quality Checks

- **Test Coverage**: Minimum percentage of code covered by tests
- **Code Complexity**: Cyclomatic complexity analysis
- **Documentation**: Docstring coverage for public APIs
- **Dependencies**: Vulnerable dependency detection

## Monitoring and Reporting

### Generated Reports

The system generates several types of reports:

1. **Status Dashboard** (`--status-only`): Real-time backlog overview
2. **Comprehensive Reports**: Full metrics analysis with trends
3. **Cycle Reports**: Per-execution cycle statistics
4. **Visualizations**: Charts and graphs in `DOCS/reports/`

### Key Metrics to Monitor

#### Velocity Metrics
- **Throughput**: Items completed per week (target: >2)
- **Cycle Time**: Hours from READY to DONE (target: <48)
- **Completion Rate**: % of started items finished (target: >90%)
- **Blocking Rate**: % of items that get blocked (target: <10%)

#### Quality Metrics
- **Test Coverage**: % of code covered by tests (target: >85%)
- **Bug Rate**: Bugs per feature implemented (target: <0.3)
- **Security Score**: Security gate compliance (target: >90)

#### Backlog Health
- **Ready Items**: Items available for immediate work
- **Aging Items**: Items older than 30 days
- **WSJF Distribution**: Priority score distribution

### Accessing Reports

Reports are stored in structured directories:

```
DOCS/
├── reports/           # Generated reports and charts
│   ├── comprehensive_report_20250723_120000.json
│   ├── latest_comprehensive_report.json
│   ├── wsjf_distribution_20250723_120000.png
│   └── quality_metrics_20250723_120000.png
├── metrics/           # Raw metrics data
│   └── cycle_20250723_120000.json
└── status/            # Real-time status files
    └── status_20250723_120000.json
```

## Troubleshooting

### Common Issues

#### 1. No Items Are Being Executed

**Symptoms**: System runs but doesn't execute any items

**Causes & Solutions**:
- **No READY items**: Check that items have acceptance criteria and status is READY
- **All items blocked**: Review `blocked_reason` fields and address dependencies
- **CI failures**: Check test failures and fix issues before execution can proceed

#### 2. Discovery Not Finding Tasks

**Symptoms**: No new tasks discovered from code comments

**Causes & Solutions**:
- **File permissions**: Ensure system can read source files
- **Tool availability**: Verify `grep`, `bandit` are installed and accessible
- **Search paths**: Check that `src/` and `tests/` directories exist

#### 3. Quality Gates Failing

**Symptoms**: Items fail quality checks and don't complete

**Causes & Solutions**:
- **Security findings**: Review and fix security issues reported
- **Test coverage**: Add tests to meet coverage threshold
- **Dependencies**: Update vulnerable dependencies

#### 4. Metrics Collection Errors

**Symptoms**: Reports are empty or missing

**Causes & Solutions**:
- **Disk space**: Ensure adequate space in `DOCS/` directory
- **Permissions**: Verify write permissions for metrics directories
- **Tool dependencies**: Install required packages (`matplotlib`, `pandas`)

### Debug Mode

Enable detailed logging to diagnose issues:

```bash
python src/autonomous_backlog_executor.py --log-level DEBUG --max-cycles 1
```

This will show:
- Detailed execution steps
- File operations and permissions
- Tool command output
- Error stack traces

### Recovery Procedures

#### Corrupted Backlog

If `DOCS/backlog.yml` becomes corrupted:

1. Restore from backup (if available)
2. Regenerate from `BACKLOG.md`:
   ```python
   from src.backlog_manager import BacklogManager
   manager = BacklogManager()
   items = manager._parse_markdown_backlog()
   manager.backlog = items
   manager.save_backlog()
   ```

#### Reset Metrics

To clear all metrics and start fresh:

```bash
rm -rf DOCS/metrics/
rm -rf DOCS/reports/
rm -rf DOCS/status/
```

#### Manual Item Management

To manually update item status:

```python
from src.backlog_manager import BacklogManager, TaskStatus

manager = BacklogManager()
manager.load_backlog()

# Find and update item
for item in manager.backlog:
    if item.id == 'problematic_item':
        item.status = TaskStatus.READY
        item.blocked_reason = None
        break

manager.save_backlog()
```

## Best Practices

### Writing Good Acceptance Criteria

✅ **Good Example**:
```yaml
acceptance_criteria:
  - 'API endpoint responds within 200ms for 95% of requests'
  - 'Input validation rejects malformed email addresses'
  - 'Error messages are user-friendly and actionable'
  - 'Unit tests achieve >95% coverage for new code'
  - 'Documentation includes usage examples'
```

❌ **Poor Example**:
```yaml
acceptance_criteria:
  - 'Make it work'
  - 'Should be fast'
  - 'Add tests'
```

### Effective WSJF Scoring

- **Business Value**: Focus on user/business impact, not technical elegance
- **Time Criticality**: Consider deadlines, dependencies, opportunity windows
- **Risk Reduction**: Value fixing known issues and preventing future problems
- **Effort**: Be realistic about implementation complexity

### Maintaining Backlog Health

1. **Regular Review**: Check aging items weekly
2. **Clear Blockers**: Address blocked items quickly
3. **Refine Items**: Keep a pipeline of READY items
4. **Remove Cruft**: Delete items that are no longer relevant
5. **Balance Types**: Mix features, bugs, tests, and refactoring

### Integration with Development Workflow

```bash
# Before starting work
python src/autonomous_backlog_executor.py --status-only

# During development (limited cycles)
python src/autonomous_backlog_executor.py --max-cycles 2 --cycle-delay 30

# Before pushing changes
python src/autonomous_backlog_executor.py --dry-run --max-cycles 1
```

## Advanced Usage

### Custom Task Discovery

Extend the discovery engine by adding custom patterns:

```python
# In your custom script
from src.backlog_manager import TaskDiscoveryEngine

engine = TaskDiscoveryEngine()
engine.comment_patterns['PERFORMANCE'] = TaskType.REFACTOR
engine.comment_patterns['DEBT'] = TaskType.TECH_DEBT
```

### Integration with CI/CD

```yaml
# .github/workflows/backlog.yml
name: Autonomous Backlog Management
on:
  schedule:
    - cron: '0 */6 * * *'  # Every 6 hours
  workflow_dispatch:

jobs:
  manage-backlog:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      - name: Run backlog management
        run: |
          python src/autonomous_backlog_executor.py --max-cycles 5
      - name: Commit changes
        run: |
          git config --local user.email "action@github.com"
          git config --local user.name "GitHub Action"
          git add DOCS/
          git commit -m "Update backlog status" || exit 0
          git push
```

### Monitoring Integration

Export metrics to monitoring systems:

```python
# Custom metrics exporter
from src.metrics_reporter import ReportGenerator

generator = ReportGenerator('/path/to/repo')
report = generator.generate_comprehensive_report(backlog, gate_results)

# Send to Prometheus, Grafana, etc.
```

This completes the comprehensive usage guide for the Autonomous Backlog Management System.