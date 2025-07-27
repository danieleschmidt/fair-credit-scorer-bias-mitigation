# Incident Response Runbook

## Overview

This runbook provides the primary incident response procedures for the Fair Credit Scorer: Bias Mitigation system. Use this as the starting point for any production incident.

## Prerequisites

- Access to production systems
- Monitoring dashboard access
- Communication channels (Slack, email, phone)
- Administrative privileges for emergency actions

## Incident Severity Levels

### Severity 1 (Critical)
- **Definition**: Complete service outage or security breach
- **Response Time**: Immediate (within 15 minutes)
- **Escalation**: Immediate management notification
- **Examples**: 
  - Application completely down
  - Data breach or unauthorized access
  - Critical bias detected in production models

### Severity 2 (High)
- **Definition**: Significant service degradation
- **Response Time**: Within 1 hour
- **Escalation**: Management notification within 2 hours
- **Examples**:
  - High error rates (>10%)
  - Performance degradation (>50% slower)
  - Partial feature failures

### Severity 3 (Medium)
- **Definition**: Minor service degradation
- **Response Time**: Within 4 hours
- **Escalation**: Daily status update
- **Examples**:
  - Intermittent errors (<5%)
  - Minor performance issues
  - Non-critical feature failures

### Severity 4 (Low)
- **Definition**: Cosmetic issues or enhancement requests
- **Response Time**: Within 24 hours
- **Escalation**: Weekly status update

## Initial Response (First 15 Minutes)

### 1. Acknowledge the Incident
```bash
# Check system status immediately
python -m src.health_check --check health --json
python -m src.health_check --check readiness --json
```

### 2. Assess Severity
- Review error rates and user impact
- Check if incident is ongoing or resolved
- Determine if immediate action is needed

### 3. Establish Communication
- Create incident channel: `#incident-YYYY-MM-DD-HHMM`
- Post initial status update
- Notify on-call team members

### 4. Begin Triage
```bash
# Quick system checks
docker-compose ps
docker-compose logs --tail=100 -f

# Check system resources
top
df -h
free -h

# Check application metrics
curl -s http://localhost:8000/metrics | jq '.'
```

## Investigation Phase

### System Health Assessment

#### 1. Check Application Health
```bash
# Comprehensive health check
python -m src.health_check --check health

# Check specific components
python -m src.health_check --check readiness
python -m src.health_check --check liveness
```

#### 2. Review System Resources
```bash
# Memory usage
free -h
cat /proc/meminfo | grep -E 'MemTotal|MemAvailable|MemFree'

# Disk usage
df -h
du -sh /var/log/*

# CPU and load
uptime
vmstat 1 5
iostat -x 1 5
```

#### 3. Check Application Logs
```bash
# Recent application logs
tail -n 100 /var/log/fairness-eval/application.log

# Error patterns
grep -i error /var/log/fairness-eval/application.log | tail -20
grep -i exception /var/log/fairness-eval/application.log | tail -20

# Docker container logs
docker-compose logs --tail=50 -f app
```

#### 4. Network and Connectivity
```bash
# Check listening ports
netstat -tuln | grep :8000

# Test connectivity
curl -I http://localhost:8000/health
curl -I http://localhost:8000/ready

# DNS resolution
nslookup [external-dependencies]
```

### Performance Analysis

#### 1. Response Time Analysis
```bash
# Test response times
time curl -s http://localhost:8000/health
time curl -s http://localhost:8000/metrics

# Check for slow queries or operations
grep -i "slow" /var/log/fairness-eval/application.log
```

#### 2. Resource Utilization
```bash
# Process monitoring
ps aux | grep python | head -10
pstree -p | grep python

# Memory usage by process
ps aux --sort=-%mem | head -10

# CPU usage by process  
ps aux --sort=-%cpu | head -10
```

#### 3. Database/Data Access
```bash
# Test data generation
python -c "
from src.data_loader_preprocessor import generate_data
import tempfile
import time
start = time.time()
with tempfile.NamedTemporaryFile(suffix='.csv') as tmp:
    data = generate_data(tmp.name, n_samples=1000)
    print(f'Generated {len(data)} samples in {time.time()-start:.2f}s')
"
```

### Security Assessment

#### 1. Check for Security Issues
```bash
# Check for failed authentication attempts
grep -i "authentication\|authorization\|failed" /var/log/fairness-eval/application.log

# Check for unusual access patterns
grep -i "suspicious\|anomaly\|unusual" /var/log/fairness-eval/application.log

# Scan for vulnerabilities
bandit -r src/ -f json -o security-scan.json
```

#### 2. Verify Data Integrity
```bash
# Test model functionality
python -c "
from src.health_check import HealthCheck
hc = HealthCheck()
result = hc._check_model_functionality()
print('Model check:', result)
"
```

## Common Resolution Procedures

### Application Restart
```bash
# Graceful restart
docker-compose restart app

# Full restart
docker-compose down
docker-compose up -d

# Check status
docker-compose ps
python -m src.health_check --check health
```

### Resource Cleanup
```bash
# Clear temporary files
find /tmp -name "*.tmp" -mtime +1 -delete
find /var/log -name "*.log.*" -mtime +7 -delete

# Clear Python cache
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete

# Docker cleanup
docker system prune -f
docker volume prune -f
```

### Configuration Reload
```bash
# Reload configuration
docker-compose exec app python -c "
from src.config import Config
config = Config()
print('Config reloaded successfully')
"

# Restart with new configuration
docker-compose restart app
```

### Scale Resources
```bash
# Scale containers (if using Docker Swarm or Kubernetes)
docker-compose up -d --scale app=3

# Check scaled services
docker-compose ps
```

## Recovery Verification

### 1. Health Checks
```bash
# Verify all health endpoints
curl -f http://localhost:8000/health || echo "Health check failed"
curl -f http://localhost:8000/ready || echo "Readiness check failed"
curl -f http://localhost:8000/alive || echo "Liveness check failed"
```

### 2. Functional Testing
```bash
# Test main functionality
fairness-eval --method baseline --test-size 0.3 --random-state 42

# Test API endpoints
curl -X POST http://localhost:8000/api/evaluate \
  -H "Content-Type: application/json" \
  -d '{"method": "baseline", "test_size": 0.3}'
```

### 3. Performance Validation
```bash
# Load testing
for i in {1..10}; do
  time curl -s http://localhost:8000/health >/dev/null
done

# Memory leak check
python -c "
import psutil
import time
process = psutil.Process()
print(f'Memory before: {process.memory_info().rss / 1024 / 1024:.2f} MB')
# Run some operations
time.sleep(60)  
print(f'Memory after: {process.memory_info().rss / 1024 / 1024:.2f} MB')
"
```

### 4. Monitoring Restoration
```bash
# Check metrics collection
curl -s http://localhost:8000/metrics | jq '.timestamp'

# Verify alerting
# (Manual verification of alert system)
```

## Post-Incident Activities

### 1. Documentation
- Update incident log with timeline
- Document root cause analysis
- Record lessons learned
- Update runbooks if needed

### 2. Communication
- Send final status update to stakeholders
- Schedule post-incident review meeting
- Update status pages and dashboards

### 3. Prevention
- Implement monitoring improvements
- Add automated detection for this issue type
- Create or update alerts
- Plan infrastructure improvements

## Escalation Procedures

### When to Escalate

#### Immediate Escalation (Severity 1)
- Security breaches or data leaks
- Complete service outages
- Model bias exceeding regulatory thresholds
- Unable to restore service within 1 hour

#### Planned Escalation (Severity 2-3)
- Unable to resolve within SLA timeframes
- Need additional resources or expertise
- Potential impact to business operations
- Customer-reported issues

### Escalation Contacts

1. **Technical Lead**: [Contact info]
2. **Engineering Manager**: [Contact info]
3. **Security Team**: [Contact info]
4. **Infrastructure Team**: [Contact info]
5. **Executive On-Call**: [Contact info]

### Escalation Process
1. Attempt resolution using this runbook
2. Consult with team members
3. If still unresolved, escalate with:
   - Incident summary
   - Actions taken
   - Current status
   - Assistance needed

## Tools and Resources

### Monitoring Dashboards
- System metrics: [Dashboard URL]
- Application metrics: [Dashboard URL]
- Business metrics: [Dashboard URL]

### Log Analysis
- Central logging: [Log system URL]
- Error tracking: [Error tracking URL]
- Performance monitoring: [APM URL]

### Communication
- Incident channel template: `#incident-YYYY-MM-DD-HHMM`
- Status page: [Status page URL]
- Escalation list: [Contact list]

### Documentation
- Architecture diagrams: [Documentation URL]
- API documentation: [API docs URL]
- Configuration docs: [Config docs URL]

## Checklist for Incident Commander

### Initial Response ✓
- [ ] Acknowledge incident within 15 minutes
- [ ] Assess severity level
- [ ] Create incident communication channel
- [ ] Gather initial investigation team
- [ ] Send initial status update

### Investigation ✓
- [ ] Complete system health assessment
- [ ] Identify root cause
- [ ] Determine resolution approach
- [ ] Estimate time to resolution
- [ ] Communicate findings to stakeholders

### Resolution ✓
- [ ] Execute resolution procedures
- [ ] Verify fix effectiveness
- [ ] Monitor for recurring issues
- [ ] Update stakeholders on resolution
- [ ] Document actions taken

### Post-Incident ✓
- [ ] Send final incident summary
- [ ] Schedule post-incident review
- [ ] Update runbooks and procedures
- [ ] Implement preventive measures
- [ ] Close incident ticket