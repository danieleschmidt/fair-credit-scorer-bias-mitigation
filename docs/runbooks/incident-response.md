# Incident Response Runbook

## Overview
This runbook provides step-by-step procedures for responding to incidents in the Fair Credit Scorer system.

## Incident Severity Classification

| Severity | Description | Response Time | Examples |
|----------|-------------|---------------|----------|
| **P1 - Critical** | System completely down, security breach, data loss | 15 minutes | API completely unavailable, data breach, model predictions failing |
| **P2 - High** | Significant degradation, partial outage | 30 minutes | High latency, some endpoints failing, bias metrics exceeding thresholds |
| **P3 - Medium** | Minor issues with workarounds | 2 hours | Minor UI bugs, non-critical feature failures |
| **P4 - Low** | Cosmetic issues, enhancements | Next business day | Documentation errors, minor UX improvements |

## Immediate Response (First 15 Minutes)

### 1. Assess the Situation
```bash
# Check system health
curl -f http://localhost:8080/health || echo "Health check failed"

# Check metrics endpoint
curl -f http://localhost:8080/metrics || echo "Metrics endpoint failed"

# Check application logs
docker logs fair-credit-scorer-app --tail 100

# Check system resources
docker stats --no-stream
```

### 2. Determine Severity
- **Is the system responding?** → Check health endpoints
- **Are predictions working?** → Test with sample data
- **Are metrics within normal ranges?** → Check monitoring dashboards
- **Is data being processed correctly?** → Verify data pipeline

### 3. Initial Triage Actions

#### For P1 Incidents
1. **Immediately notify** team lead and on-call engineer
2. **Create incident channel** in Slack: `#incident-YYYY-MM-DD-brief-description`
3. **Start incident timeline** in shared document
4. **Begin immediate mitigation** steps

#### For P2-P4 Incidents
1. **Create GitHub issue** with incident details
2. **Notify team** in appropriate Slack channel
3. **Begin investigation** following this runbook

## Investigation Steps

### 1. Gather Information
```bash
# System overview
kubectl get pods -n fair-credit-scorer  # If using Kubernetes
docker ps -a  # If using Docker

# Application logs
docker logs fair-credit-scorer-app --since=1h

# Error patterns
docker logs fair-credit-scorer-app 2>&1 | grep -i error | tail -20

# Performance metrics
docker stats --no-stream fair-credit-scorer-app
```

### 2. Check Dependencies
```bash
# Database connectivity
python -c "import src.data_loader_preprocessor; print('DB connection: OK')"

# External services
curl -f https://api.external-service.com/health || echo "External service down"

# File system
df -h  # Check disk space
ls -la data/  # Check data directory
```

### 3. Review Recent Changes
- Check recent deployments in GitHub Actions
- Review recent code changes in Git history
- Check configuration changes
- Verify environment variable changes

## Common Issues and Solutions

### Issue: High Memory Usage
```bash
# Check memory usage
free -h
docker stats --no-stream

# Check for memory leaks
ps aux --sort=-%mem | head -10

# Restart application if needed
docker restart fair-credit-scorer-app
```

### Issue: High CPU Usage
```bash
# Check CPU usage
top -bn1 | head -20

# Check application processes
ps aux --sort=-%cpu | head -10

# Scale application if needed
docker-compose up --scale app=3
```

### Issue: Model Predictions Failing
```bash
# Test model directly
python -c "
from src.evaluate_fairness import run_pipeline
try:
    result = run_pipeline('baseline', test_size=0.1)
    print('Model test: PASSED')
except Exception as e:
    print(f'Model test: FAILED - {e}')
"

# Check model files
ls -la models/ || echo "Model directory missing"

# Verify test data
python -c "
from src.data_loader_preprocessor import generate_data
try:
    data = generate_data('test_data.csv', n_samples=100)
    print(f'Data generation: PASSED - {len(data)} samples')
except Exception as e:
    print(f'Data generation: FAILED - {e}')
"
```

### Issue: Database Connection Problems
```bash
# Test database connectivity
python -c "
import sqlite3
try:
    conn = sqlite3.connect(':memory:')
    conn.close()
    print('Database test: PASSED')
except Exception as e:
    print(f'Database test: FAILED - {e}')
"
```

## Escalation Procedures

### When to Escalate
- **15 minutes**: If unable to identify root cause
- **30 minutes**: If unable to implement workaround for P1/P2
- **1 hour**: If incident is not resolved or contained

### Escalation Contacts
1. **Team Lead**: [Contact information]
2. **Engineering Manager**: [Contact information]
3. **On-Call Infrastructure**: [PagerDuty rotation]
4. **Security Team**: security@company.com (for security incidents)

### Escalation Information to Provide
- Incident severity and impact
- Timeline of events
- Steps taken so far
- Current system status
- Suspected root cause
- Proposed next steps

## Communication Templates

### Incident Announcement
```
🚨 INCIDENT ALERT - P[X] 🚨

**Summary**: [Brief description]
**Impact**: [Who/what is affected]
**Status**: Investigating
**ETA**: [Estimated resolution time]
**Updates**: Will update every 30 minutes

**Incident Commander**: @[name]
**Channel**: #incident-YYYY-MM-DD-description
```

### Status Update
```
📊 INCIDENT UPDATE - P[X] 📊

**Status**: [Investigating/Mitigating/Resolved]
**Progress**: [What has been done]
**Next Steps**: [What will be done next]
**ETA**: [Updated estimate]

**Last Updated**: [Timestamp]
```

### Resolution Announcement
```
✅ INCIDENT RESOLVED - P[X] ✅

**Summary**: [Brief description of what happened]
**Root Cause**: [Technical root cause]
**Resolution**: [What fixed it]
**Duration**: [Total incident time]

**Follow-up**: 
- Post-mortem scheduled for [date]
- Action items: [Link to tracking]
```

## Post-Incident Actions

### Immediate (Within 24 hours)
1. **Document timeline** of events
2. **Gather all logs** and evidence
3. **Update monitoring** if gaps identified
4. **Implement temporary fixes** if needed

### Short-term (Within 1 week)
1. **Conduct post-mortem** meeting
2. **Document lessons learned**
3. **Create action items** for improvements
4. **Update runbooks** based on experience

### Long-term (Within 1 month)
1. **Implement preventive measures**
2. **Update monitoring and alerting**
3. **Review and test incident procedures**
4. **Share learnings** with broader team

## Useful Commands and Queries

### System Health Checks
```bash
# Health endpoint
curl -s http://localhost:8080/health | jq .

# Metrics
curl -s http://localhost:8080/metrics

# Application logs
tail -f logs/application.log

# Error patterns
grep -i "error\|exception\|failed" logs/application.log | tail -20
```

### Performance Diagnostics
```bash
# Memory usage
free -h && echo "---" && ps aux --sort=-%mem | head -10

# CPU usage
uptime && echo "---" && ps aux --sort=-%cpu | head -10

# Disk usage
df -h && echo "---" && du -sh * | sort -hr | head -10

# Network connectivity
netstat -tuln | grep :8080
```

### Container Operations
```bash
# Container status
docker ps -a

# Container logs
docker logs --tail 100 fair-credit-scorer-app

# Container resources
docker stats --no-stream

# Container restart
docker restart fair-credit-scorer-app
```

Remember: **When in doubt, escalate early!** It's better to involve additional help sooner rather than later.