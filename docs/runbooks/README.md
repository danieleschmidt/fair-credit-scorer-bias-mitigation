# Operational Runbooks

This directory contains operational runbooks for the Fair Credit Scorer: Bias Mitigation project. These runbooks provide step-by-step procedures for common operational scenarios, troubleshooting, and incident response.

## Available Runbooks

### 🚨 Incident Response
- [**Incident Response**](incident-response.md) - Primary incident response procedures
- [**Performance Issues**](performance-troubleshooting.md) - Diagnosing and resolving performance problems
- [**Model Bias Detection**](bias-incident-response.md) - Responding to bias detection alerts

### 🔧 Maintenance & Operations
- [**Deployment Guide**](deployment.md) - Step-by-step deployment procedures
- [**Health Check Troubleshooting**](health-check-troubleshooting.md) - Resolving health check failures
- [**Dependency Updates**](dependency-updates.md) - Safe dependency update procedures

### 📊 Monitoring & Alerting
- [**Monitoring Setup**](monitoring-setup.md) - Setting up comprehensive monitoring
- [**Alert Response**](alert-response.md) - Responding to various alert types
- [**Metrics Analysis**](metrics-analysis.md) - Analyzing system and application metrics

### 🔒 Security Operations
- [**Security Incident Response**](security-incident-response.md) - Security breach procedures
- [**Vulnerability Management**](vulnerability-management.md) - Managing security vulnerabilities
- [**Access Management**](access-management.md) - User access and permission management

## Runbook Structure

Each runbook follows a consistent structure:

1. **Overview** - Brief description of the scenario
2. **Prerequisites** - Required access, tools, and knowledge
3. **Detection** - How to identify the issue
4. **Immediate Response** - Quick actions to stabilize
5. **Investigation** - Detailed diagnostic steps
6. **Resolution** - Step-by-step fix procedures
7. **Verification** - Confirming the fix works
8. **Post-Incident** - Follow-up actions and prevention
9. **Escalation** - When and how to escalate

## Quick Reference

### Emergency Contacts
- **On-call Engineer**: [Contact details]
- **Security Team**: [Contact details]
- **Infrastructure Team**: [Contact details]
- **Management Escalation**: [Contact details]

### Essential Commands
```bash
# Health checks
python -m src.health_check --check health
python -m src.health_check --check readiness
python -m src.health_check --check metrics

# System status
docker-compose ps
docker-compose logs -f
systemctl status [service-name]

# Performance analysis
htop
iotop
netstat -tuln
ps aux | grep python
```

### Common Endpoints
- **Health**: `/health`
- **Readiness**: `/ready`
- **Metrics**: `/metrics`
- **Liveness**: `/alive`

### Log Locations
- **Application logs**: `/var/log/fairness-eval/`
- **System logs**: `/var/log/syslog`
- **Container logs**: `docker logs [container-id]`

## Using These Runbooks

### For Operators
1. **Bookmark this page** for quick access during incidents
2. **Review runbooks regularly** to stay familiar with procedures
3. **Practice scenarios** during planned maintenance windows
4. **Update runbooks** based on new learnings and infrastructure changes

### For Developers
1. **Contribute to runbooks** when adding new features
2. **Test procedures** in development environments
3. **Document new failure modes** and their solutions
4. **Review runbooks** during code reviews

### During Incidents
1. **Stay calm** and follow the procedures
2. **Document actions** taken during the incident
3. **Communicate status** to stakeholders
4. **Escalate early** if procedures don't resolve the issue

## Runbook Maintenance

### Regular Reviews
- **Monthly**: Review and update all runbooks
- **Quarterly**: Practice major incident scenarios
- **Post-incident**: Update relevant runbooks with new learnings
- **After changes**: Update affected runbooks when systems change

### Contributing
1. **Create new runbooks** for recurring operational scenarios
2. **Update existing runbooks** when procedures change
3. **Test procedures** before documenting them
4. **Get peer review** for all runbook changes

### Quality Standards
- **Clear language**: Use simple, actionable language
- **Tested procedures**: All steps should be tested
- **Complete information**: Include all necessary context
- **Regular updates**: Keep information current

## Training Resources

### New Operator Onboarding
1. Read all runbooks
2. Practice procedures in development environment
3. Shadow experienced operators during incidents
4. Complete incident response training

### Ongoing Training
- **Disaster recovery drills**: Monthly
- **Security incident simulations**: Quarterly
- **New feature training**: As needed
- **Tool updates**: When tools change

## Feedback

Runbooks are living documents that should evolve based on operational experience. Please provide feedback:

- **GitHub Issues**: Report problems or suggest improvements
- **Post-incident reviews**: Identify runbook gaps or inaccuracies
- **Team discussions**: Share experiences and best practices
- **Documentation updates**: Keep runbooks current with system changes