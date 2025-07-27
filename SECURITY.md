# Security Policy

## Supported Versions

The following versions of Fair Credit Scorer are currently supported with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| 0.1.x   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security vulnerability in this project, please report it responsibly.

### How to Report

1. **DO NOT** create a public GitHub issue for security vulnerabilities
2. Send an email to: security@example.com (replace with actual email)
3. Include the following information:
   - Description of the vulnerability
   - Steps to reproduce the issue
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- **Acknowledgment**: We will acknowledge receipt of your report within 48 hours
- **Investigation**: We will investigate and respond with our findings within 5 business days
- **Fix Timeline**: Critical vulnerabilities will be patched within 7 days, others within 30 days
- **Disclosure**: We follow responsible disclosure practices

### Security Best Practices

When using this project:

1. **Environment Variables**: Never commit secrets to version control
2. **Dependencies**: Keep dependencies updated using Dependabot
3. **Input Validation**: Validate all user inputs
4. **Data Privacy**: Ensure synthetic data doesn't contain real PII
5. **Access Control**: Implement proper authentication for production deployments

### Security Features

This project includes several security measures:

- **Dependency Scanning**: Automated vulnerability scanning with Dependabot
- **Code Analysis**: Static analysis with CodeQL and Bandit
- **Container Scanning**: Docker image vulnerability scanning with Trivy
- **Secret Detection**: Automated secret scanning in commits
- **SBOM Generation**: Software Bill of Materials for supply chain transparency

### Security Hardening

For production deployments:

1. Run containers as non-root user
2. Use minimal base images
3. Enable security policies (AppArmor/SELinux)
4. Implement network segmentation
5. Use secrets management systems
6. Enable audit logging
7. Regular security assessments

### Compliance

This project follows:

- OWASP Top 10 security practices
- NIST Cybersecurity Framework
- OpenSSF Scorecard recommendations
- Python security best practices

### Contact

For security-related questions or concerns:
- Email: security@example.com
- PGP Key: [Link to public key]

Thank you for helping keep Fair Credit Scorer secure!