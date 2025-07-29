# Security Policy

## Supported Versions

This project maintains security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| 0.1.x   | :x:                |

## Reporting Security Vulnerabilities

We take security vulnerabilities seriously. If you discover a security issue, please follow these steps:

### 1. Private Disclosure

**DO NOT** create a public GitHub issue for security vulnerabilities. Instead:

- **Email**: security@terragonlabs.io (if available)  
- **GitHub Security Advisory**: Use GitHub's private vulnerability reporting
- **PGP Key**: Available upon request for encrypted communication

### 2. Information to Include

Please provide as much information as possible:

- **Description**: Clear description of the vulnerability
- **Impact**: Potential impact and affected versions
- **Reproduction**: Step-by-step reproduction instructions
- **Environment**: OS, Python version, dependency versions
- **Proof of Concept**: If available (non-destructive only)

### 3. Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 7 days
- **Progress Updates**: Every 14 days until resolution
- **Disclosure**: Coordinated disclosure after fix is available

## Security Measures

### Automated Security Scanning

- **CodeQL**: Static analysis for code vulnerabilities
- **Dependabot**: Automated dependency vulnerability scanning
- **Bandit**: Python security linting in pre-commit hooks
- **Safety**: Python package vulnerability scanning
- **SBOM Generation**: Software Bill of Materials tracking

### Supply Chain Security

- **SLSA Level 3**: Provenance generation for releases
- **Dependency Pinning**: Exact version pinning in requirements
- **Signature Verification**: Package signing with Cosign
- **Container Scanning**: Docker image vulnerability scanning

### Development Security

- **Pre-commit Hooks**: Security scanning before commits
- **Secret Detection**: Automated secret scanning
- **Code Review**: All changes require review
- **Branch Protection**: Main branch protection rules

## Security Best Practices

### For Contributors

1. **Keep Dependencies Updated**: Regularly update dependencies
2. **Follow Secure Coding**: Avoid common security pitfalls
3. **Validate Inputs**: Always validate and sanitize inputs
4. **Handle Secrets**: Never commit secrets or keys
5. **Review Changes**: Carefully review security-related changes

### For Users

1. **Update Regularly**: Keep the package updated
2. **Monitor Advisories**: Watch for security announcements
3. **Validate Signatures**: Verify package signatures when possible
4. **Report Issues**: Report suspected vulnerabilities promptly

## Acknowledgments

We appreciate responsible disclosure of security vulnerabilities and will acknowledge contributors in security advisories (unless anonymity is requested).

## Contact

For security-related questions or concerns:
- **General Inquiries**: Create a GitHub issue (non-security)
- **Security Issues**: Follow private disclosure process above
- **Questions**: Contact maintainers through standard channels

Last updated: 2025-07-29