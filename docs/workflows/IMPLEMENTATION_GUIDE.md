# GitHub Actions Implementation Guide

## 🚨 MANUAL SETUP REQUIRED

Due to GitHub App permission limitations, the following workflow files must be created manually by repository maintainers. This document provides complete templates and implementation guidance.

## Required Workflows

### 1. CI Pipeline (.github/workflows/ci.yml)

```yaml
name: Continuous Integration

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

env:
  PYTHON_VERSION: '3.11'

jobs:
  test:
    name: Test Suite
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.8', '3.9', '3.10', '3.11']
    
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt', '**/pyproject.toml') }}
        restore-keys: |
          ${{ runner.os }}-pip-
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev,test,lint]
        
    - name: Run comprehensive tests
      run: |
        python scripts/run_comprehensive_tests.py --skip mutation performance
        
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true

  security:
    name: Security Scan
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'
        
    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'

  build:
    name: Build and Test Container
    runs-on: ubuntu-latest
    needs: [test, security]
    steps:
    - uses: actions/checkout@v4
    - name: Build and scan container
      run: |
        ./scripts/build_and_scan.sh build production
        ./scripts/build_and_scan.sh test production
```

### 2. Security Scanning (.github/workflows/security.yml)

```yaml
name: Security Scanning

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 6 * * 1'  # Weekly on Monday

jobs:
  codeql:
    name: CodeQL Analysis
    runs-on: ubuntu-latest
    permissions:
      actions: read
      contents: read
      security-events: write
    
    strategy:
      fail-fast: false
      matrix:
        language: [ 'python' ]
    
    steps:
    - name: Checkout repository
      uses: actions/checkout@v4
      
    - name: Initialize CodeQL
      uses: github/codeql-action/init@v2
      with:
        languages: ${{ matrix.language }}
        
    - name: Autobuild
      uses: github/codeql-action/autobuild@v2
      
    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v2

  dependency-review:
    name: Dependency Review
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    steps:
    - name: Checkout repository
      uses: actions/checkout@v4
    - name: Dependency Review
      uses: actions/dependency-review-action@v3

  container-security:
    name: Container Security Scan
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Build image
      run: docker build -t test-image .
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: 'test-image'
        format: 'sarif'
        output: 'trivy-results.sarif'
    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'
```

### 3. Release Pipeline (.github/workflows/release.yml)

```yaml
name: Release Pipeline

on:
  push:
    tags:
      - 'v*'

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  release:
    name: Create Release
    runs-on: ubuntu-latest
    permissions:
      contents: write
      packages: write
    
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0
        
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
        
    - name: Build package
      run: |
        python -m build
        
    - name: Generate changelog
      id: changelog
      run: |
        echo "## Changes" > CHANGELOG.md
        git log --oneline --no-merges $(git describe --tags --abbrev=0 HEAD~1)..HEAD >> CHANGELOG.md
        
    - name: Create Release
      uses: softprops/action-gh-release@v1
      with:
        body_path: CHANGELOG.md
        files: dist/*
        
    - name: Build and push Docker image
      run: |
        echo ${{ secrets.GITHUB_TOKEN }} | docker login $REGISTRY -u ${{ github.actor }} --password-stdin
        REGISTRY=$REGISTRY VERSION=${{ github.ref_name }} ./scripts/build_and_scan.sh all production
```

## Implementation Steps

### 1. Repository Settings

Enable the following in repository settings:

```bash
# Branch protection rules for main branch
- Require pull request reviews (2 reviewers)
- Dismiss stale reviews when new commits are pushed
- Require status checks to pass before merging
- Require conversation resolution before merging
- Include administrators in restrictions

# Security settings
- Enable dependency graph
- Enable Dependabot alerts
- Enable Dependabot security updates
- Enable secret scanning
- Enable push protection for secrets
```

### 2. Required Secrets

Add these secrets in repository settings:

```bash
# Container registry
GHCR_TOKEN=<github_token_with_packages_write>

# Code coverage
CODECOV_TOKEN=<codecov_token>

# Image signing (optional)
COSIGN_PRIVATE_KEY=<cosign_private_key>
COSIGN_PASSWORD=<cosign_password>
```

### 3. Branch Protection

Configure branch protection rules:

```json
{
  "required_status_checks": {
    "strict": true,
    "checks": [
      "test",
      "security", 
      "build"
    ]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "required_approving_review_count": 2,
    "dismiss_stale_reviews": true
  },
  "restrictions": null
}
```

## Monitoring and Alerts

### 1. Set up Status Badges

Add to README.md:

```markdown
[![CI](https://github.com/danieleschmidt/fair-credit-scorer-bias-mitigation/workflows/CI/badge.svg)](https://github.com/danieleschmidt/fair-credit-scorer-bias-mitigation/actions/workflows/ci.yml)
[![Security](https://github.com/danieleschmidt/fair-credit-scorer-bias-mitigation/workflows/Security/badge.svg)](https://github.com/danieleschmidt/fair-credit-scorer-bias-mitigation/actions/workflows/security.yml)
[![Coverage](https://codecov.io/gh/danieleschmidt/fair-credit-scorer-bias-mitigation/branch/main/graph/badge.svg)](https://codecov.io/gh/danieleschmidt/fair-credit-scorer-bias-mitigation)
```

### 2. Configure Notifications

Set up Slack/email notifications for:
- Failed builds on main branch
- Security vulnerabilities
- Dependency updates
- Release deployments

## Quality Gates

All workflows implement these quality gates:

✅ **Code Quality**
- Ruff linting passes
- Black formatting check passes  
- MyPy type checking passes
- Test coverage ≥ 85%

✅ **Security**
- No high/critical vulnerabilities
- Dependency scanning passes
- Secret scanning passes
- SAST analysis passes

✅ **Testing**
- Unit tests pass (≥ 85% coverage)
- Integration tests pass
- Contract tests pass
- Performance benchmarks within thresholds

✅ **Build**
- Container builds successfully
- Security scan passes
- SBOM generated
- Image signed (if configured)

## Troubleshooting

### Common Issues

1. **Permission Denied**: Ensure GitHub token has correct permissions
2. **Docker Build Fails**: Check Dockerfile syntax and dependencies
3. **Test Failures**: Run tests locally first with comprehensive test runner
4. **Security Scan Failures**: Review vulnerability reports and update dependencies

### Debug Commands

```bash
# Local testing
./scripts/run_comprehensive_tests.py --verbose
./scripts/build_and_scan.sh all development

# Check workflow syntax
act --dry-run  # Requires act CLI tool
```

## Maintenance

### Weekly Tasks
- Review Dependabot PRs
- Check security scan results
- Update base images if needed

### Monthly Tasks  
- Review and update workflows
- Performance benchmark analysis
- Dependency audit and cleanup

### Quarterly Tasks
- Full security assessment
- Workflow optimization review
- Documentation updates