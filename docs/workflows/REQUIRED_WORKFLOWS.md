# Required GitHub Actions Workflows

## Overview

This document outlines the essential GitHub Actions workflows that need to be manually created in `.github/workflows/` to complete the SDLC automation for this advanced repository.

## Core CI/CD Pipeline

### 1. Main CI Pipeline (`ci.yml`)

```yaml
name: CI Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev]
    
    - name: Run tests with coverage
      run: |
        pytest --cov=src --cov-report=xml --cov-report=term-missing
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true

  lint:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[lint]
    
    - name: Run linting
      run: |
        ruff check src tests
        black --check src tests
        mypy src
        bandit -r src -f json -o bandit-report.json

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Run security scan
      run: |
        pip install safety
        safety check --json --output safety-report.json
```

### 2. Advanced Testing Pipeline (`advanced-testing.yml`)

```yaml
name: Advanced Testing

on:
  schedule:
    - cron: '0 2 * * 1'  # Weekly on Monday
  workflow_dispatch:

jobs:
  mutation-testing:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        pip install -e .[test]
        pip install mutmut
    
    - name: Run mutation tests
      run: python scripts/run_mutation_tests.py

  contract-testing:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        pip install -e .[test]
        pip install pact-python
    
    - name: Run contract tests
      run: ./scripts/run_contract_tests.sh --ci

  load-testing:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        pip install -e .[test]
        pip install locust faker
    
    - name: Run load tests
      run: python scripts/run_load_tests.py baseline --ci
```

### 3. Security and Compliance (`security.yml`)

```yaml
name: Security & Compliance

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 4 * * 1'  # Weekly security scan

jobs:
  codeql:
    name: CodeQL Analysis
    runs-on: ubuntu-latest
    permissions:
      actions: read
      contents: read
      security-events: write
    
    steps:
    - name: Checkout repository
      uses: actions/checkout@v4
    
    - name: Initialize CodeQL
      uses: github/codeql-action/init@v2
      with:
        languages: python
    
    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v2

  sbom-generation:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Install SBOM tools
      run: ./scripts/setup_sbom_tools.sh
    
    - name: Generate SBOM
      run: python sbom/generate_sbom.py --format spdx-json --output sbom.spdx.json
    
    - name: Upload SBOM artifact
      uses: actions/upload-artifact@v3
      with:
        name: sbom
        path: sbom.spdx.json

  slsa-compliance:
    permissions:
      actions: read
      id-token: write
      contents: write
    uses: slsa-framework/slsa-github-generator/.github/workflows/generator_generic_slsa3.yml@v1.9.0
    with:
      base64-subjects: "${{ needs.build.outputs.hashes }}"
      provenance-name: "provenance.intoto.jsonl"
```

### 4. Release Automation (`release.yml`)

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
    
    - name: Build package
      run: |
        pip install build
        python -m build
    
    - name: Upload artifacts
      uses: actions/upload-artifact@v3
      with:
        name: dist
        path: dist/

  release:
    needs: build
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Download artifacts
      uses: actions/download-artifact@v3
      with:
        name: dist
        path: dist/
    
    - name: Create Release
      uses: actions/create-release@v1
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      with:
        tag_name: ${{ github.ref }}
        release_name: Release ${{ github.ref }}
        draft: false
        prerelease: false
    
    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      with:
        password: ${{ secrets.PYPI_API_TOKEN }}
```

## Implementation Steps

1. **Create workflows directory**: `mkdir -p .github/workflows`
2. **Add workflow files**: Copy the above templates into individual `.yml` files
3. **Configure secrets**: Add required secrets to repository settings
4. **Test workflows**: Push changes and verify workflow execution
5. **Monitor and adjust**: Review workflow runs and optimize as needed

## Required Secrets

Configure these secrets in GitHub repository settings:

- `CODECOV_TOKEN`: For coverage reporting
- `PYPI_API_TOKEN`: For package publishing
- `GITHUB_TOKEN`: Automatically provided by GitHub

## Monitoring and Alerts

Workflows include built-in monitoring and will fail fast on issues. Review the Actions tab regularly for status updates and optimization opportunities.