# 🤖 Repository Hygiene Bot

An autonomous DevSecOps bot that applies standardized security and development practices across all your GitHub repositories.

## ✨ Why this exists

Managing repository hygiene across multiple projects is time-consuming and error-prone. This bot automates the application of DevSecOps best practices, ensuring consistent security scanning, dependency management, and documentation standards across your entire GitHub organization.

## ⚡ Quick Start

### Prerequisites

- GitHub personal access token with repo permissions
- Python 3.8+
- `requests` and `PyYAML` packages

### Installation

```bash
# Install dependencies
pip install requests PyYAML

# Set environment variables
export GITHUB_TOKEN="your_github_token_here"
export GITHUB_USER="your_github_username"
```

### Basic Usage

```bash
# Run for all repositories
python -m src.repo_hygiene_cli

# Dry run (see what would change)
python -m src.repo_hygiene_cli --dry-run

# Process single repository
python -m src.repo_hygiene_cli --single-repo my-repo

# Use wrapper script with better UX
./scripts/run-hygiene-bot.sh --dry-run --verbose
```

## 🔍 Key Features

| Feature | Description | Impact |
|---------|-------------|---------|
| **Metadata Standardization** | Ensures descriptions, topics, and homepages | Better discoverability |
| **Community Health Files** | Adds LICENSE, CODE_OF_CONDUCT, CONTRIBUTING, SECURITY | Professional appearance |
| **Security Scanning** | Deploys CodeQL, Dependabot, OpenSSF Scorecard | Automated vulnerability detection |
| **SBOM Generation** | Creates Software Bill of Materials | Supply chain transparency |
| **README Enhancement** | Adds badges and required sections | Improved documentation |
| **Stale Repo Archival** | Archives old "Main-Project" repositories | Repository cleanup |
| **Metrics Collection** | Tracks compliance across repositories | Visibility into hygiene status |

## 🏗️ What the Bot Does

### STEP 0: Repository Enumeration
- Fetches all owned repositories via GitHub API
- Filters out archived, forked, and disabled repos
- Skips templates to avoid unintended modifications

### STEP 1: Metadata Updates
- Adds meaningful descriptions if missing
- Sets homepage to `https://{user}.github.io`
- Applies standard topics: `llmops`, `rag`, `semantic-release`, `sbom`, `github-actions`

### STEP 2: Community Health Files
Creates missing files with standard templates:
- `LICENSE` (Apache 2.0)
- `CODE_OF_CONDUCT.md` (Contributor Covenant v2.1)
- `CONTRIBUTING.md` (Conventional Commits guide)
- `SECURITY.md` (Vulnerability disclosure policy)
- `.github/ISSUE_TEMPLATE/{bug,feature}.yml`

### STEP 3: Security Workflows
Deploys automated security scanning:
- **CodeQL**: Static analysis for security vulnerabilities
- **Dependabot**: Automated dependency updates
- **OpenSSF Scorecard**: Supply chain security assessment

### STEP 4: SBOM & Release Signing
- Generates Software Bill of Materials with CycloneDX
- Sets up Cosign keyless signing for container images
- Nightly SBOM diff checking for new vulnerabilities

### STEP 5: README Badges
Injects status badges for:
- Build status
- Semantic release
- SBOM availability

### STEP 6: Stale Repository Archival
- Archives "Main-Project" repositories older than 400 days
- Prevents clutter in active repository lists

### STEP 7: README Sections
Ensures presence of required documentation sections:
- ✨ Why this exists
- ⚡ Quick Start
- 🔍 Key Features
- 🗺 Road Map
- 🤝 Contributing

### STEP 8: Metrics Collection
Generates compliance metrics in `metrics/profile_hygiene.json`:
```json
{
  "repo": "example-repo",
  "description_set": true,
  "topics_count": 7,
  "license_exists": true,
  "code_scanning": true,
  "dependabot": true,
  "scorecard": true,
  "sbom_workflow": true
}
```

### STEP 9: Pull Request Creation
- Creates branch: `repo-hygiene-bot-updates`
- Opens PR titled: "✨ repo‑hygiene‑bot update"
- Includes detailed change summary
- Applies `automated-maintenance` label

## 🗺 Road Map

### Current Release (v1.0)
- [x] Core hygiene automation
- [x] Security workflow deployment
- [x] SBOM generation
- [x] Metrics collection
- [x] CLI interface

### Planned Features (v1.1)
- [ ] Custom configuration profiles
- [ ] Organization-wide deployment
- [ ] Slack/Teams notifications
- [ ] Integration with existing CI/CD

### Future Enhancements (v2.0)
- [ ] AI-powered commit message generation
- [ ] Automated vulnerability remediation
- [ ] Policy compliance checking
- [ ] Advanced metrics and reporting

## 🛠️ Configuration

The bot uses `config/repo-hygiene.yaml` for customization:

```yaml
# Standard topics to apply
metadata:
  standard_topics:
    - "llmops"
    - "rag"
    - "semantic-release"
    - "sbom" 
    - "github-actions"

# Security scanning settings
security:
  required_workflows:
    - ".github/workflows/codeql.yml"
    - ".github/workflows/scorecard.yml"
    - ".github/workflows/sbom.yml"

# Pull request behavior
pull_requests:
  title: "✨ repo‑hygiene‑bot update"
  auto_assign: true
  max_prs_per_run: 10
```

## 📊 Metrics and Reporting

### Compliance Dashboard
The bot generates detailed compliance metrics:

```bash
🎯 Repository Hygiene Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Overall Statistics:
   Total repositories processed: 25
   Fully compliant repositories: 18
   Repositories needing improvements: 7
   Overall compliance rate: 72.0%

🔍 Issues Found:
   Missing descriptions: 2
   Missing licenses: 3
   Missing CodeQL scanning: 5
   Missing Dependabot: 4
   Insufficient topics (<5): 6
   Missing SBOM workflow: 7
```

### GitHub Actions Integration
To enable automated weekly runs, manually create `.github/workflows/repo-hygiene-bot.yml`:

```yaml
name: Repository Hygiene Bot
on:
  schedule:
    - cron: '0 2 * * 0'  # Sundays at 2 AM UTC
  workflow_dispatch:
permissions:
  contents: write
  pull-requests: write
jobs:
  hygiene-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: |
          pip install requests PyYAML
          python -m src.repo_hygiene_bot
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          GITHUB_USER: ${{ github.repository_owner }}
```

Features include:
- Compliance reporting in GitHub Actions summary
- Artifact upload for metrics history
- Failure notifications via GitHub Issues

## 🚀 Deployment Options

### Manual Execution
```bash
# One-time run
./scripts/run-hygiene-bot.sh --user myuser --token $GITHUB_TOKEN
```

### Scheduled GitHub Actions
Create the workflow file manually (see GitHub Actions Integration section above) to enable:
- Weekly automated runs on Sundays at 2 AM UTC
- Manual triggering via workflow_dispatch
- Automated reporting and metrics collection

### CI/CD Integration
```bash
# Add to your deployment pipeline
- name: Repository Hygiene Check
  run: python -m src.repo_hygiene_cli --dry-run
```

## 🔒 Security Considerations

### Token Permissions
Required GitHub token scopes:
- `repo` (full repository access)
- `workflow` (manage GitHub Actions)

### Safe Operation
- **Idempotent**: Safe to run multiple times
- **Branch-based**: Changes via pull requests only
- **Dry-run mode**: Preview changes before applying
- **Rate limiting**: Respects GitHub API limits

### Audit Trail
- All changes tracked in pull request history
- Metrics provide compliance visibility
- Detailed logging for troubleshooting

## 🤝 Contributing

### Development Setup
```bash
git clone <repository>
cd repository
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Running Tests
```bash
python -m pytest tests/test_repo_hygiene_bot.py -v
```

### Code Standards
- Follow existing code style
- Add tests for new features
- Update documentation
- Run security scans: `bandit -r src/`

## 📜 License

Apache License 2.0 - see LICENSE file for details.

## 🆘 Support

### Troubleshooting
- Check GitHub token permissions
- Verify repository access
- Review logs in `repo-hygiene-bot.log`

### Common Issues
1. **Rate limiting**: Reduce `max_prs_per_run` in config
2. **Permission errors**: Ensure token has `repo` scope
3. **Branch conflicts**: Bot creates unique branch names

### Getting Help
- Review logs for error details
- Check GitHub API status
- Validate configuration syntax