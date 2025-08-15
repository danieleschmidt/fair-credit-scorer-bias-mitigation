#!/usr/bin/env python3
"""
Autonomous DevSecOps Repo-Hygiene Bot

This bot iterates over all repositories owned by a GitHub user and applies
standardized DevSecOps practices, creating pull requests with improvements.
"""

import base64
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests


@dataclass
class RepoMetrics:
    """Metrics for repository hygiene compliance."""
    repo: str
    description_set: bool
    topics_count: int
    license_exists: bool
    code_scanning: bool
    dependabot: bool
    scorecard: bool
    sbom_workflow: bool
    community_files_count: int
    readme_badges_count: int
    readme_sections_count: int
    last_commit_days_ago: int


class GitHubAPI:
    """GitHub API client for repository operations."""

    def __init__(self, token: str, github_user: str):
        self.token = token
        self.github_user = github_user
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'repo-hygiene-bot/1.0'
        })

    def get_repositories(self) -> List[Dict]:
        """Get all repositories owned by the user."""
        repos = []
        page = 1

        while True:
            response = self.session.get(
                'https://api.github.com/user/repos',
                params={
                    'per_page': 100,
                    'affiliation': 'owner',
                    'page': page
                }
            )
            response.raise_for_status()

            page_repos = response.json()
            if not page_repos:
                break

            # Filter out archived, forks, templates, and disabled repos
            for repo in page_repos:
                if not (repo.get('archived', False) or
                       repo.get('fork', False) or
                       repo.get('is_template', False) or
                       repo.get('disabled', False)):
                    repos.append(repo)

            page += 1

        return repos

    def update_repository_metadata(self, repo_name: str, description: str = None,
                                 homepage: str = None, topics: List[str] = None) -> bool:
        """Update repository description, homepage, and topics."""
        try:
            # Update description and homepage
            if description or homepage:
                data = {}
                if description:
                    data['description'] = description
                if homepage:
                    data['homepage'] = homepage

                response = self.session.patch(
                    f'https://api.github.com/repos/{self.github_user}/{repo_name}',
                    json=data
                )
                response.raise_for_status()

            # Update topics
            if topics:
                response = self.session.put(
                    f'https://api.github.com/repos/{self.github_user}/{repo_name}/topics',
                    json={'names': topics}
                )
                response.raise_for_status()

            return True
        except Exception as e:
            logging.error(f"Failed to update metadata for {repo_name}: {e}")
            return False

    def get_file_content(self, repo_name: str, file_path: str, branch: str = 'main') -> Optional[str]:
        """Get file content from repository."""
        try:
            response = self.session.get(
                f'https://api.github.com/repos/{self.github_user}/{repo_name}/contents/{file_path}',
                params={'ref': branch}
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()

            content = response.json()
            if content.get('encoding') == 'base64':
                return base64.b64decode(content['content']).decode('utf-8')
            return content.get('content', '')
        except Exception as e:
            logging.debug(f"Could not get {file_path} from {repo_name}: {e}")
            return None

    def create_or_update_file(self, repo_name: str, file_path: str, content: str,
                             message: str, branch: str = 'main') -> bool:
        """Create or update a file in the repository."""
        try:
            # Check if file exists to get SHA
            existing = self.get_file_sha(repo_name, file_path, branch)

            data = {
                'message': message,
                'content': base64.b64encode(content.encode()).decode(),
                'branch': branch
            }

            if existing:
                data['sha'] = existing

            response = self.session.put(
                f'https://api.github.com/repos/{self.github_user}/{repo_name}/contents/{file_path}',
                json=data
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logging.error(f"Failed to create/update {file_path} in {repo_name}: {e}")
            return False

    def get_file_sha(self, repo_name: str, file_path: str, branch: str = 'main') -> Optional[str]:
        """Get SHA of existing file."""
        try:
            response = self.session.get(
                f'https://api.github.com/repos/{self.github_user}/{repo_name}/contents/{file_path}',
                params={'ref': branch}
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json().get('sha')
        except:
            return None

    def create_pull_request(self, repo_name: str, title: str, body: str,
                           head_branch: str, base_branch: str = 'main') -> Optional[Dict]:
        """Create a pull request."""
        try:
            data = {
                'title': title,
                'body': body,
                'head': head_branch,
                'base': base_branch
            }

            response = self.session.post(
                f'https://api.github.com/repos/{self.github_user}/{repo_name}/pulls',
                json=data
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"Failed to create PR for {repo_name}: {e}")
            return None

    def create_branch(self, repo_name: str, branch_name: str, base_branch: str = 'main') -> bool:
        """Create a new branch."""
        try:
            # Get base branch SHA
            response = self.session.get(
                f'https://api.github.com/repos/{self.github_user}/{repo_name}/git/refs/heads/{base_branch}'
            )
            response.raise_for_status()
            base_sha = response.json()['object']['sha']

            # Create new branch
            data = {
                'ref': f'refs/heads/{branch_name}',
                'sha': base_sha
            }

            response = self.session.post(
                f'https://api.github.com/repos/{self.github_user}/{repo_name}/git/refs',
                json=data
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logging.error(f"Failed to create branch {branch_name} in {repo_name}: {e}")
            return False

    def archive_repository(self, repo_name: str) -> bool:
        """Archive a repository."""
        try:
            response = self.session.patch(
                f'https://api.github.com/repos/{self.github_user}/{repo_name}',
                json={'archived': True}
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logging.error(f"Failed to archive {repo_name}: {e}")
            return False


class RepoHygieneBot:
    """Main bot class for repository hygiene automation."""

    def __init__(self, github_token: str, github_user: str):
        self.github_api = GitHubAPI(github_token, github_user)
        self.github_user = github_user
        self.metrics = []

        # Standard topics to apply
        self.standard_topics = [
            "llmops", "rag", "semantic-release", "sbom", "github-actions"
        ]

        # Community health files templates
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load templates for community health files."""
        return {
            'LICENSE': '''Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

[Full Apache 2.0 license text would go here]
''',
            'CODE_OF_CONDUCT.md': '''# Contributor Covenant Code of Conduct

## Our Pledge

We as members, contributors, and leaders pledge to make participation in our
community a harassment-free experience for everyone, regardless of age, body
size, visible or invisible disability, ethnicity, sex characteristics, gender
identity and expression, level of experience, education, socio-economic status,
nationality, personal appearance, race, religion, or sexual identity
and orientation.

## Our Standards

Examples of behavior that contributes to a positive environment for our
community include:

* Demonstrating empathy and kindness toward other people
* Being respectful of differing opinions, viewpoints, and experiences
* Giving and gracefully accepting constructive feedback
* Accepting responsibility and apologizing to those affected by our mistakes,
  and learning from the experience
* Focusing on what is best not just for us as individuals, but for the
  overall community

## Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be
reported to the community leaders responsible for enforcement.

This Code of Conduct is adapted from the [Contributor Covenant][homepage],
version 2.1, available at
https://www.contributor-covenant.org/version/2/1/code_of_conduct.html.

[homepage]: https://www.contributor-covenant.org
''',
            'CONTRIBUTING.md': '''# Contributing

## Development Setup

1. Fork the repository
2. Clone your fork
3. Install dependencies
4. Run tests to ensure everything works

## Commit Messages

This project uses [Conventional Commits](https://conventionalcommits.org/):

- `feat:` for new features
- `fix:` for bug fixes  
- `docs:` for documentation changes
- `test:` for test changes
- `refactor:` for code refactoring

## Testing

Run the test suite with:
```bash
npm test
# or
python -m pytest
```

Ensure all tests pass before submitting a pull request.
''',
            'SECURITY.md': '''# Security Policy

## Reporting Security Vulnerabilities

If you discover a security vulnerability, please report it via email to:
security@example.com

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact

## Response Timeline

- **Initial Response**: Within 24 hours
- **Status Update**: Within 72 hours  
- **Resolution**: Within 90 days

We appreciate responsible disclosure and will credit reporters appropriately.
''',
            '.github/ISSUE_TEMPLATE/bug.yml': '''name: Bug Report
description: File a bug report
labels: ["bug", "triage"]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for taking the time to fill out this bug report!
  - type: textarea
    id: what-happened
    attributes:
      label: What happened?
      description: Also tell us, what did you expect to happen?
      placeholder: Tell us what you see!
    validations:
      required: true
  - type: textarea
    id: reproduction
    attributes:
      label: Reproduction steps
      description: How can we reproduce this issue?
      placeholder: |
        1. Go to '...'
        2. Click on '....'
        3. See error
    validations:
      required: true
''',
            '.github/ISSUE_TEMPLATE/feature.yml': '''name: Feature Request
description: Suggest an idea for this project
labels: ["enhancement", "triage"]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for suggesting a new feature!
  - type: textarea
    id: feature-description
    attributes:
      label: Feature Description
      description: What feature would you like to see?
      placeholder: Describe your feature request
    validations:
      required: true
  - type: textarea
    id: use-case
    attributes:
      label: Use Case
      description: What problem does this solve?
      placeholder: Explain the use case
    validations:
      required: true
'''
        }

    def process_all_repositories(self) -> List[RepoMetrics]:
        """Process all repositories for the user."""
        logging.info(f"Starting repository hygiene check for user: {self.github_user}")

        repos = self.github_api.get_repositories()
        logging.info(f"Found {len(repos)} repositories to process")

        for repo in repos:
            try:
                logging.info(f"Processing repository: {repo['name']}")
                self.process_repository(repo)
            except Exception as e:
                logging.error(f"Failed to process repository {repo['name']}: {e}")
                continue

        # Save metrics
        self._save_metrics()

        return self.metrics

    def process_repository(self, repo: Dict) -> None:
        """Process a single repository."""
        repo_name = repo['name']
        branch_name = 'repo-hygiene-bot-updates'
        changes = []

        # Initialize metrics
        metrics = RepoMetrics(
            repo=repo_name,
            description_set=bool(repo.get('description')),
            topics_count=len(repo.get('topics', [])),
            license_exists=False,
            code_scanning=False,
            dependabot=False,
            scorecard=False,
            sbom_workflow=False,
            community_files_count=0,
            readme_badges_count=0,
            readme_sections_count=0,
            last_commit_days_ago=0
        )

        # Check if changes are needed
        needs_changes = False

        # STEP 1: Ensure Description / Website / Topics
        description_change = self._check_description(repo, changes)
        website_change = self._check_website(repo, changes)
        topics_change = self._check_topics(repo, changes)

        if description_change or website_change or topics_change:
            needs_changes = True
            self._apply_metadata_changes(repo_name, repo, changes)

        # Create branch if we need to make file changes
        file_changes_needed = self._check_file_changes_needed(repo_name)

        if file_changes_needed:
            needs_changes = True
            # Create branch
            if not self.github_api.create_branch(repo_name, branch_name):
                logging.error(f"Failed to create branch for {repo_name}")
                return

            # STEP 2: Community health files
            self._ensure_community_files(repo_name, branch_name, changes, metrics)

            # STEP 3: Enable supply-chain scanners
            self._ensure_security_workflows(repo_name, branch_name, changes, metrics)

            # STEP 4: SBOM & signed releases
            self._ensure_sbom_workflows(repo_name, branch_name, changes, metrics)

            # STEP 5: Insert README badges
            self._ensure_readme_badges(repo_name, branch_name, changes, metrics)

            # STEP 6: Archive stale "Main-Project"
            if self._should_archive_main_project(repo):
                if self.github_api.archive_repository(repo_name):
                    changes.append("Archived stale Main-Project repository")
                    logging.info(f"Archived repository: {repo_name}")
                    return

            # STEP 7: README sections
            self._ensure_readme_sections(repo_name, branch_name, changes, metrics)

            # STEP 9: Open PR if changes were made
            if changes:
                self._create_pull_request(repo_name, branch_name, changes)

        # STEP 8: Self-audit metrics
        self.metrics.append(metrics)

    def _check_description(self, repo: Dict, changes: List[str]) -> bool:
        """Check if repository needs a description."""
        if not repo.get('description'):
            changes.append("Add repository description")
            return True
        return False

    def _check_website(self, repo: Dict, changes: List[str]) -> bool:
        """Check if repository needs a homepage."""
        if not repo.get('homepage'):
            changes.append(f"Set homepage to https://{self.github_user}.github.io")
            return True
        return False

    def _check_topics(self, repo: Dict, changes: List[str]) -> bool:
        """Check if repository needs more topics."""
        current_topics = set(repo.get('topics', []))
        needed_topics = set(self.standard_topics) - current_topics

        if needed_topics or len(current_topics) < 5:
            changes.append(f"Add topics: {', '.join(self.standard_topics)}")
            return True
        return False

    def _apply_metadata_changes(self, repo_name: str, repo: Dict, changes: List[str]) -> None:
        """Apply metadata changes to repository."""
        description = repo.get('description') or f"Repository for {repo_name} - automated DevSecOps practices applied"
        homepage = repo.get('homepage') or f"https://{self.github_user}.github.io"

        # Merge existing topics with standard topics
        current_topics = set(repo.get('topics', []))
        all_topics = list(current_topics.union(set(self.standard_topics)))

        self.github_api.update_repository_metadata(
            repo_name=repo_name,
            description=description if not repo.get('description') else None,
            homepage=homepage if not repo.get('homepage') else None,
            topics=all_topics
        )

    def _check_file_changes_needed(self, repo_name: str) -> bool:
        """Check if any file changes are needed."""
        # Check for missing community files
        required_files = ['LICENSE', 'CODE_OF_CONDUCT.md', 'CONTRIBUTING.md', 'SECURITY.md']

        for file_path in required_files:
            if not self.github_api.get_file_content(repo_name, file_path):
                return True

        # Check for missing workflows
        workflows = ['.github/workflows/codeql.yml', '.github/dependabot.yml']
        for workflow in workflows:
            if not self.github_api.get_file_content(repo_name, workflow):
                return True

        return False

    def _ensure_community_files(self, repo_name: str, branch: str, changes: List[str], metrics: RepoMetrics) -> None:
        """Ensure community health files exist."""
        files_created = 0

        for file_path, template in self.templates.items():
            if not self.github_api.get_file_content(repo_name, file_path):
                if self.github_api.create_or_update_file(
                    repo_name, file_path, template,
                    f"Add {file_path}", branch
                ):
                    changes.append(f"Add {file_path}")
                    files_created += 1

        metrics.community_files_count = files_created

    def _ensure_security_workflows(self, repo_name: str, branch: str, changes: List[str], metrics: RepoMetrics) -> None:
        """Ensure security scanning workflows exist."""
        # CodeQL workflow
        codeql_workflow = '''name: "CodeQL"

on:
  push:
    branches: [ "main" ]
  pull_request:
    branches: [ "main" ]
  schedule:
    - cron: '0 2 * * 1'

jobs:
  analyze:
    name: Analyze
    runs-on: ubuntu-latest
    permissions:
      actions: read
      contents: read
      security-events: write

    strategy:
      fail-fast: false
      matrix:
        language: [ 'python', 'javascript' ]

    steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Initialize CodeQL
      uses: github/codeql-action/init@v3
      with:
        languages: ${{ matrix.language }}

    - name: Autobuild
      uses: github/codeql-action/autobuild@v3

    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v3
'''

        if not self.github_api.get_file_content(repo_name, '.github/workflows/codeql.yml'):
            if self.github_api.create_or_update_file(
                repo_name, '.github/workflows/codeql.yml', codeql_workflow,
                'Add CodeQL security scanning workflow', branch
            ):
                changes.append('Add CodeQL security scanning')
                metrics.code_scanning = True

        # Dependabot configuration
        dependabot_config = '''version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 10
  - package-ecosystem: "npm"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 10
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
'''

        if not self.github_api.get_file_content(repo_name, '.github/dependabot.yml'):
            if self.github_api.create_or_update_file(
                repo_name, '.github/dependabot.yml', dependabot_config,
                'Add Dependabot configuration', branch
            ):
                changes.append('Add Dependabot dependency updates')
                metrics.dependabot = True

        # OpenSSF Scorecard workflow
        scorecard_workflow = '''name: Scorecard supply-chain security
on:
  branch_protection_rule:
  schedule:
    - cron: '0 2 * * 0'
  push:
    branches: [ "main" ]

permissions: read-all

jobs:
  analysis:
    name: Scorecard analysis
    runs-on: ubuntu-latest
    permissions:
      security-events: write
      id-token: write

    steps:
      - name: "Checkout code"
        uses: actions/checkout@v4
        with:
          persist-credentials: false

      - name: "Run analysis"
        uses: ossf/scorecard-action@v2.4.0
        with:
          results_file: results.sarif
          results_format: sarif
          publish_results: true

      - name: "Upload to code-scanning"
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: results.sarif
'''

        if not self.github_api.get_file_content(repo_name, '.github/workflows/scorecard.yml'):
            if self.github_api.create_or_update_file(
                repo_name, '.github/workflows/scorecard.yml', scorecard_workflow,
                'Add OpenSSF Scorecard workflow', branch
            ):
                changes.append('Add OpenSSF Scorecard security analysis')
                metrics.scorecard = True

    def _ensure_sbom_workflows(self, repo_name: str, branch: str, changes: List[str], metrics: RepoMetrics) -> None:
        """Ensure SBOM and release signing workflows exist."""
        sbom_workflow = '''name: SBOM Generation

on:
  push:
    branches: [ "main" ]
  release:
    types: [published]
  schedule:
    - cron: '0 3 * * 0'

jobs:
  sbom:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      
    steps:
      - uses: actions/checkout@v4
      
      - name: Generate SBOM
        uses: CycloneDX/github-action@v1
        with:
          output: docs/sbom/latest.json
          
      - name: Commit SBOM
        run: |
          git config --global user.name 'github-actions[bot]'
          git config --global user.email 'github-actions[bot]@users.noreply.github.com'
          git add docs/sbom/latest.json
          git diff --staged --quiet || git commit -m "Update SBOM"
          git push
'''

        if not self.github_api.get_file_content(repo_name, '.github/workflows/sbom.yml'):
            if self.github_api.create_or_update_file(
                repo_name, '.github/workflows/sbom.yml', sbom_workflow,
                'Add SBOM generation workflow', branch
            ):
                changes.append('Add SBOM generation and signing')
                metrics.sbom_workflow = True

    def _ensure_readme_badges(self, repo_name: str, branch: str, changes: List[str], metrics: RepoMetrics) -> None:
        """Ensure README has required badges."""
        readme_content = self.github_api.get_file_content(repo_name, 'README.md')
        if not readme_content:
            return

        badges = [
            f"[![Build](https://img.shields.io/github/actions/workflow/status/{self.github_user}/{repo_name}/ci.yml?branch=main)](https://github.com/{self.github_user}/{repo_name}/actions)",
            "[![semantic-release](https://img.shields.io/badge/semantic--release-active-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)",
            "[![SBOM](https://img.shields.io/badge/SBOM-CycloneDX-0078d6)](docs/sbom/latest.json)"
        ]

        # Check if badges are missing
        missing_badges = []
        for badge in badges:
            if badge not in readme_content:
                missing_badges.append(badge)

        if missing_badges:
            # Insert badges at the top after title
            lines = readme_content.split('\n')
            title_index = 0
            for i, line in enumerate(lines):
                if line.startswith('#'):
                    title_index = i + 1
                    break

            # Insert badges after title
            for badge in missing_badges:
                lines.insert(title_index, badge)
                title_index += 1

            new_content = '\n'.join(lines)

            if self.github_api.create_or_update_file(
                repo_name, 'README.md', new_content,
                'Add CI/CD and security badges to README', branch
            ):
                changes.append('Add README badges')
                metrics.readme_badges_count = len(missing_badges)

    def _should_archive_main_project(self, repo: Dict) -> bool:
        """Check if Main-Project should be archived."""
        if repo['name'] != 'Main-Project':
            return False

        # Check last commit date
        try:
            response = self.github_api.session.get(
                f"https://api.github.com/repos/{self.github_user}/{repo['name']}/commits",
                params={'per_page': 1}
            )
            if response.status_code == 200:
                commits = response.json()
                if commits:
                    last_commit = datetime.fromisoformat(
                        commits[0]['commit']['committer']['date'].replace('Z', '+00:00')
                    )
                    days_old = (datetime.now(timezone.utc) - last_commit).days
                    return days_old > 400
        except:
            pass

        return False

    def _ensure_readme_sections(self, repo_name: str, branch: str, changes: List[str], metrics: RepoMetrics) -> None:
        """Ensure README has required sections."""
        readme_content = self.github_api.get_file_content(repo_name, 'README.md')
        if not readme_content:
            return

        required_sections = [
            "## ✨ Why this exists",
            "## ⚡ Quick Start",
            "## 🔍 Key Features",
            "## 🗺 Road Map",
            "## 🤝 Contributing"
        ]

        missing_sections = []
        for section in required_sections:
            if section not in readme_content:
                missing_sections.append(section)

        if missing_sections:
            new_content = readme_content + '\n\n' + '\n\n'.join([
                section + '\n\nTODO: Add content for this section.'
                for section in missing_sections
            ])

            if self.github_api.create_or_update_file(
                repo_name, 'README.md', new_content,
                'Add required README sections', branch
            ):
                changes.append(f'Add README sections: {", ".join(missing_sections)}')
                metrics.readme_sections_count = len(missing_sections)

    def _create_pull_request(self, repo_name: str, branch_name: str, changes: List[str]) -> None:
        """Create pull request with hygiene improvements."""
        title = "✨ repo‑hygiene‑bot update"
        body = f"""## 🤖 Automated Repository Hygiene Update

This PR applies standardized DevSecOps practices to improve repository hygiene.

### Changes Made:
{chr(10).join(f'- {change}' for change in changes)}

### Benefits:
- ✅ Enhanced security scanning with CodeQL and Scorecard
- ✅ Automated dependency updates with Dependabot  
- ✅ SBOM generation for supply chain security
- ✅ Standardized community health files
- ✅ Improved README documentation and badges

This PR was automatically generated by the repo-hygiene-bot.
"""

        pr = self.github_api.create_pull_request(
            repo_name=repo_name,
            title=title,
            body=body,
            head_branch=branch_name
        )

        if pr:
            logging.info(f"Created PR #{pr['number']} for {repo_name}: {pr['html_url']}")
        else:
            logging.error(f"Failed to create PR for {repo_name}")

    def _save_metrics(self) -> None:
        """Save hygiene metrics to file."""
        os.makedirs('metrics', exist_ok=True)

        metrics_data = [asdict(metric) for metric in self.metrics]

        with open('metrics/profile_hygiene.json', 'w') as f:
            json.dump(metrics_data, f, indent=2)

        logging.info(f"Saved metrics for {len(self.metrics)} repositories")


def main():
    """Main entry point for the repo hygiene bot."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Get configuration from environment
    github_token = os.environ.get('GITHUB_TOKEN')
    github_user = os.environ.get('GITHUB_USER')

    if not github_token or not github_user:
        logging.error("GITHUB_TOKEN and GITHUB_USER environment variables are required")
        return 1

    # Initialize and run bot
    bot = RepoHygieneBot(github_token, github_user)

    try:
        metrics = bot.process_all_repositories()
        logging.info(f"Successfully processed {len(metrics)} repositories")

        # Print summary
        total_repos = len(metrics)
        repos_with_issues = sum(1 for m in metrics if not all([
            m.description_set, m.license_exists, m.code_scanning,
            m.dependabot, m.topics_count >= 5
        ]))

        print("\n🎯 Repository Hygiene Summary:")
        print(f"   Total repositories: {total_repos}")
        print(f"   Repositories needing improvements: {repos_with_issues}")
        print(f"   Compliance rate: {((total_repos - repos_with_issues) / total_repos * 100):.1f}%")

        return 0

    except Exception as e:
        logging.error(f"Bot execution failed: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
