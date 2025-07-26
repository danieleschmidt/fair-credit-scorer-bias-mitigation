# GitHub Actions Setup for Repository Hygiene Bot

Due to GitHub App permissions, the workflow file must be created manually.

## Create Workflow File

Create `.github/workflows/repo-hygiene-bot.yml` with the following content:

```yaml
name: Repository Hygiene Bot

on:
  schedule:
    # Run weekly on Sundays at 2 AM UTC
    - cron: '0 2 * * 0'
  workflow_dispatch:
    # Allow manual triggering
    inputs:
      dry_run:
        description: 'Run in dry-run mode (no changes made)'
        required: false
        default: 'false'
        type: boolean

permissions:
  contents: write
  pull-requests: write
  security-events: write
  actions: read

jobs:
  hygiene-check:
    name: Repository Hygiene Check
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
          
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'
          
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install requests PyYAML
          
      - name: Run Repository Hygiene Bot
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          GITHUB_USER: ${{ github.repository_owner }}
          DRY_RUN: ${{ inputs.dry_run || 'false' }}
        run: |
          python -m src.repo_hygiene_bot
          
      - name: Upload metrics artifact
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: hygiene-metrics
          path: metrics/profile_hygiene.json
          retention-days: 30
          
      - name: Create hygiene report
        if: always()
        run: |
          echo "## 🤖 Repository Hygiene Report" >> $GITHUB_STEP_SUMMARY
          echo "" >> $GITHUB_STEP_SUMMARY
          echo "**Date:** $(date -u)" >> $GITHUB_STEP_SUMMARY
          echo "" >> $GITHUB_STEP_SUMMARY
          
          if [ -f "metrics/profile_hygiene.json" ]; then
            echo "**Metrics collected for $(jq length metrics/profile_hygiene.json) repositories**" >> $GITHUB_STEP_SUMMARY
            echo "" >> $GITHUB_STEP_SUMMARY
            echo "### Summary" >> $GITHUB_STEP_SUMMARY
            echo "" >> $GITHUB_STEP_SUMMARY
            echo "| Metric | Count |" >> $GITHUB_STEP_SUMMARY
            echo "|--------|-------|" >> $GITHUB_STEP_SUMMARY
            echo "| Repositories with descriptions | $(jq '[.[] | select(.description_set == true)] | length' metrics/profile_hygiene.json) |" >> $GITHUB_STEP_SUMMARY
            echo "| Repositories with licenses | $(jq '[.[] | select(.license_exists == true)] | length' metrics/profile_hygiene.json) |" >> $GITHUB_STEP_SUMMARY
            echo "| Repositories with CodeQL | $(jq '[.[] | select(.code_scanning == true)] | length' metrics/profile_hygiene.json) |" >> $GITHUB_STEP_SUMMARY
            echo "| Repositories with Dependabot | $(jq '[.[] | select(.dependabot == true)] | length' metrics/profile_hygiene.json) |" >> $GITHUB_STEP_SUMMARY
            echo "| Repositories with SBOM | $(jq '[.[] | select(.sbom_workflow == true)] | length' metrics/profile_hygiene.json) |" >> $GITHUB_STEP_SUMMARY
          else
            echo "❌ No metrics file generated" >> $GITHUB_STEP_SUMMARY
          fi
          
      - name: Notify on failure
        if: failure()
        uses: actions/github-script@v7
        with:
          script: |
            github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: '🚨 Repository Hygiene Bot Failed',
              body: `The automated repository hygiene check failed.
              
              **Run Details:**
              - Workflow: ${context.workflow}
              - Run ID: ${context.runId}
              - Run URL: ${context.payload.repository.html_url}/actions/runs/${context.runId}
              
              Please check the logs and fix any issues.`,
              labels: ['bug', 'automated-maintenance']
            });
```

## Features

This workflow provides:

- **Scheduled Execution**: Runs every Sunday at 2 AM UTC
- **Manual Triggering**: Can be run manually with optional dry-run mode
- **Metrics Collection**: Uploads compliance metrics as artifacts
- **Reporting**: Creates detailed reports in GitHub Actions summary
- **Error Handling**: Creates GitHub issues when the bot fails

## Setup Instructions

1. Create the `.github/workflows/` directory if it doesn't exist
2. Create the file `repo-hygiene-bot.yml` with the content above
3. Commit and push the workflow file
4. The bot will run automatically every Sunday
5. You can also trigger it manually from the Actions tab

## Manual Triggering

To run the bot manually:

1. Go to the Actions tab in your repository
2. Select "Repository Hygiene Bot" workflow
3. Click "Run workflow"
4. Optionally check "Run in dry-run mode" to preview changes
5. Click "Run workflow" to start

The dry-run mode is useful for testing and seeing what changes would be made without actually applying them.