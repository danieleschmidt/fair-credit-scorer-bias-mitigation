#!/usr/bin/env python3
"""
CLI for the Repository Hygiene Bot

Provides command-line interface to run the repo hygiene bot with various options.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from .repo_hygiene_bot import RepoHygieneBot


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('repo-hygiene-bot.log')
        ]
    )


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Autonomous DevSecOps Repository Hygiene Bot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --user myuser --token $GITHUB_TOKEN
  %(prog)s --dry-run --verbose
  %(prog)s --single-repo myrepo
        """
    )
    
    parser.add_argument(
        '--token', 
        help='GitHub personal access token (or set GITHUB_TOKEN env var)',
        default=os.environ.get('GITHUB_TOKEN')
    )
    
    parser.add_argument(
        '--user',
        help='GitHub username (or set GITHUB_USER env var)', 
        default=os.environ.get('GITHUB_USER')
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )
    
    parser.add_argument(
        '--single-repo',
        help='Process only a single repository by name'
    )
    
    parser.add_argument(
        '--exclude',
        nargs='+',
        help='Repository names to exclude from processing'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--config',
        help='Path to configuration file',
        default='config/repo-hygiene.yaml'
    )
    
    parser.add_argument(
        '--metrics-dir',
        help='Directory to save metrics',
        default='metrics'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Validate required arguments
    if not args.token:
        logging.error("GitHub token is required. Set GITHUB_TOKEN env var or use --token")
        return 1
        
    if not args.user:
        logging.error("GitHub user is required. Set GITHUB_USER env var or use --user")
        return 1
    
    # Create metrics directory
    Path(args.metrics_dir).mkdir(exist_ok=True)
    
    # Initialize bot
    try:
        bot = RepoHygieneBot(args.token, args.user)
        
        if args.dry_run:
            logging.info("🔍 Running in DRY-RUN mode - no changes will be made")
            bot.dry_run = True
        
        # Process repositories
        if args.single_repo:
            logging.info(f"Processing single repository: {args.single_repo}")
            # Get specific repo info
            repos = bot.github_api.get_repositories()
            target_repo = next((r for r in repos if r['name'] == args.single_repo), None)
            
            if not target_repo:
                logging.error(f"Repository '{args.single_repo}' not found")
                return 1
                
            bot.process_repository(target_repo)
        else:
            # Process all repositories
            logging.info("Processing all repositories")
            metrics = bot.process_all_repositories()
            
            # Print summary
            print_summary(metrics, args.exclude or [])
        
        logging.info("✅ Repository hygiene bot completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logging.info("❌ Process interrupted by user")
        return 1
    except Exception as e:
        logging.error(f"❌ Bot execution failed: {e}")
        if args.verbose:
            logging.exception("Full error details:")
        return 1


def print_summary(metrics, excluded_repos):
    """Print execution summary."""
    total_repos = len(metrics)
    
    if total_repos == 0:
        print("\n📋 No repositories processed")
        return
    
    # Calculate compliance metrics
    compliant_repos = 0
    issues_found = {
        'missing_description': 0,
        'missing_license': 0,
        'missing_code_scanning': 0,
        'missing_dependabot': 0,
        'insufficient_topics': 0,
        'missing_sbom': 0
    }
    
    for metric in metrics:
        is_compliant = all([
            metric.description_set,
            metric.license_exists,
            metric.code_scanning,
            metric.dependabot,
            metric.topics_count >= 5,
            metric.sbom_workflow
        ])
        
        if is_compliant:
            compliant_repos += 1
        else:
            if not metric.description_set:
                issues_found['missing_description'] += 1
            if not metric.license_exists:
                issues_found['missing_license'] += 1
            if not metric.code_scanning:
                issues_found['missing_code_scanning'] += 1
            if not metric.dependabot:
                issues_found['missing_dependabot'] += 1
            if metric.topics_count < 5:
                issues_found['insufficient_topics'] += 1
            if not metric.sbom_workflow:
                issues_found['missing_sbom'] += 1
    
    compliance_rate = (compliant_repos / total_repos) * 100
    
    print(f"""
🎯 Repository Hygiene Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Overall Statistics:
   Total repositories processed: {total_repos}
   Fully compliant repositories: {compliant_repos}
   Repositories needing improvements: {total_repos - compliant_repos}
   Overall compliance rate: {compliance_rate:.1f}%

🔍 Issues Found:
   Missing descriptions: {issues_found['missing_description']}
   Missing licenses: {issues_found['missing_license']}
   Missing CodeQL scanning: {issues_found['missing_code_scanning']}
   Missing Dependabot: {issues_found['missing_dependabot']}
   Insufficient topics (<5): {issues_found['insufficient_topics']}
   Missing SBOM workflow: {issues_found['missing_sbom']}

📁 Metrics saved to: metrics/profile_hygiene.json
""")
    
    if excluded_repos:
        print(f"🚫 Excluded repositories: {', '.join(excluded_repos)}")


if __name__ == '__main__':
    sys.exit(main())