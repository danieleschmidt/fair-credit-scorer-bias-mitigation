#!/usr/bin/env python3
"""
Automated release management and deployment script.

This script handles version bumping, changelog generation, tag creation,
and release preparation for the fair credit scorer project.
"""

import re
import os
import json
import subprocess
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import sys
import toml


class ReleaseManager:
    """Automated release management system."""
    
    def __init__(self, repo_path: str = "."):
        """Initialize release manager."""
        self.repo_path = Path(repo_path)
        self.pyproject_path = self.repo_path / "pyproject.toml"
        self.changelog_path = self.repo_path / "CHANGELOG.md"
        
    def get_current_version(self) -> str:
        """Get current version from pyproject.toml."""
        if not self.pyproject_path.exists():
            raise FileNotFoundError("pyproject.toml not found")
        
        with open(self.pyproject_path, 'r') as f:
            config = toml.load(f)
        
        return config.get("project", {}).get("version", "0.0.0")
    
    def bump_version(self, bump_type: str) -> str:
        """Bump version according to semantic versioning."""
        current = self.get_current_version()
        major, minor, patch = map(int, current.split('.'))
        
        if bump_type == "major":
            major += 1
            minor = 0
            patch = 0
        elif bump_type == "minor":
            minor += 1
            patch = 0
        elif bump_type == "patch":
            patch += 1
        else:
            raise ValueError(f"Invalid bump type: {bump_type}")
        
        new_version = f"{major}.{minor}.{patch}"
        self._update_version_in_files(new_version)
        return new_version
    
    def _update_version_in_files(self, new_version: str):
        """Update version in relevant files."""
        # Update pyproject.toml
        with open(self.pyproject_path, 'r') as f:
            content = f.read()
        
        content = re.sub(
            r'version = "[^"]*"',
            f'version = "{new_version}"',
            content
        )
        
        with open(self.pyproject_path, 'w') as f:
            f.write(content)
        
        # Update README.md version badge if present
        readme_path = self.repo_path / "README.md"
        if readme_path.exists():
            with open(readme_path, 'r') as f:
                readme_content = f.read()
            
            # Update version in README
            readme_content = re.sub(
                r'\*\*Version [^*]*\*\*',
                f'**Version {new_version}**',
                readme_content
            )
            
            with open(readme_path, 'w') as f:
                f.write(readme_content)
    
    def generate_changelog_entry(self, version: str, changes: List[str]) -> str:
        """Generate changelog entry for new version."""
        date = datetime.now().strftime("%Y-%m-%d")
        
        entry = [
            f"## [{version}] - {date}",
            ""
        ]
        
        # Categorize changes
        categories = {
            "### Added": [],
            "### Changed": [],
            "### Fixed": [],
            "### Security": [],
            "### Deprecated": [],
            "### Removed": []
        }
        
        for change in changes:
            if any(keyword in change.lower() for keyword in ["add", "new", "introduce"]):
                categories["### Added"].append(change)
            elif any(keyword in change.lower() for keyword in ["fix", "resolve", "correct"]):
                categories["### Fixed"].append(change)
            elif any(keyword in change.lower() for keyword in ["security", "vulnerability", "cve"]):
                categories["### Security"].append(change)
            elif any(keyword in change.lower() for keyword in ["deprecate", "obsolete"]):
                categories["### Deprecated"].append(change)
            elif any(keyword in change.lower() for keyword in ["remove", "delete"]):
                categories["### Removed"].append(change)
            else:
                categories["### Changed"].append(change)
        
        # Add non-empty categories
        for category, items in categories.items():
            if items:
                entry.append(category)
                for item in items:
                    entry.append(f"- {item}")
                entry.append("")
        
        return "\n".join(entry)
    
    def update_changelog(self, version: str, changes: List[str]):
        """Update CHANGELOG.md with new version entry."""
        new_entry = self.generate_changelog_entry(version, changes)
        
        if not self.changelog_path.exists():
            # Create new changelog
            changelog_content = f"""# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

{new_entry}
"""
        else:
            # Update existing changelog
            with open(self.changelog_path, 'r') as f:
                content = f.read()
            
            # Insert new entry after the header
            lines = content.split('\n')
            header_end = 0
            for i, line in enumerate(lines):
                if line.startswith('## [') or line.startswith('## v'):
                    header_end = i
                    break
            
            if header_end == 0:
                # No previous entries, add after header
                for i, line in enumerate(lines):
                    if line.strip() == "" and i > 3:  # After title and description
                        header_end = i + 1
                        break
            
            lines.insert(header_end, new_entry)
            changelog_content = '\n'.join(lines)
        
        with open(self.changelog_path, 'w') as f:
            f.write(changelog_content)
    
    def get_git_changes_since_tag(self, since_tag: Optional[str] = None) -> List[str]:
        """Get git commit messages since last tag."""
        if since_tag is None:
            # Get latest tag
            try:
                result = subprocess.run(
                    ["git", "describe", "--tags", "--abbrev=0"],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True,
                    check=True
                )
                since_tag = result.stdout.strip()
            except subprocess.CalledProcessError:
                # No tags found, get all commits
                since_tag = None
        
        # Get commit messages
        if since_tag:
            cmd = ["git", "log", f"{since_tag}..HEAD", "--oneline", "--no-merges"]
        else:
            cmd = ["git", "log", "--oneline", "--no-merges", "-10"]  # Last 10 commits
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            commits = result.stdout.strip().split('\n')
            return [commit.split(' ', 1)[1] for commit in commits if commit.strip()]
        except subprocess.CalledProcessError:
            return []
    
    def create_git_tag(self, version: str, message: Optional[str] = None):
        """Create and push git tag for release."""
        tag_name = f"v{version}"
        tag_message = message or f"Release version {version}"
        
        # Create annotated tag
        subprocess.run([
            "git", "tag", "-a", tag_name, "-m", tag_message
        ], cwd=self.repo_path, check=True)
        
        print(f"Created tag: {tag_name}")
        
        # Push tag
        try:
            subprocess.run([
                "git", "push", "origin", tag_name
            ], cwd=self.repo_path, check=True)
            print(f"Pushed tag: {tag_name}")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to push tag: {e}")
    
    def build_package(self) -> Path:
        """Build Python package for distribution."""
        print("Building package...")
        
        # Clean previous builds
        dist_dir = self.repo_path / "dist"
        if dist_dir.exists():
            import shutil
            shutil.rmtree(dist_dir)
        
        # Build package
        subprocess.run([
            "python", "-m", "build"
        ], cwd=self.repo_path, check=True)
        
        return dist_dir
    
    def generate_release_notes(self, version: str, changes: List[str]) -> str:
        """Generate release notes for GitHub release."""
        notes = [
            f"# Release {version}",
            "",
            "## What's Changed",
            ""
        ]
        
        if changes:
            for change in changes:
                notes.append(f"- {change}")
        else:
            notes.append("- Bug fixes and improvements")
        
        notes.extend([
            "",
            "## Installation",
            "",
            "```bash",
            f"pip install fair-credit-scorer-bias-mitigation=={version}",
            "```",
            "",
            "**Full Changelog**: https://github.com/danieleschmidt/fair-credit-scorer-bias-mitigation/blob/main/CHANGELOG.md"
        ])
        
        return "\n".join(notes)
    
    def prepare_release(self, bump_type: str, custom_changes: Optional[List[str]] = None) -> Dict[str, Any]:
        """Prepare complete release."""
        print(f"🚀 Preparing {bump_type} release...")
        
        # Get current version and changes
        current_version = self.get_current_version()
        print(f"Current version: {current_version}")
        
        # Get changes from git if not provided
        if custom_changes is None:
            changes = self.get_git_changes_since_tag()
        else:
            changes = custom_changes
        
        if not changes:
            changes = ["Bug fixes and improvements"]
        
        print(f"Found {len(changes)} changes")
        
        # Bump version
        new_version = self.bump_version(bump_type)
        print(f"New version: {new_version}")
        
        # Update changelog
        self.update_changelog(new_version, changes)
        print("Updated CHANGELOG.md")
        
        # Generate release notes
        release_notes = self.generate_release_notes(new_version, changes)
        
        return {
            "version": new_version,
            "previous_version": current_version,
            "changes": changes,
            "release_notes": release_notes,
            "files_updated": [
                str(self.pyproject_path),
                str(self.changelog_path),
                str(self.repo_path / "README.md")
            ]
        }


def main():
    """Main release automation entry point."""
    parser = argparse.ArgumentParser(description="Automated release management")
    parser.add_argument("action", choices=["prepare", "tag", "build", "full"],
                       help="Release action to perform")
    parser.add_argument("--bump", choices=["major", "minor", "patch"], 
                       default="patch", help="Version bump type")
    parser.add_argument("--changes", nargs="*", help="Custom change descriptions")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--output", help="Output release info to file")
    
    args = parser.parse_args()
    
    try:
        manager = ReleaseManager()
        
        if args.action in ["prepare", "full"]:
            release_info = manager.prepare_release(args.bump, args.changes)
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(release_info, f, indent=2)
            
            print(f"\n✅ Release {release_info['version']} prepared")
            print(f"📝 Updated files: {', '.join(release_info['files_updated'])}")
            
            if args.action == "full":
                if not args.dry_run:
                    manager.create_git_tag(release_info['version'])
                    dist_dir = manager.build_package()
                    print(f"📦 Package built in {dist_dir}")
                else:
                    print("🔍 Dry run - would create tag and build package")
        
        elif args.action == "tag":
            version = manager.get_current_version()
            if not args.dry_run:
                manager.create_git_tag(version)
            else:
                print(f"🔍 Dry run - would create tag v{version}")
        
        elif args.action == "build":
            if not args.dry_run:
                dist_dir = manager.build_package()
                print(f"📦 Package built in {dist_dir}")
            else:
                print("🔍 Dry run - would build package")
        
        print("\n🎉 Release automation completed successfully!")
        
    except Exception as e:
        print(f"❌ Release automation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()