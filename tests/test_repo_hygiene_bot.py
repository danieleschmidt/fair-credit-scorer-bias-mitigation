"""
Test suite for the Repository Hygiene Bot

Tests the core functionality of the automated DevSecOps repository hygiene system.
"""

import pytest
import json
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

from src.repo_hygiene_bot import (
    GitHubAPI,
    RepoHygieneBot,
    RepoMetrics
)


class TestGitHubAPI:
    """Test the GitHub API client functionality."""
    
    @pytest.fixture
    def github_api(self):
        """Create a GitHub API instance for testing."""
        return GitHubAPI("fake_token", "test_user")
    
    def test_initialization(self, github_api):
        """Test GitHub API client initialization."""
        assert github_api.token == "fake_token"
        assert github_api.github_user == "test_user"
        assert "Authorization" in github_api.session.headers
        assert github_api.session.headers["Authorization"] == "token fake_token"
    
    @patch('requests.Session.get')
    def test_get_repositories_success(self, mock_get, github_api):
        """Test successful repository enumeration."""
        # Mock API response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = [
            {
                "name": "test-repo",
                "archived": False,
                "fork": False,
                "is_template": False,
                "disabled": False
            }
        ]
        mock_get.return_value = mock_response
        
        repos = github_api.get_repositories()
        
        assert len(repos) == 1
        assert repos[0]["name"] == "test-repo"
        mock_get.assert_called_once()
    
    @patch('requests.Session.get')
    def test_get_repositories_filters_excluded(self, mock_get, github_api):
        """Test repository filtering logic."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = [
            {"name": "active-repo", "archived": False, "fork": False, "is_template": False, "disabled": False},
            {"name": "archived-repo", "archived": True, "fork": False, "is_template": False, "disabled": False},
            {"name": "fork-repo", "archived": False, "fork": True, "is_template": False, "disabled": False},
            {"name": "template-repo", "archived": False, "fork": False, "is_template": True, "disabled": False},
            {"name": "disabled-repo", "archived": False, "fork": False, "is_template": False, "disabled": True},
        ]
        mock_get.return_value = mock_response
        
        repos = github_api.get_repositories()
        
        assert len(repos) == 1
        assert repos[0]["name"] == "active-repo"
    
    @patch('requests.Session.patch')
    def test_update_repository_metadata(self, mock_patch, github_api):
        """Test repository metadata updates."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_patch.return_value = mock_response
        
        result = github_api.update_repository_metadata(
            "test-repo",
            description="Test description",
            homepage="https://example.com"
        )
        
        assert result is True
        mock_patch.assert_called_once()
        call_args = mock_patch.call_args
        assert "description" in call_args[1]["json"]
        assert "homepage" in call_args[1]["json"]
    
    @patch('requests.Session.get')
    def test_get_file_content_success(self, mock_get, github_api):
        """Test successful file content retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "content": "VGVzdCBjb250ZW50",  # Base64 for "Test content"
            "encoding": "base64"
        }
        mock_get.return_value = mock_response
        
        content = github_api.get_file_content("test-repo", "README.md")
        
        assert content == "Test content"
    
    @patch('requests.Session.get')
    def test_get_file_content_not_found(self, mock_get, github_api):
        """Test file content retrieval when file doesn't exist."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        content = github_api.get_file_content("test-repo", "NONEXISTENT.md")
        
        assert content is None
    
    @patch('requests.Session.put')
    def test_create_or_update_file(self, mock_put, github_api):
        """Test file creation/update."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_put.return_value = mock_response
        
        # Mock get_file_sha to return None (new file)
        with patch.object(github_api, 'get_file_sha', return_value=None):
            result = github_api.create_or_update_file(
                "test-repo",
                "test.txt", 
                "Test content",
                "Add test file"
            )
        
        assert result is True
        mock_put.assert_called_once()


class TestRepoHygieneBot:
    """Test the main repository hygiene bot functionality."""
    
    @pytest.fixture
    def mock_github_api(self):
        """Create a mock GitHub API for testing."""
        return Mock(spec=GitHubAPI)
    
    @pytest.fixture
    def hygiene_bot(self, mock_github_api):
        """Create a hygiene bot instance for testing."""
        bot = RepoHygieneBot("fake_token", "test_user")
        bot.github_api = mock_github_api
        return bot
    
    def test_initialization(self, hygiene_bot):
        """Test hygiene bot initialization."""
        assert hygiene_bot.github_user == "test_user"
        assert len(hygiene_bot.standard_topics) == 5
        assert "llmops" in hygiene_bot.standard_topics
        assert "LICENSE" in hygiene_bot.templates
    
    def test_check_description_missing(self, hygiene_bot):
        """Test description checking when missing."""
        repo = {"description": None}
        changes = []
        
        result = hygiene_bot._check_description(repo, changes)
        
        assert result is True
        assert len(changes) == 1
        assert "description" in changes[0].lower()
    
    def test_check_description_present(self, hygiene_bot):
        """Test description checking when present."""
        repo = {"description": "A test repository"}
        changes = []
        
        result = hygiene_bot._check_description(repo, changes)
        
        assert result is False
        assert len(changes) == 0
    
    def test_check_topics_insufficient(self, hygiene_bot):
        """Test topic checking with insufficient topics."""
        repo = {"topics": ["python"]}
        changes = []
        
        result = hygiene_bot._check_topics(repo, changes)
        
        assert result is True
        assert len(changes) == 1
    
    def test_check_topics_sufficient(self, hygiene_bot):
        """Test topic checking with sufficient topics."""
        repo = {"topics": ["python", "machine-learning", "ai", "data-science", "research", "automation"]}
        changes = []
        
        result = hygiene_bot._check_topics(repo, changes)
        
        # Should still want to add standard topics
        assert result is True
    
    def test_should_archive_main_project_old(self, hygiene_bot):
        """Test archival logic for old Main-Project."""
        repo = {"name": "Main-Project"}
        
        # Mock GitHub API call for commits
        old_date = datetime.now(timezone.utc).replace(year=2022).isoformat()
        hygiene_bot.github_api.session.get.return_value.status_code = 200
        hygiene_bot.github_api.session.get.return_value.json.return_value = [
            {"commit": {"committer": {"date": old_date}}}
        ]
        
        result = hygiene_bot._should_archive_main_project(repo)
        
        assert result is True
    
    def test_should_archive_main_project_recent(self, hygiene_bot):
        """Test archival logic for recent Main-Project."""
        repo = {"name": "Main-Project"}
        
        # Mock GitHub API call for commits
        recent_date = datetime.now(timezone.utc).isoformat()
        hygiene_bot.github_api.session.get.return_value.status_code = 200
        hygiene_bot.github_api.session.get.return_value.json.return_value = [
            {"commit": {"committer": {"date": recent_date}}}
        ]
        
        result = hygiene_bot._should_archive_main_project(repo)
        
        assert result is False
    
    def test_should_archive_main_project_wrong_name(self, hygiene_bot):
        """Test archival logic for non-Main-Project."""
        repo = {"name": "Other-Project"}
        
        result = hygiene_bot._should_archive_main_project(repo)
        
        assert result is False
    
    @patch('os.makedirs')
    @patch('builtins.open')
    @patch('json.dump')
    def test_save_metrics(self, mock_json_dump, mock_open, mock_makedirs, hygiene_bot):
        """Test metrics saving functionality."""
        # Add some test metrics
        test_metric = RepoMetrics(
            repo="test-repo",
            description_set=True,
            topics_count=5,
            license_exists=True,
            code_scanning=True,
            dependabot=True,
            scorecard=True,
            sbom_workflow=True,
            community_files_count=4,
            readme_badges_count=3,
            readme_sections_count=2,
            last_commit_days_ago=10
        )
        hygiene_bot.metrics = [test_metric]
        
        # Mock file operations
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        hygiene_bot._save_metrics()
        
        mock_makedirs.assert_called_once_with('metrics', exist_ok=True)
        mock_open.assert_called_once_with('metrics/profile_hygiene.json', 'w')
        mock_json_dump.assert_called_once()
    
    def test_ensure_readme_badges_missing(self, hygiene_bot):
        """Test README badge injection when badges are missing."""
        readme_content = "# Test Repo\n\nThis is a test repository."
        changes = []
        metrics = RepoMetrics(
            repo="test-repo", description_set=True, topics_count=5,
            license_exists=True, code_scanning=True, dependabot=True,
            scorecard=True, sbom_workflow=True, community_files_count=0,
            readme_badges_count=0, readme_sections_count=0, last_commit_days_ago=0
        )
        
        # Mock file operations
        hygiene_bot.github_api.get_file_content.return_value = readme_content
        hygiene_bot.github_api.create_or_update_file.return_value = True
        
        hygiene_bot._ensure_readme_badges("test-repo", "test-branch", changes, metrics)
        
        assert len(changes) > 0
        assert any("badges" in change.lower() for change in changes)
        assert metrics.readme_badges_count > 0
    
    def test_ensure_readme_sections_missing(self, hygiene_bot):
        """Test README section enforcement when sections are missing."""
        readme_content = "# Test Repo\n\nBasic content."
        changes = []
        metrics = RepoMetrics(
            repo="test-repo", description_set=True, topics_count=5,
            license_exists=True, code_scanning=True, dependabot=True,
            scorecard=True, sbom_workflow=True, community_files_count=0,
            readme_badges_count=0, readme_sections_count=0, last_commit_days_ago=0
        )
        
        # Mock file operations
        hygiene_bot.github_api.get_file_content.return_value = readme_content
        hygiene_bot.github_api.create_or_update_file.return_value = True
        
        hygiene_bot._ensure_readme_sections("test-repo", "test-branch", changes, metrics)
        
        assert len(changes) > 0
        assert metrics.readme_sections_count > 0
        assert any("sections" in change.lower() for change in changes)


class TestRepoMetrics:
    """Test the repository metrics data structure."""
    
    def test_repo_metrics_creation(self):
        """Test creating a RepoMetrics instance."""
        metrics = RepoMetrics(
            repo="test-repo",
            description_set=True,
            topics_count=5,
            license_exists=True,
            code_scanning=True,
            dependabot=True,
            scorecard=True,
            sbom_workflow=True,
            community_files_count=4,
            readme_badges_count=3,
            readme_sections_count=5,
            last_commit_days_ago=7
        )
        
        assert metrics.repo == "test-repo"
        assert metrics.description_set is True
        assert metrics.topics_count == 5
        assert metrics.license_exists is True
        assert metrics.code_scanning is True
        assert metrics.dependabot is True
        assert metrics.scorecard is True
        assert metrics.sbom_workflow is True
        assert metrics.community_files_count == 4
        assert metrics.readme_badges_count == 3
        assert metrics.readme_sections_count == 5
        assert metrics.last_commit_days_ago == 7
    
    def test_repo_metrics_serialization(self):
        """Test that RepoMetrics can be converted to dict for JSON serialization."""
        from dataclasses import asdict
        
        metrics = RepoMetrics(
            repo="test-repo",
            description_set=True,
            topics_count=5,
            license_exists=False,
            code_scanning=True,
            dependabot=False,
            scorecard=True,
            sbom_workflow=False,
            community_files_count=2,
            readme_badges_count=1,
            readme_sections_count=3,
            last_commit_days_ago=14
        )
        
        metrics_dict = asdict(metrics)
        
        assert isinstance(metrics_dict, dict)
        assert metrics_dict["repo"] == "test-repo"
        assert metrics_dict["description_set"] is True
        assert metrics_dict["license_exists"] is False


class TestIntegration:
    """Integration tests for the complete workflow."""
    
    @pytest.fixture
    def mock_repo_data(self):
        """Sample repository data for testing."""
        return {
            "name": "test-repo",
            "description": None,
            "homepage": None,
            "topics": [],
            "archived": False,
            "fork": False,
            "is_template": False,
            "disabled": False
        }
    
    @patch('src.repo_hygiene_bot.GitHubAPI')
    def test_end_to_end_workflow(self, mock_github_api_class, mock_repo_data):
        """Test the complete end-to-end workflow."""
        # Setup mocks
        mock_api = Mock()
        mock_github_api_class.return_value = mock_api
        
        # Mock repository data
        mock_api.get_repositories.return_value = [mock_repo_data]
        mock_api.get_file_content.return_value = None  # No existing files
        mock_api.create_branch.return_value = True
        mock_api.create_or_update_file.return_value = True
        mock_api.create_pull_request.return_value = {"number": 1, "html_url": "https://github.com/test/test-repo/pull/1"}
        mock_api.update_repository_metadata.return_value = True
        
        # Create bot and run
        bot = RepoHygieneBot("fake_token", "test_user")
        
        with patch.object(bot, '_save_metrics'):
            metrics = bot.process_all_repositories()
        
        # Verify operations
        assert len(metrics) == 1
        assert metrics[0].repo == "test-repo"
        
        # Verify API calls were made
        mock_api.get_repositories.assert_called_once()
        mock_api.update_repository_metadata.assert_called_once()
        mock_api.create_branch.assert_called_once()
        mock_api.create_pull_request.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])