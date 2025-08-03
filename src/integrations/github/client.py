"""
GitHub API client for repository management and automation.

Provides comprehensive GitHub API integration with rate limiting,
error handling, and batch operations for repository hygiene automation.
"""

import asyncio
import base64
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
import json

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ...logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class GitHubRepository:
    """GitHub repository information."""
    name: str
    full_name: str
    description: Optional[str]
    private: bool
    default_branch: str
    topics: List[str]
    created_at: str
    updated_at: str
    pushed_at: str
    language: Optional[str]
    size: int
    stargazers_count: int
    watchers_count: int
    forks_count: int
    open_issues_count: int
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GitHubRepository":
        """Create repository object from GitHub API response."""
        return cls(
            name=data["name"],
            full_name=data["full_name"],
            description=data.get("description"),
            private=data["private"],
            default_branch=data["default_branch"],
            topics=data.get("topics", []),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            pushed_at=data["pushed_at"],
            language=data.get("language"),
            size=data["size"],
            stargazers_count=data["stargazers_count"],
            watchers_count=data["watchers_count"],
            forks_count=data["forks_count"],
            open_issues_count=data["open_issues_count"]
        )


@dataclass
class RateLimitInfo:
    """GitHub API rate limit information."""
    limit: int
    remaining: int
    reset_timestamp: int
    used: int
    
    @property
    def reset_time(self) -> datetime:
        """Get reset time as datetime object."""
        return datetime.fromtimestamp(self.reset_timestamp)
    
    @property
    def time_until_reset(self) -> timedelta:
        """Get time until rate limit reset."""
        return self.reset_time - datetime.utcnow()


class GitHubClient:
    """
    Production-ready GitHub API client.
    
    Features:
    - Automatic rate limit handling with exponential backoff
    - Retry logic for transient errors
    - Batch operations for efficient API usage
    - Comprehensive error handling and logging
    - Support for GitHub Apps and OAuth tokens
    """
    
    def __init__(
        self,
        token: str,
        base_url: str = "https://api.github.com",
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize GitHub client.
        
        Args:
            token: GitHub token (personal access token or app token)
            base_url: GitHub API base URL
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.token = token
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        
        # Setup session with retry strategy
        self.session = requests.Session()
        
        retry_strategy = Retry(
            total=max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "OPTIONS"],
            backoff_factor=1
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set default headers
        self.session.headers.update({
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "fair-credit-scorer-bot/1.0"
        })
        
        # Rate limiting
        self._rate_limit_info: Optional[RateLimitInfo] = None
        self._last_request_time = 0.0
        
        logger.info("GitHubClient initialized")
    
    def get_rate_limit(self) -> RateLimitInfo:
        """Get current rate limit information."""
        response = self._make_request("GET", "/rate_limit")
        rate_data = response["rate"]
        
        return RateLimitInfo(
            limit=rate_data["limit"],
            remaining=rate_data["remaining"],
            reset_timestamp=rate_data["reset"],
            used=rate_data["used"]
        )
    
    def get_user_repositories(
        self,
        username: Optional[str] = None,
        per_page: int = 100,
        repository_type: str = "all"
    ) -> List[GitHubRepository]:
        """
        Get repositories for a user.
        
        Args:
            username: GitHub username (current user if None)
            per_page: Number of repositories per page
            repository_type: Type of repositories (all, owner, member)
            
        Returns:
            List of GitHubRepository objects
        """
        if username:
            endpoint = f"/users/{username}/repos"
        else:
            endpoint = "/user/repos"
        
        params = {
            "per_page": per_page,
            "type": repository_type,
            "sort": "updated",
            "direction": "desc"
        }
        
        repositories = []
        page = 1
        
        while True:
            params["page"] = page
            response = self._make_request("GET", endpoint, params=params)
            
            if not response:  # Empty response means no more pages
                break
            
            for repo_data in response:
                repositories.append(GitHubRepository.from_dict(repo_data))
            
            page += 1
            
            # Break if we got less than requested (last page)
            if len(response) < per_page:
                break
        
        logger.info(f"Retrieved {len(repositories)} repositories")
        return repositories
    
    def get_repository(self, owner: str, repo: str) -> GitHubRepository:
        """
        Get information about a specific repository.
        
        Args:
            owner: Repository owner
            repo: Repository name
            
        Returns:
            GitHubRepository object
        """
        endpoint = f"/repos/{owner}/{repo}"
        response = self._make_request("GET", endpoint)
        
        return GitHubRepository.from_dict(response)
    
    def update_repository(
        self,
        owner: str,
        repo: str,
        description: Optional[str] = None,
        homepage: Optional[str] = None,
        topics: Optional[List[str]] = None,
        private: Optional[bool] = None,
        has_issues: Optional[bool] = None,
        has_projects: Optional[bool] = None,
        has_wiki: Optional[bool] = None
    ) -> GitHubRepository:
        """
        Update repository settings.
        
        Args:
            owner: Repository owner
            repo: Repository name
            description: Repository description
            homepage: Repository homepage URL
            topics: Repository topics
            private: Whether repository is private
            has_issues: Whether issues are enabled
            has_projects: Whether projects are enabled
            has_wiki: Whether wiki is enabled
            
        Returns:
            Updated GitHubRepository object
        """
        endpoint = f"/repos/{owner}/{repo}"
        
        data = {}
        if description is not None:
            data["description"] = description
        if homepage is not None:
            data["homepage"] = homepage
        if private is not None:
            data["private"] = private
        if has_issues is not None:
            data["has_issues"] = has_issues
        if has_projects is not None:
            data["has_projects"] = has_projects
        if has_wiki is not None:
            data["has_wiki"] = has_wiki
        
        # Update topics separately (different endpoint)
        if topics is not None:
            self.update_repository_topics(owner, repo, topics)
        
        if data:
            response = self._make_request("PATCH", endpoint, json=data)
            return GitHubRepository.from_dict(response)
        else:
            return self.get_repository(owner, repo)
    
    def update_repository_topics(self, owner: str, repo: str, topics: List[str]) -> List[str]:
        """
        Update repository topics.
        
        Args:
            owner: Repository owner
            repo: Repository name
            topics: List of topics
            
        Returns:
            Updated list of topics
        """
        endpoint = f"/repos/{owner}/{repo}/topics"
        
        data = {"names": topics}
        response = self._make_request("PUT", endpoint, json=data)
        
        return response["names"]
    
    def create_file(
        self,
        owner: str,
        repo: str,
        path: str,
        content: str,
        message: str,
        branch: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a file in repository.
        
        Args:
            owner: Repository owner
            repo: Repository name
            path: File path
            content: File content
            message: Commit message
            branch: Target branch (default branch if None)
            
        Returns:
            GitHub API response
        """
        endpoint = f"/repos/{owner}/{repo}/contents/{path}"
        
        # Encode content to base64
        content_encoded = base64.b64encode(content.encode('utf-8')).decode('utf-8')
        
        data = {
            "message": message,
            "content": content_encoded
        }
        
        if branch:
            data["branch"] = branch
        
        return self._make_request("PUT", endpoint, json=data)
    
    def update_file(
        self,
        owner: str,
        repo: str,
        path: str,
        content: str,
        message: str,
        sha: str,
        branch: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update an existing file in repository.
        
        Args:
            owner: Repository owner
            repo: Repository name
            path: File path
            content: New file content
            message: Commit message
            sha: Current file SHA
            branch: Target branch
            
        Returns:
            GitHub API response
        """
        endpoint = f"/repos/{owner}/{repo}/contents/{path}"
        
        content_encoded = base64.b64encode(content.encode('utf-8')).decode('utf-8')
        
        data = {
            "message": message,
            "content": content_encoded,
            "sha": sha
        }
        
        if branch:
            data["branch"] = branch
        
        return self._make_request("PUT", endpoint, json=data)
    
    def get_file(self, owner: str, repo: str, path: str, ref: Optional[str] = None) -> Dict[str, Any]:
        """
        Get file contents from repository.
        
        Args:
            owner: Repository owner
            repo: Repository name
            path: File path
            ref: Git reference (branch, tag, commit)
            
        Returns:
            File information including content
        """
        endpoint = f"/repos/{owner}/{repo}/contents/{path}"
        
        params = {}
        if ref:
            params["ref"] = ref
        
        return self._make_request("GET", endpoint, params=params)
    
    def create_pull_request(
        self,
        owner: str,
        repo: str,
        title: str,
        head: str,
        base: str,
        body: Optional[str] = None,
        draft: bool = False
    ) -> Dict[str, Any]:
        """
        Create a pull request.
        
        Args:
            owner: Repository owner
            repo: Repository name
            title: Pull request title
            head: Source branch
            base: Target branch
            body: Pull request body
            draft: Whether PR is a draft
            
        Returns:
            Pull request information
        """
        endpoint = f"/repos/{owner}/{repo}/pulls"
        
        data = {
            "title": title,
            "head": head,
            "base": base,
            "draft": draft
        }
        
        if body:
            data["body"] = body
        
        return self._make_request("POST", endpoint, json=data)
    
    def create_issue(
        self,
        owner: str,
        repo: str,
        title: str,
        body: Optional[str] = None,
        labels: Optional[List[str]] = None,
        assignees: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create an issue.
        
        Args:
            owner: Repository owner
            repo: Repository name
            title: Issue title
            body: Issue body
            labels: Issue labels
            assignees: Issue assignees
            
        Returns:
            Issue information
        """
        endpoint = f"/repos/{owner}/{repo}/issues"
        
        data = {"title": title}
        
        if body:
            data["body"] = body
        if labels:
            data["labels"] = labels
        if assignees:
            data["assignees"] = assignees
        
        return self._make_request("POST", endpoint, json=data)
    
    def enable_vulnerability_alerts(self, owner: str, repo: str) -> bool:
        """
        Enable vulnerability alerts for repository.
        
        Args:
            owner: Repository owner
            repo: Repository name
            
        Returns:
            Success status
        """
        endpoint = f"/repos/{owner}/{repo}/vulnerability-alerts"
        
        try:
            self._make_request("PUT", endpoint)
            return True
        except Exception as e:
            logger.error(f"Failed to enable vulnerability alerts: {e}")
            return False
    
    def enable_automated_security_fixes(self, owner: str, repo: str) -> bool:
        """
        Enable automated security fixes for repository.
        
        Args:
            owner: Repository owner
            repo: Repository name
            
        Returns:
            Success status
        """
        endpoint = f"/repos/{owner}/{repo}/automated-security-fixes"
        
        try:
            self._make_request("PUT", endpoint)
            return True
        except Exception as e:
            logger.error(f"Failed to enable automated security fixes: {e}")
            return False
    
    def batch_update_repositories(
        self,
        repositories: List[Tuple[str, str]],
        update_func,
        max_concurrent: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Batch update multiple repositories with rate limiting.
        
        Args:
            repositories: List of (owner, repo) tuples
            update_func: Function to apply to each repository
            max_concurrent: Maximum concurrent operations
            
        Returns:
            List of update results
        """
        async def update_with_semaphore(semaphore, owner, repo):
            async with semaphore:
                try:
                    result = update_func(owner, repo)
                    return {"owner": owner, "repo": repo, "success": True, "result": result}
                except Exception as e:
                    logger.error(f"Failed to update {owner}/{repo}: {e}")
                    return {"owner": owner, "repo": repo, "success": False, "error": str(e)}
        
        async def batch_process():
            semaphore = asyncio.Semaphore(max_concurrent)
            tasks = [
                update_with_semaphore(semaphore, owner, repo)
                for owner, repo in repositories
            ]
            return await asyncio.gather(*tasks)
        
        return asyncio.run(batch_process())
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """
        Make authenticated request to GitHub API with rate limiting.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            json: JSON payload
            **kwargs: Additional request arguments
            
        Returns:
            API response data
        """
        # Check rate limiting
        self._handle_rate_limiting()
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=json,
                timeout=self.timeout,
                **kwargs
            )
            
            # Update rate limit info from headers
            self._update_rate_limit_from_headers(response.headers)
            
            response.raise_for_status()
            
            # Return JSON if available, otherwise empty dict
            try:
                return response.json()
            except ValueError:
                return {}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"GitHub API request failed: {e}")
            raise
    
    def _handle_rate_limiting(self):
        """Handle GitHub API rate limiting."""
        if self._rate_limit_info and self._rate_limit_info.remaining == 0:
            sleep_time = self._rate_limit_info.time_until_reset.total_seconds()
            if sleep_time > 0:
                logger.warning(f"Rate limit exceeded, sleeping for {sleep_time} seconds")
                time.sleep(sleep_time + 1)  # Add 1 second buffer
        
        # Respect rate limiting with delay between requests
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        min_interval = 0.1  # 100ms between requests
        
        if time_since_last < min_interval:
            time.sleep(min_interval - time_since_last)
        
        self._last_request_time = time.time()
    
    def _update_rate_limit_from_headers(self, headers: Dict[str, str]):
        """Update rate limit info from response headers."""
        if 'X-RateLimit-Remaining' in headers:
            self._rate_limit_info = RateLimitInfo(
                limit=int(headers.get('X-RateLimit-Limit', 0)),
                remaining=int(headers.get('X-RateLimit-Remaining', 0)),
                reset_timestamp=int(headers.get('X-RateLimit-Reset', 0)),
                used=int(headers.get('X-RateLimit-Used', 0))
            )


# CLI interface
def main():
    """CLI interface for GitHub client operations."""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="GitHub Client CLI")
    parser.add_argument("command", choices=["repos", "repo-info", "update-repo", "create-pr"])
    parser.add_argument("--token", help="GitHub token", default=os.getenv("GITHUB_TOKEN"))
    parser.add_argument("--owner", help="Repository owner")
    parser.add_argument("--repo", help="Repository name")
    parser.add_argument("--description", help="Repository description")
    parser.add_argument("--topics", nargs="+", help="Repository topics")
    
    args = parser.parse_args()
    
    if not args.token:
        print("Error: GitHub token required (set GITHUB_TOKEN env var or use --token)")
        return
    
    client = GitHubClient(args.token)
    
    if args.command == "repos":
        repositories = client.get_user_repositories()
        print(f"Found {len(repositories)} repositories:")
        for repo in repositories[:10]:  # Show first 10
            print(f"  {repo.full_name} - {repo.description or 'No description'}")
    
    elif args.command == "repo-info":
        if not args.owner or not args.repo:
            print("Error: --owner and --repo required")
            return
        
        repo = client.get_repository(args.owner, args.repo)
        print(f"Repository: {repo.full_name}")
        print(f"Description: {repo.description}")
        print(f"Language: {repo.language}")
        print(f"Stars: {repo.stargazers_count}")
        print(f"Topics: {', '.join(repo.topics)}")
    
    elif args.command == "update-repo":
        if not args.owner or not args.repo:
            print("Error: --owner and --repo required")
            return
        
        repo = client.update_repository(
            args.owner,
            args.repo,
            description=args.description,
            topics=args.topics
        )
        print(f"Updated repository: {repo.full_name}")


if __name__ == "__main__":
    main()